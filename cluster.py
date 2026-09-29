"""Split a model over any set of boards: which layers go where, and how
each pair of neighbours talks.

The boards run as GALS stages (gals.py): each on its own clock, a
contiguous range of layers each, passing the hidden state on and the
chosen token back. This decides the ranges and the links from what
boards.py knows about each board:

  speed     bytes of weights a second it can stream, the smaller of its
            DDR's sustained rate and what the design consumes at its
            clock: a byte a lane each core cycle (16 lanes on the Zybo,
            32 on the ZC706), over the bus cycles a core cycle measured
            through that package's registers
  capacity  DDR left for weights once the program, the constants, its
            share of the KV cache and, on the first and last stage, the
            tied embedding and head table are placed
  link      Ethernet between two boards whose ARMs can run it, otherwise
            the fabric UART, which every FPGA can build

Two ways to split. "balanced" gives each board layers in proportion to
its speed, so every stage takes about as long: the throughput split,
for several streams in flight. "fast" fills the fastest boards first
and uses as few as it can: a single stream's latency is the sum of its
stages, so hops only cost it. A board that cannot hold a layer (no DRAM
the design can reach) gets none.

Run: python3 cluster.py zc706 zybo_z7_20 [--model qwen3] [--mode fast]
     FPGAI_QWEN=qwen3 python3 cluster.py zc706 zybo_z7_20 --package build_cluster
"""
import argparse
import json
import math
import os

import boards

ROOT = os.path.dirname(os.path.abspath(__file__))
BUS_PER_CORE = 1.18      # measured, 24 layers through the Zybo's registers
BASE = 0x08000000        # below it, the ARM's program
UART_BAUD = 6.25e6       # board to board on two wires: clk/8 at 50 MHz
ETH_BYTES_S = 100e6      # UDP over Gigabit Ethernet through lwIP, sustained
LINK_LATENCY_S = {"ethernet": 200e-6, "uart": 20e-6}

MODELS = {
    # The shapes that set the bytes; the checkpoint's config.json has them.
    "qwen2.5": dict(D=896, F=4864, H=14, KV=2, hd=64, NL=24, V=151936, qkn=False),
    "qwen3": dict(D=1024, F=3072, H=16, KV=8, hd=128, NL=28, V=151936, qkn=True),
}


def shape_bytes(m, seq_len=256):
    """Per layer: int8 weights, 64-bit column constants, the 16-bit KV
    cache for seq_len positions. The tied table: int8 rows plus two
    constant words a token (the head's and the embedding's)."""
    D, F, H, KV, hd = m["D"], m["F"], m["H"], m["KV"], m["hd"]
    rows = H * hd + 2 * KV * hd + D + 2 * F + D
    w = D * (H * hd + 2 * KV * hd) + H * hd * D + 3 * D * F
    return dict(layer=w + rows * 8 + 2 * KV * hd * seq_len * 2,
                layer_weights=w, table=m["V"] * m["D"] + 2 * m["V"] * 8,
                hidden_msg=2 * D + 15)


def proj_cycles(rows, depth, lanes):
    """One projection through the multi-lane block: every group of lanes
    columns reads the input once, five cycles to turn the group round;
    then the last group's sums drain through the one requantizer, a cycle
    a lane, and 16 more start and finish the block (32 in all at 16 lanes,
    48 at 32)."""
    return -(-rows // lanes) * (depth + 5) + lanes + 16


def layer_cycles(m, lanes, n):
    """Core cycles for one layer at context length n, state by state, as
    the sequencer (qwen_full.py) runs them: the seven projections, the two
    norms, RoPE on q and k, each head's scores, softmax and weighted sum,
    the head norms of a model that has them, the SiLU gate and the two
    residual adds. Measured against its simulation it is within 0.1% on
    Qwen2.5-0.5B and Qwen3-0.6B (RESULTS.md)."""
    D, F, H, KV, hd = m["D"], m["F"], m["H"], m["KV"], m["hd"]
    la = min(lanes, hd)
    c = (proj_cycles(H * hd, D, lanes) + 2 * proj_cycles(KV * hd, D, lanes)
         + proj_cycles(D, H * hd, lanes) + 2 * proj_cycles(F, D, lanes)
         + proj_cycles(D, F, lanes))
    c += 2 * (2 * D + 28)                               # the two RMSNorms
    c += (H * hd // 2 + 7) + (KV * hd // 2 + 7)         # RoPE on q, k
    c += H * (hd + 1)                                   # each head's q in
    c += H * (-(-n // la) * hd + 3 * n + (hd // la) * n + hd + 43)
    if m.get("qkn"):
        c += (H + KV) * (2 * hd + 28)                   # Qwen3's head norms
    c += F + 23 + 2 * (D + 5)                           # SiLU gate, residuals
    return c


def head_cycles(m, lanes):
    """The final norm and the tied head, in chunks of the widest matrix."""
    chunk = max(m["D"], m["F"])
    V, D = m["V"], m["D"]
    return 2 * D + 28 + sum(proj_cycles(min(chunk, V - c0), D, lanes)
                            for c0 in range(0, V, chunk))


def stage_cycles(m, lanes, n_layers, n, emb, head):
    """One position through a stage: its layers, the embedding lookup on
    the first stage, the head on the last."""
    c = n_layers * layer_cycles(m, lanes, n)
    if emb:
        c += m["D"] + 12
    if head:
        c += head_cycles(m, lanes)
    return c


def _even(m, T):
    return dict(H=m["H"] // T, KV=m["KV"] // T, F=m["F"] // T, D=m["D"] // T)


def tp_segments(m, lanes, share, n):
    """One layer on one rank of a split of the weights (tp.py), as the
    core cycles it computes between its four gathers, not the ones it
    waits: up to the context's (the residual add, the first norm, its own
    q, k and v heads, their norms and RoPE, and its heads' attention), up
    to o's output (its share of o's columns over their full depth), up to
    the gated product's (the second norm and its share of d_ff), and up
    to down's output. share: this rank's H, KV, F and D. Each ends in a
    cycle to ask for the gather."""
    D, F, H, hd = m["D"], m["F"], m["H"], m["hd"]
    Hl, KVl, Fl, Dl = share["H"], share["KV"], share["F"], share["D"]
    la = min(lanes, hd)
    s1 = (D + 5) + (2 * D + 28) + proj_cycles(Hl * hd, D, lanes) \
        + 2 * proj_cycles(KVl * hd, D, lanes) + (Hl * hd // 2 + 7) + (KVl * hd // 2 + 7) \
        + Hl * (hd + 1) + Hl * (-(-n // la) * hd + 3 * n + (hd // la) * n + hd + 43) + 1
    if m.get("qkn"):
        s1 += (Hl + KVl) * (2 * hd + 28)
    s2 = proj_cycles(Dl, H * hd, lanes) + 1
    s3 = (D + 5) + (2 * D + 28) + 2 * proj_cycles(Fl, D, lanes) + Fl + 23 + 1
    s4 = proj_cycles(Dl, F, lanes) + 1
    return [s1, s2, s3, s4]


def tp_layer_cycles(m, lanes, T, n, share=None):
    """The same layer's cycles in all, even shares unless share."""
    return sum(tp_segments(m, lanes, share or _even(m, T), n))


def _chunks(m):
    chunk = max(m["D"], m["F"])
    return chunk, -(-m["V"] // chunk)


def tp_head_cycles(m, lanes, T, rank, hk=None):
    """The final norm on every rank, then this rank's run of the head's
    chunks, hk = (first, last), even runs unless given (a rank past the
    last chunk runs none), and its gather."""
    chunk, nck = _chunks(m)
    if hk is None:
        per = -(-nck // T)
        hk = (rank * per, min(nck - 1, rank * per + per - 1))
    return 2 * m["D"] + 28 + 1 + sum(proj_cycles(min(chunk, m["V"] - k * chunk), m["D"], lanes)
                                     for k in range(hk[0], hk[1] + 1))


def tp_rank_cycles(m, lanes, T, rank, n_layers, n, head, share=None, hk=None):
    """One position on one rank: the embedding, its layers, the head."""
    return (m["D"] + 12 + n_layers * tp_layer_cycles(m, lanes, T, n, share)
            + (tp_head_cycles(m, lanes, T, rank, hk) if head else 0))


def seconds_per_cycle(name):
    """A core cycle on this board, in seconds: its bus clock over the bus
    cycles a core cycle its package measured."""
    pk = boards.PACKAGES.get(name, {})
    return pk.get("bus_per_core", BUS_PER_CORE) / (pk.get("fpgai_mhz", 50) * 1e6)


# The ARM's side of a gather: one GADDR/GDATA access through GP0, which
# waits for the core's edges. An estimate: no board has timed one yet.
ARM_ACCESS_S = 0.25e-6


def tp_plan(names, m, ctx=128, even=False):
    """Shares of a split of the weights over these boards, in rank order.
    Each rank holds whole KV heads (with their query heads), and d_ff and
    d_model columns in multiples of every rank's lanes; the head's chunks
    go in contiguous runs, so a tie still goes to the lower token. The
    shares start in proportion to each board's rate and then move a unit
    at a time while that shortens the layer, whose time is, gather by
    gather, the slowest rank's; the head's runs likewise. even: the
    equal shares, for comparison. Returns the part (qwen_full.tp_share),
    each rank's seconds a layer and for the head, and the group's
    seconds a token with the ARMs' gathers estimated."""
    T = len(names)
    lanes = [boards.PACKAGES[n]["lanes"] for n in names]
    spc = [seconds_per_cycle(n) for n in names]
    unit = 1
    for N in lanes:
        unit = unit * N // math.gcd(unit, N)
    grp = m["H"] // m["KV"]
    chunk, nck = _chunks(m)

    def layer_s(kv, f, d):
        seg = [[c * spc[r] for c in tp_segments(m, lanes[r], dict(
            H=grp * kv[r], KV=kv[r], F=f[r] * unit, D=d[r] * unit), ctx)] for r in range(T)]
        return sum(max(seg[r][k] for r in range(T)) for k in range(4)), seg

    def head_s(c):
        hk, k0 = [], 0
        for r in range(T):
            hk.append((k0, k0 + c[r] - 1))
            k0 += c[r]
        t = [tp_head_cycles(m, lanes[r], T, r, hk[r]) * spc[r] for r in range(T)]
        return max(t), t, hk

    def apportion(total, lo):
        rate = [lanes[r] / spc[r] for r in range(T)]
        want = [total * x / sum(rate) for x in rate]
        c = [max(lo, int(w)) for w in want]
        while sum(c) > total:
            c[max(range(T), key=lambda r: c[r] - want[r] if c[r] > lo else -1e9)] -= 1
        while sum(c) < total:
            c[max(range(T), key=lambda r: want[r] - c[r])] += 1
        return c

    def improve(cs, cost, lo):
        best = cost(cs)
        moved = True
        while moved:
            moved = False
            for a in range(T):
                for b in range(T):
                    if a == b or cs[a] <= lo:
                        continue
                    t = list(cs)
                    t[a] -= 1
                    t[b] += 1
                    c = cost(t)
                    if c < best - 1e-12:
                        cs, best, moved = t, c, True
        return cs

    if m["D"] % unit or m["F"] % unit or m["D"] // unit < T or m["F"] // unit < T \
            or m["KV"] < T or any(m["hd"] % N for N in lanes):
        raise ValueError("this shape does not split %d ways at %s lanes"
                         % (T, "/".join(map(str, lanes))))
    if even:
        if m["KV"] % T or (m["D"] // unit) % T or (m["F"] // unit) % T:
            raise ValueError("this shape does not split evenly %d ways" % T)
        kv, f, d = [m["KV"] // T] * T, [m["F"] // unit // T] * T, [m["D"] // unit // T] * T
        per = -(-nck // T)
        c = [max(0, min(nck, (r + 1) * per) - r * per) for r in range(T)]
    else:
        kv, f, d = (apportion(m["KV"], 1), apportion(m["F"] // unit, 1),
                    apportion(m["D"] // unit, 1))
        for _ in range(3):
            kv = improve(kv, lambda x: layer_s(x, f, d)[0], 1)
            f = improve(f, lambda x: layer_s(kv, x, d)[0], 1)
            d = improve(d, lambda x: layer_s(kv, f, x)[0], 1)
        c = improve(apportion(nck, 0), lambda x: head_s(x)[0], 0)
    lay, seg = layer_s(kv, f, d)
    hs, ht, hk = head_s(c)
    part = dict(kv=kv, f=[x * unit for x in f], d=[x * unit for x in d], hk=[list(x) for x in hk])
    # The gathers, four a layer and the head's: each rank reads its slice
    # through the registers and writes everyone else's, and the slices
    # cross the Ethernet once.
    hd = m["hd"]
    vecs = [[grp * kv[r] * hd for r in range(T)], [x * unit for x in d],
            [x * unit for x in f], [x * unit for x in d]]
    g = 0.0
    for sl in vecs:
        # Every rank reads its own words and writes all the others'.
        tot = sum(sl)
        g += tot * ARM_ACCESS_S + LINK_LATENCY_S["ethernet"] \
            + max(2 * (tot - x) for x in sl) / ETH_BYTES_S
    head_g = 3 * T * ARM_ACCESS_S + LINK_LATENCY_S["ethernet"]
    tok = m["NL"] * (lay + g) + hs + head_g + (m["D"] + 12) * max(spc)
    return dict(part=part, lanes=lanes, layer_seconds=lay, gather_seconds=g,
                rank_layer_seconds=[sum(x) for x in seg], head_seconds=ht,
                seconds_per_token=tok, tokens_per_s=1.0 / tok)


def stage_seconds(name, m, n_layers, n, emb, head):
    """The same in seconds on that board: its core's cycles at the bus
    clock times the bus cycles a core cycle its package measured, and no
    faster than its DDR can deliver the stage's weights."""
    pk = boards.PACKAGES.get(name, {})
    mhz = pk.get("fpgai_mhz", 50)
    t = stage_cycles(m, pk.get("lanes", 16), n_layers, n, emb, head) \
        * pk.get("bus_per_core", BUS_PER_CORE) / (mhz * 1e6)
    sb = shape_bytes(m)
    byts = n_layers * sb["layer_weights"] + (m["V"] * m["D"] if head else 0)
    return max(t, byts / (boards.BOARDS[name]["mem_gbytes_per_s"] * 1e9))


def speed(name):
    """Weight bytes a second the board streams through this design."""
    b, pk = boards.BOARDS[name], boards.PACKAGES.get(name, {})
    mhz = pk.get("fpgai_mhz", 50)
    design = (pk.get("lanes", 16) * mhz * 1e6
              / pk.get("bus_per_core", BUS_PER_CORE))
    return min(b["mem_gbytes_per_s"] * 1e9, design)


def capacity(name):
    pk = boards.PACKAGES.get(name)
    if not pk or pk.get("ddr") is None:
        return 0
    return pk["ddr_bytes"] - BASE - (8 << 20)     # vocab text and margin


def link_seconds(kind, nbytes):
    rate = ETH_BYTES_S if kind == "ethernet" else UART_BAUD / 10.0
    return LINK_LATENCY_S[kind] + nbytes / rate


def plan(names, model="qwen3", mode="balanced", shape=None, ctx=128):
    """Stages in chain order: [{board, layers, emb, head, seconds}], the
    links between them, and the estimated rates. The order given is the
    chain order; boards that get no layers drop out of it. shape, in
    MODELS' keys, plans a model that is not one of them."""
    m = shape or MODELS[model]
    sb = shape_bytes(m)
    NL = m["NL"]
    # Boards by position, so a cluster can hold several of one kind: four
    # Zybos are four stages, not one.
    names = list(names)
    usable = [i for i, n in enumerate(names) if capacity(n) >= sb["layer"]]
    if not usable:
        raise ValueError("no board can hold even one layer")
    cap = {}
    for j, i in enumerate(usable):
        # First and last hold the table; a single board is both.
        t = sb["table"] if j in (0, len(usable) - 1) else 0
        cap[i] = max(0, (capacity(names[i]) - t) // sb["layer"])
    if sum(cap.values()) < NL:
        raise ValueError("%d layers do not fit: these boards hold %d"
                         % (NL, sum(cap.values())))
    # Layers a second, from the cycle model at a mid-decode context.
    sp = {i: 1.0 / stage_seconds(names[i], m, 1, ctx, False, False) for i in usable}
    count = {i: 0 for i in usable}
    if mode == "fast":
        for i in sorted(usable, key=lambda i: -sp[i]):
            count[i] = min(cap[i], NL - sum(count.values()))
    else:
        # Water-filling: layers in proportion to speed, capped by
        # capacity, the remainder spread by largest share.
        left, free = NL, set(usable)
        while left and free:
            tot = sum(sp[i] for i in free)
            share = {i: left * sp[i] / tot for i in free}
            capped = {i for i in free if count[i] + share[i] >= cap[i]}
            if capped:
                for i in capped:
                    left -= cap[i] - count[i]
                    count[i] = cap[i]
                free -= capped
                continue
            base = {i: int(share[i]) for i in free}
            for i in free:
                count[i] += base[i]
            left -= sum(base.values())
            for i in sorted(free, key=lambda i: (-(share[i] - base[i]), i))[:left]:
                count[i] += 1
            left = 0
    chain = [i for i in usable if count[i] > 0]
    stages, l0 = [], 0
    for j, i in enumerate(chain):
        layers = list(range(l0, l0 + count[i]))
        l0 += count[i]
        last = j == len(chain) - 1
        t = stage_seconds(names[i], m, len(layers), ctx, j == 0, last)
        stages.append(dict(board=names[i], layers=[layers[0], layers[-1] + 1],
                           emb=j == 0, head=last, seconds=t))
    links = []
    for j in range(len(chain)):
        a, b = chain[j], chain[(j + 1) % len(chain)]
        if a == b:
            continue
        kind = boards.link_between(names[a], names[b])
        nb = sb["hidden_msg"] if j + 1 < len(chain) else 17
        links.append(dict(src=names[a], dst=names[b], kind=kind,
                          seconds=link_seconds(kind, nb)))
    single = sum(s["seconds"] for s in stages) + sum(l["seconds"] for l in links)
    piped = max(s["seconds"] for s in stages)
    return dict(model=model, mode=mode, stages=stages, links=links,
                cannot_hold=[names[i] for i in range(len(names)) if i not in usable],
                not_needed=[names[i] for i in usable if i not in chain],
                seconds_per_token=single, tokens_per_s=1.0 / single,
                pipelined_tokens_per_s=1.0 / piped)


def package(p, out_root, prompt="The capital of France is", tokens=16,
            subnet=(192, 168, 1), log=print):
    """Every stage's board package for a plan: each stage's own build of
    the sequencer (its layers, its images, the hidden-state port), then
    that board's package around it with its place in the chain and its
    neighbours' addresses. The first stage holds the prompt and the
    vocabulary; only the first and last hold the tied table. Needs the
    checkpoint (fetch_qwen.py) and FPGAI_QWEN set to the plan's model."""
    import board_zybo
    import qwen_cosim
    import qwen_full
    import qwen_int
    import qwen_real
    if qwen_real.MODEL != p["model"]:
        raise ValueError("set FPGAI_QWEN=%s for this plan" % p["model"])
    cfg, W = qwen_real.load()
    tok = qwen_real.Tokenizer()
    cal = qwen_cosim.calibration(cfg, W, tok)
    im = qwen_int.IntQwen(cfg, W, 16, True, cal, exact_io=True, log=log)
    ids = tok.encode(prompt)
    n = len(p["stages"])
    ips = [subnet + (10 + i,) for i in range(n)]
    outs = []
    if n == 1:
        # One board is not a pipeline: its package is the ordinary one,
        # every layer, the embedding and the head, no network.
        st = p["stages"][0]
        work = os.path.join(out_root, "build_single")
        im.ms["lanes"] = boards.PACKAGES[st["board"]]["lanes"]
        qwen_full.build_model(im, ids, 0, work, log=log, want=list(ids) + [0])
        out = os.path.join(out_root, "single_%s" % st["board"])
        board_zybo.package(work, st["board"], out, prompt, tokens)
        with open(os.path.join(out_root, "plan.json"), "w") as f:
            json.dump(dict(p, packages=[out]), f, indent=1)
        return [out]
    for i, st in enumerate(p["stages"]):
        pk = boards.PACKAGES[st["board"]]
        if not pk.get("ps7"):
            raise ValueError("%s has no ARM: its stage runs gals.stage_ctrl "
                             "over the fabric UART, not a Zynq package" % st["board"])
        layers = list(range(*st["layers"]))
        work = os.path.join(out_root, "build_stage%d" % i)
        im.ms["lanes"] = pk["lanes"]        # each board its own width
        qwen_full.build_model(im, ids, 0, work, log=log, layers=layers, stage=True,
                              want=list(ids) + [0], table=st["emb"] or st["head"])
        out = os.path.join(out_root, "stage%d_%s" % (i, st["board"]))
        board_zybo.package(work, st["board"], out, prompt, tokens, stage=dict(
            D=im.D, index=i, count=n, l0=layers[0], l1=layers[-1] + 1,
            emb=st["emb"], head=st["head"], ip=ips[i],
            next_ip=ips[(i + 1) % n], first_ip=ips[0]))
        outs.append(out)
    with open(os.path.join(out_root, "plan.json"), "w") as f:
        json.dump(dict(p, ips=ips, packages=outs), f, indent=1)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("boards", nargs="+", choices=sorted(boards.BOARDS))
    ap.add_argument("--model", default="qwen3", choices=sorted(MODELS))
    ap.add_argument("--mode", default="balanced", choices=("balanced", "fast"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--package", metavar="DIR",
                    help="write every stage's board package under DIR")
    a = ap.parse_args()
    p = plan(a.boards, a.model, a.mode)
    if a.package:
        for o in package(p, a.package):
            print("package:", o)
        return
    if a.json:
        print(json.dumps(p, indent=1))
        return
    print("%s over %s (%s)" % (a.model, ", ".join(a.boards), a.mode))
    for s in p["stages"]:
        print("  %-12s layers %2d-%2d%s%s  %.2f s/token"
              % (s["board"], s["layers"][0], s["layers"][1] - 1,
                 "  +embedding" if s["emb"] else "", "  +head" if s["head"] else "",
                 s["seconds"]))
    for l in p["links"]:
        print("  %s -> %s over %s, %.1f ms" % (l["src"], l["dst"], l["kind"],
                                                l["seconds"] * 1e3))
    if p["cannot_hold"]:
        print("  cannot hold a layer (no DRAM the design reaches): %s"
              % ", ".join(p["cannot_hold"]))
    if p["not_needed"]:
        print("  not needed in this mode: %s" % ", ".join(p["not_needed"]))
    print("  one stream: %.2f s/token, %.2f tokens/s; every stage busy: %.2f tokens/s"
          % (p["seconds_per_token"], p["tokens_per_s"], p["pipelined_tokens_per_s"]))


if __name__ == "__main__":
    main()
