"""Split a model over any set of boards: which layers go where, and how
each pair of neighbours talks.

The boards run as GALS stages (gals.py): each on its own clock, a
contiguous range of layers each, passing the hidden state on and the
chosen token back. This decides the ranges and the links from what
boards.py knows about each board:

  speed     bytes of weights a second it can stream, the smaller of its
            DDR's sustained rate and what the design consumes at its
            clock: 16 bytes a core cycle, over the 1.18 bus cycles a core
            cycle measured through the Zybo's registers
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
LANE_BYTES = 16          # weight bytes the core takes a cycle
BUS_PER_CORE = 1.18      # measured, 24 layers through the Zybo's registers
BASE = 0x08000000        # below it, the ARM's program
UART_BAUD = 6.25e6       # board to board on two wires: clk/8 at 50 MHz
ETH_BYTES_S = 100e6      # UDP over Gigabit Ethernet through lwIP, sustained
LINK_LATENCY_S = {"ethernet": 200e-6, "uart": 20e-6}

MODELS = {
    # The shapes that set the bytes; the checkpoint's config.json has them.
    "qwen2.5": dict(D=896, F=4864, H=14, KV=2, hd=64, NL=24, V=151936),
    "qwen3": dict(D=1024, F=3072, H=16, KV=8, hd=128, NL=28, V=151936),
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


def speed(name):
    """Weight bytes a second the board streams through this design."""
    b, pk = boards.BOARDS[name], boards.PACKAGES.get(name, {})
    mhz = pk.get("fpgai_mhz", 50)
    design = LANE_BYTES * mhz * 1e6 / BUS_PER_CORE
    return min(b["mem_gbytes_per_s"] * 1e9, design)


def capacity(name):
    pk = boards.PACKAGES.get(name)
    if not pk or pk.get("ddr") is None:
        return 0
    return pk["ddr_bytes"] - BASE - (8 << 20)     # vocab text and margin


def link_seconds(kind, nbytes):
    rate = ETH_BYTES_S if kind == "ethernet" else UART_BAUD / 10.0
    return LINK_LATENCY_S[kind] + nbytes / rate


def plan(names, model="qwen3", mode="balanced"):
    """Stages in chain order: [{board, layers, emb, head, seconds}], the
    links between them, and the estimated rates. The order given is the
    chain order; boards that get no layers drop out of it."""
    m = MODELS[model]
    sb = shape_bytes(m)
    NL = m["NL"]
    usable = [n for n in names if capacity(n) >= sb["layer"]]
    if not usable:
        raise ValueError("no board can hold even one layer")
    cap = {}
    for i, n in enumerate(usable):
        # First and last hold the table; a single board is both.
        t = sb["table"] if i in (0, len(usable) - 1) else 0
        cap[n] = max(0, (capacity(n) - t) // sb["layer"])
    if sum(cap.values()) < NL:
        raise ValueError("%d layers do not fit: these boards hold %d"
                         % (NL, sum(cap.values())))
    sp = {n: speed(n) for n in usable}
    count = {n: 0 for n in usable}
    if mode == "fast":
        for n in sorted(usable, key=lambda n: -sp[n]):
            count[n] = min(cap[n], NL - sum(count.values()))
    else:
        # Water-filling: layers in proportion to speed, capped by
        # capacity, the remainder spread by largest share.
        left, free = NL, set(usable)
        while left and free:
            tot = sum(sp[n] for n in free)
            share = {n: left * sp[n] / tot for n in free}
            capped = {n for n in free if count[n] + share[n] >= cap[n]}
            if capped:
                for n in capped:
                    left -= cap[n] - count[n]
                    count[n] = cap[n]
                free -= capped
                continue
            base = {n: int(share[n]) for n in free}
            for n in free:
                count[n] += base[n]
            left -= sum(base.values())
            for n in sorted(free, key=lambda n: -(share[n] - base[n]))[:left]:
                count[n] += 1
            left = 0
    chain = [n for n in usable if count[n] > 0]
    stages, l0 = [], 0
    for i, n in enumerate(chain):
        layers = list(range(l0, l0 + count[n]))
        l0 += count[n]
        last = i == len(chain) - 1
        t = len(layers) * sb["layer_weights"] / sp[n]
        if last:
            t += m["V"] * m["D"] / sp[n]           # the head
        stages.append(dict(board=n, layers=[layers[0], layers[-1] + 1],
                           emb=i == 0, head=last, seconds=t))
    links = []
    for i in range(len(chain)):
        a, b = chain[i], chain[(i + 1) % len(chain)]
        if a == b:
            continue
        kind = boards.link_between(a, b)
        nb = sb["hidden_msg"] if i + 1 < len(chain) else 17
        links.append(dict(src=a, dst=b, kind=kind, seconds=link_seconds(kind, nb)))
    single = sum(s["seconds"] for s in stages) + sum(l["seconds"] for l in links)
    piped = max(s["seconds"] for s in stages)
    return dict(model=model, mode=mode, stages=stages, links=links,
                cannot_hold=[n for n in names if n not in usable],
                not_needed=[n for n in usable if n not in chain],
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
