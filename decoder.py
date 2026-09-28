"""The trained checkpoint as the hardware runs it: integers only, from the
embedding lookup to the argmax.

generate.py runs the same checkpoint through bit-accurate models of the
generated units, but the host sequences it and picks every activation's
scale at run time in floating point. A decoder on an FPGA cannot do that.
Here every scale is fixed ahead of time by calibration, every stage is one
of the generated blocks' own golden models called with those constants,
and the only thing between two blocks is an int8 code. That makes it the
reference a hardware sequencer can be checked against bit for bit.

One decode step, for token t at position i:

    x    = resadd(tok[t], pos[i])            embedding, two int8 tables
    xn   = rmsnorm(x, g1)
    q, k, v = proj(Wq, xn), proj(Wk, xn), proj(Wv, xn)
    K[i], V[i] = k, v                        the KV cache
    c    = attn(q, K[0..i], V[0..i])
    x1   = resadd(x, proj(Wo, c))
    h    = relu(proj(W1, rmsnorm(x1, g2)))
    x2   = resadd(x1, proj(W2, h))
    next = argmax(proj(Whead, x2))

Run: python3 decoder.py [--prompt "the agent"] [--tokens 40]
"""
import argparse
import json
import math
import os

import specgen
from train_tiny import CKPT

ROOT = os.path.dirname(os.path.abspath(__file__))


def model_spec(ck):
    """The checkpoint's geometry as a model spec, so every block derives at
    the size this model actually has."""
    base = specgen.load_model_spec()
    return dict(base, name="tiny_llm", n_layer=1, d_model=ck["d_model"],
                n_head=1, n_kv_head=1, head_dim=ck["d_model"],
                d_ff=ck["d_ff"], vocab=len(ck["chars"]),
                seq_len=ck["seq"], weight_bits=8, activation_bits=8)


def specs(ms):
    return {"rmsnorm": specgen.derive_rmsnorm_spec(ms),
            "attn": specgen.derive_attn_spec(ms),
            "resadd": specgen.derive_resadd_spec(ms),
            "proj": specgen.derive_proj_spec(ms),
            "softmax": specgen.derive_softmax_spec(ms)}


# ---------------------------------------------------------------- float

def _mv(x, w):
    return [sum(x[r] * w[r][c] for r in range(len(w)))
            for c in range(len(w[0]))]


def _rms(v, g):
    inv = (sum(u * u for u in v) / len(v) + 1e-6) ** -0.5
    return [v[i] * inv * g[i] for i in range(len(v))]


def float_step(ck, ids):
    """The checkpoint's forward pass for the last position, in plain
    floats, returning every intermediate so calibration can see them."""
    w = ck["weights"]
    D = ck["d_model"]
    t = {}
    h = [[w["tok"][tk][j] + w["pos"][i][j] for j in range(D)]
         for i, tk in enumerate(ids)]
    hn = [_rms(v, w["g1"][0]) for v in h]
    q = [_mv(v, w["wq"]) for v in hn]
    k = [_mv(v, w["wk"]) for v in hn]
    val = [_mv(v, w["wv"]) for v in hn]
    i = len(ids) - 1
    sc = [sum(a * b for a, b in zip(q[i], k[j])) / math.sqrt(D)
          for j in range(i + 1)]
    mx = max(sc)
    ex = [math.exp(s - mx) for s in sc]
    tot = sum(ex)
    p = [e / tot for e in ex]
    ctx = [sum(p[j] * val[j][d] for j in range(i + 1)) for d in range(D)]
    a = _mv(ctx, w["wo"])
    x1 = [h[i][j] + a[j] for j in range(D)]
    hn2 = _rms(x1, w["g2"][0])
    u = [max(0.0, v) for v in _mv(hn2, w["w1"])]
    m = _mv(u, w["w2"])
    x2 = [x1[j] + m[j] for j in range(D)]
    lg = _mv(x2, w["head"])
    t.update(x=h[i], xn=hn[i], q=q[i], k=k[i], v=val[i], score=sc, ctx=ctx,
             a=a, x1=x1, xn2=hn2, h=u, m=m, x2=x2, lg=lg,
             tok=[w["tok"][ids[i]]], pos=[w["pos"][i]])
    return t


# ---------------------------------------------------------------- scales

def pick(ratio, p):
    """A (scale, shift) with scale / 2**shift closest to ratio: the largest
    shift whose scale still fits the requantizer's multiplier."""
    mw, sw = p["scale_width"], p["shift_width"]
    best = (1, 1)
    for sh in range(1, min(1 << sw, 64)):
        sc = int(round(ratio * (1 << sh)))
        if sc >= 1 << mw:
            break
        if sc >= 1:
            best = (sc, sh)
    return best


def qtensor(m, bits=8):
    """Symmetric per-tensor int8: the projection has one output scale per
    call, so a weight matrix gets one scale."""
    hi = (1 << (bits - 1)) - 1
    flat = [v for row in m for v in row]
    s = (max(abs(v) for v in flat) or 1.0) / hi
    return [[max(-hi - 1, min(hi, int(round(v / s)))) for v in row]
            for row in m], s


def calibrate(ck, windows=None):
    """Largest magnitude of every tensor over teacher-forced corpus
    windows: static scales, fixed before the hardware ever runs."""
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    L = ck["seq"]
    mx = {}
    ends = windows or range(1, min(len(ids), 4 * L) + 1)
    for end in ends:
        ctx = ids[max(0, end - L):end]
        for k, v in float_step(ck, ctx).items():
            flat = [abs(u) for row in (v if isinstance(v[0], list) else [v])
                    for u in row]
            mx[k] = max(mx.get(k, 0.0), max(flat))
    return mx


class IntDecoder:
    """Integer-only decode, every stage a generated block's golden model."""

    def __init__(self, ck):
        self.ck = ck
        self.ms = model_spec(ck)
        self.sp = specs(self.ms)
        self.D, self.F = ck["d_model"], ck["d_ff"]
        self.L = ck["seq"]
        w = ck["weights"]
        cal = calibrate(ck)
        self.cal = cal
        hi = 127.0
        s = {k: v / hi for k, v in cal.items()}
        self.W = {}
        for n in ("wq", "wk", "wv", "wo", "w1", "w2", "head"):
            self.W[n], s["w_" + n] = qtensor(w[n])
        self.tok, s["tok"] = qtensor(w["tok"])
        self.pos, s["pos"] = qtensor(w["pos"])
        g1, s["g1"] = qtensor(w["g1"])
        g2, s["g2"] = qtensor(w["g2"])
        self.g1, self.g2 = g1[0], g2[0]

        rq = self.sp["proj"]["parameters"]
        rn = self.sp["rmsnorm"]["parameters"]
        at = self.sp["attn"]["parameters"]
        ra = self.sp["resadd"]["parameters"]
        sm = self.sp["softmax"]["parameters"]
        D = self.D

        # Attention scores: q.k / sqrt(D) has to land in the softmax's
        # score format, 2**-score_frac per count, after a shift of
        # shift_s. The shift is a power of two, so the rest of the factor
        # is folded into q's own scale, which the q projection's
        # requantizer sets for free. shift_s is the largest that leaves
        # the calibrated q inside int8.
        sf = sm["score_frac"]
        g = at.get("score_guard", 0)          # scores are (t << g) >> shift_s
        need = s["q"]
        shift_s = 0
        while shift_s + 1 < (1 << at["shift_s_width"]):
            sq = math.sqrt(D) * 2.0 ** (g - (shift_s + 1) - sf) / s["k"]
            if sq < need:
                break
            shift_s += 1
        s["q"] = math.sqrt(D) * 2.0 ** (g - shift_s - sf) / s["k"]
        self.shift_s = shift_s
        self.s = s

        def ratio(src, dst):
            return pick(src / s[dst], rq)

        self.k_ = {
            "q": ratio(s["xn"] * s["w_wq"], "q"),
            "k": ratio(s["xn"] * s["w_wk"], "k"),
            "v": ratio(s["xn"] * s["w_wv"], "v"),
            "a": ratio(s["ctx"] * s["w_wo"], "a"),
            "h": ratio(s["xn2"] * s["w_w1"], "h"),
            "m": ratio(s["h"] * s["w_w2"], "m"),
            "lg": ratio(s["x2"] * s["w_head"], "lg"),
            # weighted sum of V: weights are Q0.weight_frac
            "ctx": pick(2.0 ** -at["weight_frac"] * s["v"] / s["ctx"], at),
        }
        # RMSNorm: t = x*g*m >> (e + k), with 1/sqrt(ssq) = m >> (ow + e),
        # so the normalised value is t * 2**(k - ow) * sqrt(D) * s_g.
        ow, kk = rn["rsqrt_out_width"], rn["norm_shift"]
        for name, g, dst in (("n1", "g1", "xn"), ("n2", "g2", "xn2")):
            self.k_[name] = pick(2.0 ** (kk - ow) * math.sqrt(D) * s[g]
                                 / s[dst], rn)
        self.eps = 1
        # Residual adds: two operands at their own scales into a third.
        for name, a, b, dst in (("emb", "tok", "pos", "x"),
                                ("r1", "x", "a", "x1"),
                                ("r2", "x1", "m", "x2")):
            self.k_[name] = self._resadd_consts(s[a] / s[dst], s[b] / s[dst],
                                                ra)
        self.reset()

    @staticmethod
    def _resadd_consts(ra_, rb_, p):
        mw, sw = p["scale_width"], p["shift_width"]
        best = None
        for sh in range(1, min(1 << sw, 48)):
            sa, sb = int(round(ra_ * (1 << sh))), int(round(rb_ * (1 << sh)))
            if max(sa, sb) >= 1 << mw:
                break
            best = (sa, sb, sh)
        return best

    def reset(self):
        self.K, self.V = [], []

    def _proj(self, x, name, key):
        W = self.W[name]
        sc, sh = self.k_[key]
        dw = 8
        return [specgen.requant_golden(
            sum(x[r] * W[r][c] for r in range(len(W))), sc, sh, dw)[0]
            for c in range(len(W[0]))]

    def _rmsnorm(self, x, g, key):
        sc, sh = self.k_[key]
        p = self.sp["rmsnorm"]
        rs_p = p["derivation"]["rsqrt"]
        return specgen.rmsnorm_golden(x, g, self.eps, sc, sh,
                                      p["parameters"], rs_p)[4]

    def _resadd(self, a, b, key):
        sa, sb, sh = self.k_[key]
        return [specgen.resadd_golden(u, v, sa, sb, sh, 8)
                for u, v in zip(a, b)]

    def step(self, tok, i):
        """One decode step. Returns (next token, logits, trace)."""
        assert i == len(self.K), "positions must arrive in order"
        x = self._resadd(self.tok[tok], self.pos[i], "emb")
        xn = self._rmsnorm(x, self.g1, "n1")
        q = self._proj(xn, "wq", "q")
        self.K.append(self._proj(xn, "wk", "k"))
        self.V.append(self._proj(xn, "wv", "v"))
        at = self.sp["attn"]
        sc, sh = self.k_["ctx"]
        c = specgen.attn_golden(q, self.K, self.V, i + 1, self.shift_s,
                                sc, sh, at["parameters"],
                                self.sp["softmax"]["parameters"])[4]
        a = self._proj(c, "wo", "a")
        x1 = self._resadd(x, a, "r1")
        xn2 = self._rmsnorm(x1, self.g2, "n2")
        h = [max(0, v) for v in self._proj(xn2, "w1", "h")]
        m = self._proj(h, "w2", "m")
        x2 = self._resadd(x1, m, "r2")
        lg = self._proj(x2, "head", "lg")
        nxt = max(range(len(lg)), key=lambda k: (lg[k], -k))
        return nxt, lg, dict(x=x, xn=xn, q=q, c=c, a=a, x1=x1, xn2=xn2,
                             h=h, m=m, x2=x2, lg=lg)

    def generate(self, prompt_ids, n):
        """Greedy decode. When the context is full the window slides and
        is prefilled again, which is what the float reference does."""
        out = list(prompt_ids)
        for _ in range(n):
            ctx = out[-self.L:]
            self.reset()
            for i, t in enumerate(ctx):
                nxt, _, _ = self.step(t, i)
            out.append(nxt)
        return out


# ---------------------------------------------------------------- RTL

REGIONS = ("tok", "pos", "x", "n", "q", "c", "a", "x1", "h", "x2", "lg")


def _clog2(n):
    return max(1, (n - 1).bit_length())


def layout(dec):
    """Where everything lives. Weights, embeddings and gains sit behind one
    registered read port, as they would in DDR; activations, the KV cache
    and the constants are on chip."""
    D, F, V, L = dec.D, dec.F, len(dec.ck["chars"]), dec.L
    off, p = {}, 0
    for name, size in (("tok", V * D), ("pos", L * D), ("g1", D), ("g2", D),
                       ("wq", D * D), ("wk", D * D), ("wv", D * D),
                       ("wo", D * D), ("w1", D * F), ("w2", F * D),
                       ("head", D * V)):
        off[name] = p
        p += size
    return {"D": D, "F": F, "V": V, "L": L, "B": max(D, F, V),
            "off": off, "size": p}


def param_image(dec):
    """The parameter memory's contents, as the layout places them. Weight
    matrices are column-major, which is how the projection addresses
    them: column c, row r at c * depth + r."""
    lay = layout(dec)
    img = [0] * lay["size"]
    D, L = lay["D"], lay["L"]
    for t_, row in enumerate(dec.tok):
        for j, v in enumerate(row):
            img[lay["off"]["tok"] + t_ * D + j] = v
    for i, row in enumerate(dec.pos[:L]):
        for j, v in enumerate(row):
            img[lay["off"]["pos"] + i * D + j] = v
    for j in range(D):
        img[lay["off"]["g1"] + j] = dec.g1[j]
        img[lay["off"]["g2"] + j] = dec.g2[j]
    for name in ("wq", "wk", "wv", "wo", "w1", "w2", "head"):
        W = dec.W[name]
        depth = len(W)
        for c in range(len(W[0])):
            for r in range(depth):
                img[lay["off"][name] + c * depth + r] = W[r][c]
    return img


FIX_RELU = "clamp_the_up_projection_at_zero"


def _port_widths(sp):
    """The sub-blocks' port widths, from their own specs, so the decoder
    wires whatever the derivation produced at this model's size."""
    rn, at, pj = (sp[b]["parameters"] for b in ("rmsnorm", "attn", "proj"))
    return {"rn_addr": rn["addr_width"], "rn_eps": rn["rsqrt_in_width"],
            "at_addr": at["addr_width"], "at_hd": at["head_dim_width"],
            "at_n": at["n_width"], "at_shs": at["shift_s_width"],
            "pj_depth": pj["depth_width"], "pj_col": pj["col_width"],
            "pj_addr": pj["addr_width"], "sc": pj["scale_width"],
            "sh": pj["shift_width"]}


def _ports(lay):
    tw, pw, paw = _clog2(lay["V"]), _clog2(lay["L"]), _clog2(lay["size"])
    P = lambda n_, d, w, desc: {"name": n_, "dir": d, "width": w,
                                "desc": desc}
    return [P("clk", "input", 1, "clock, rising edge"),
            P("rst_n", "input", 1, "synchronous active-low reset"),
            P("start", "input", 1, "one-cycle pulse: decode tok at pos"),
            P("tok", "input", tw, "the token at this position"),
            P("pos", "input", pw, "its position; the KV cache slot"),
            P("p_addr", "output", paw, "parameter memory address"),
            P("p_data", "input", 8, "parameter word, one cycle after "
                                    "its address"),
            P("lg_valid", "output", 1, "a logit is on lg_index/lg_data"),
            P("lg_index", "output", tw, "which logit"),
            P("lg_data", "output", 8, "its int8 value"),
            P("next_tok", "output", tw, "argmax, lowest index on a tie"),
            P("done", "output", 1, "one-cycle pulse: next_tok is valid"),
            P("busy", "output", 1, "high from start to done")]


def derive_spec(dec):
    """checkpoint -> decoder spec. Everything the RTL needs and nothing it
    has to guess: the geometry, where each tensor sits in the parameter
    memory, and every calibrated (scale, shift) with the block it drives."""
    lay = layout(dec)
    return {
        "name": "decoder_%s" % dec.ms["name"],
        "description": "One decode step of the %d-dimensional checkpoint, "
                       "embedding lookup to argmax, sequencing the "
                       "generated projection, RMSNorm, attention and "
                       "residual add blocks" % lay["D"],
        "top_module": "decoder",
        "unit": "token",
        "parameters": {
            "d_model": lay["D"], "d_ff": lay["F"], "vocab": lay["V"],
            "context": lay["L"], "region": lay["B"],
            "param_offsets": lay["off"], "param_words": lay["size"],
            "constants": {k_: list(v) for k_, v in dec.k_.items()},
            "shift_s": dec.shift_s, "eps": dec.eps,
            "ports": _port_widths(dec.sp),
            "data_width": 8, "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "ports": _ports(lay),
        "derivation": {
            "model": dec.ms["name"],
            "rule": "block specs derived at the checkpoint's own size; "
                    "activation scales fixed by calibration over the "
                    "corpus; every constant a (scale, shift) pair for the "
                    "requantizer or residual add it drives",
        },
    }


def render_decoder(spec, fixes=frozenset((FIX_RELU,))):
    """One decode step in hardware: the generated blocks, sequenced.

    Every stage is an instance of a generated block, and one instance
    serves every use of it: one projection runs all seven matrices, one
    RMSNorm both norms, one residual add all three adds. What is new here
    is only the routing: which region of the activation memory each block
    reads and writes, which constants it runs with, and the order.

    The seeded first cut leaves out the MLP's ReLU, the one piece of
    arithmetic the sequencer does itself rather than a block doing it.
    """
    p = spec["parameters"]
    D, F, V, L, B = (p["d_model"], p["d_ff"], p["vocab"], p["context"],
                     p["region"])
    off = p["param_offsets"]
    k = {k_: tuple(v) for k_, v in p["constants"].items()}
    lay = {"size": p["param_words"]}
    relu = ("pj_data[7] ? 8'sd0 : pj_data" if FIX_RELU in fixes
            else "pj_data")
    w = p["ports"]
    bw = _clog2(B)
    aw = _clog2(len(REGIONS) * B)
    paw = _clog2(lay["size"])
    tw, pw = _clog2(V), _clog2(L)
    kvw = _clog2(L * D)
    R = {r: i * B for i, r in enumerate(REGIONS)}
    # (state, block, source region, param base, depth, cols, scale, shift,
    #  destination)
    projs = [("S_Q", "x", "n", "wq", D, D, k["q"], "q"),
             ("S_K", "x", "n", "wk", D, D, k["k"], "kc"),
             ("S_V", "x", "n", "wv", D, D, k["v"], "vc"),
             ("S_O", "x", "c", "wo", D, D, k["a"], "a"),
             ("S_UP", "x", "n", "w1", D, F, k["h"], "h"),
             ("S_DN", "x", "h", "w2", F, D, k["m"], "a"),
             ("S_HD", "x", "x2", "head", D, V, k["lg"], "lg")]
    states = ["S_IDLE", "S_CPTOK", "S_CPPOS", "S_EMB", "S_N1", "S_Q", "S_K",
              "S_V", "S_LQ", "S_ATT", "S_O", "S_R1", "S_N2", "S_UP", "S_DN",
              "S_R2", "S_HD", "S_ARG"]
    sw = _clog2(len(states))
    lines = []
    A = lines.append
    A("// GENERATED by decoder.py: do not edit by hand.")
    A("// One decode step of the %d-dimensional checkpoint, from the" % D)
    A("// embedding lookup to the argmax, over the generated blocks.")
    A("module decoder (")
    A("  input                    clk,")
    A("  input                    rst_n,")
    A("  input                    start,")
    A("  input      [%d:0] tok," % (tw - 1))
    A("  input      [%d:0] pos," % (pw - 1))
    A("  output     [%d:0] p_addr," % (paw - 1))
    A("  input      signed [7:0] p_data,")
    A("  output reg               lg_valid,")
    A("  output reg [%d:0] lg_index," % (tw - 1))
    A("  output reg signed [7:0] lg_data,")
    A("  output reg [%d:0] next_tok," % (tw - 1))
    A("  output reg               done,")
    A("  output reg               busy")
    A(");")
    A("  localparam " + ", ".join("%s = %d'd%d" % (s, sw, i)
                                  for i, s in enumerate(states)) + ";")
    A("  reg [%d:0] st;" % (sw - 1))
    A("  reg ph;")
    A("  reg [%d:0] tok_r;" % (tw - 1))
    A("  reg [%d:0] pos_r;" % (pw - 1))
    A("  reg signed [7:0] act [0:%d];" % (len(REGIONS) * B - 1))
    A("  reg signed [7:0] kc [0:%d];" % (L * D - 1))
    A("  reg signed [7:0] vc [0:%d];" % (L * D - 1))
    A("  reg [%d:0] fj, ocnt;" % bw)
    A("  reg cp_v;")
    A("  reg [%d:0] cp_i;" % bw)
    A("")
    A("  // ---- the blocks, one instance each")
    A("  reg pj_start;")
    A("  reg [%d:0] pj_depth, pj_cols;" % (w["pj_depth"] - 1))
    A("  reg [%d:0] pj_scale;" % (w["sc"] - 1))
    A("  reg [%d:0] pj_shift;" % (w["sh"] - 1))
    A("  wire [%d:0] pj_a_addr;" % (w["pj_depth"] - 1))
    A("  wire [%d:0] pj_w_addr;" % (w["pj_addr"] - 1))
    A("  reg signed [7:0] pj_a_data;")
    A("  wire pj_valid, pj_busy;")
    A("  wire [%d:0] pj_index;" % (w["pj_col"] - 1))
    A("  wire signed [7:0] pj_data;")
    A("  proj u_proj (.clk(clk), .rst_n(rst_n), .start(pj_start),")
    A("    .depth(pj_depth), .cols(pj_cols), .scale(pj_scale),")
    A("    .shift(pj_shift), .a_addr(pj_a_addr), .a_data(pj_a_data),")
    A("    .w_addr(pj_w_addr), .w_data(p_data), .o_valid(pj_valid),")
    A("    .o_index(pj_index), .o_data(pj_data), .busy(pj_busy));")
    A("")
    A("  reg rn_start;")
    A("  reg [%d:0] rn_scale;" % (w["sc"] - 1))
    A("  reg [%d:0] rn_shift;" % (w["sh"] - 1))
    A("  wire [%d:0] rn_x_addr, rn_g_addr;" % (w["rn_addr"] - 1))
    A("  reg signed [7:0] rn_x_data;")
    A("  wire rn_valid, rn_busy;")
    A("  wire [%d:0] rn_index;" % (w["rn_addr"] - 1))
    A("  wire signed [7:0] rn_data;")
    A("  rmsnorm u_norm (.clk(clk), .rst_n(rst_n), .start(rn_start),")
    A("    .eps(%d'd%d), .scale_o(rn_scale), .shift_o(rn_shift),"
      % (w["rn_eps"], p["eps"]))
    A("    .x_addr(rn_x_addr), .x_data(rn_x_data), .g_addr(rn_g_addr),")
    A("    .g_data(p_data), .o_valid(rn_valid), .o_index(rn_index),")
    A("    .o_data(rn_data), .busy(rn_busy));")
    A("")
    A("  reg at_start, at_load_valid;")
    A("  reg signed [7:0] at_load_data;")
    A("  wire [%d:0] at_k_addr, at_v_addr;" % (w["at_addr"] - 1))
    A("  reg signed [7:0] at_k_data, at_v_data;")
    A("  wire at_valid, at_busy;")
    A("  wire [%d:0] at_index;" % (w["at_hd"] - 1))
    A("  wire signed [7:0] at_data;")
    A("  attn u_attn (.clk(clk), .rst_n(rst_n), .load_valid(at_load_valid),")
    A("    .load_data(at_load_data), .start(at_start),")
    A("    .n({%d'd0, pos_r} + %d'd1)," % (w["at_n"] - pw, w["at_n"]))
    A("    .shift_s(%d'd%d), .scale_o(%d'd%d), .shift_o(%d'd%d),"
      % (w["at_shs"], p["shift_s"], w["sc"], k["ctx"][0], w["sh"],
         k["ctx"][1]))
    A("    .k_addr(at_k_addr), .k_data(at_k_data), .v_addr(at_v_addr),")
    A("    .v_data(at_v_data), .o_valid(at_valid), .o_index(at_index),")
    A("    .o_data(at_data), .busy(at_busy));")
    A("")
    A("  reg signed [7:0] ra_a, ra_b;")
    A("  reg ra_v;")
    A("  reg [%d:0] ra_sa, ra_sb;" % (w["sc"] - 1))
    A("  reg [%d:0] ra_sh;" % (w["sh"] - 1))
    A("  wire signed [7:0] ra_y;")
    A("  wire ra_vout;")
    A("  resadd u_add (.clk(clk), .rst_n(rst_n), .a(ra_a), .b(ra_b),")
    A("    .scale_a(ra_sa), .scale_b(ra_sb), .shift(ra_sh),")
    A("    .valid_in(ra_v), .y(ra_y), .valid_out(ra_vout));")
    A("")
    A("  // ---- per-state routing and constants")
    A("  reg [%d:0] src, dst, ra_src_a, ra_src_b;" % (aw - 1))
    A("  reg [%d:0] pbase;" % (paw - 1))
    A("  reg [%d:0] expect_n;" % bw)
    A("  always @(*) begin")
    A("    src = 0; dst = 0; pbase = 0; ra_src_a = 0; ra_src_b = 0;")
    A("    pj_depth = 0; pj_cols = 0; pj_scale = 0; pj_shift = 0;")
    A("    rn_scale = 0; rn_shift = 0; ra_sa = 0; ra_sb = 0; ra_sh = 0;")
    A("    expect_n = %d;" % D)
    A("    case (st)")
    A("      S_CPTOK: begin dst = %d; pbase = %d + tok_r * %d; end"
      % (R["tok"], off["tok"], D))
    A("      S_CPPOS: begin dst = %d; pbase = %d + pos_r * %d; end"
      % (R["pos"], off["pos"], D))
    for s_, a_, b_, d_, key in (("S_EMB", "tok", "pos", "x", "emb"),
                                ("S_R1", "x", "a", "x1", "r1"),
                                ("S_R2", "x1", "a", "x2", "r2")):
        sa, sb, sh = k[key]
        A("      %s: begin ra_src_a = %d; ra_src_b = %d; dst = %d;"
          % (s_, R[a_], R[b_], R[d_]))
        A("        ra_sa = %d; ra_sb = %d; ra_sh = %d; end" % (sa, sb, sh))
    for s_, src_, g_, key in (("S_N1", "x", "g1", "n1"),
                              ("S_N2", "x1", "g2", "n2")):
        sc, sh = k[key]
        A("      %s: begin src = %d; dst = %d; pbase = %d;"
          % (s_, R[src_], R["n"], off[g_]))
        A("        rn_scale = %d; rn_shift = %d; end" % (sc, sh))
    for s_, _, src_, w_, dep, cols, (sc, sh), d_ in projs:
        dnum = R[d_] if d_ in R else 0
        A("      %s: begin src = %d; dst = %d; pbase = %d;"
          % (s_, R[src_], dnum, off[w_]))
        A("        pj_depth = %d; pj_cols = %d; pj_scale = %d;"
          % (dep, cols, sc))
        A("        pj_shift = %d; expect_n = %d; end" % (sh, cols))
    A("      S_LQ: src = %d;" % R["q"])
    A("      S_ATT: dst = %d;" % R["c"])
    A("      S_ARG: begin src = %d; expect_n = %d; end" % (R["lg"], V))
    A("      default: ;")
    A("    endcase")
    A("  end")
    A("")
    A("  // One parameter port, shared: the norm's gain in a norm state,")
    A("  // the projection's weight in a projection state, the embedding")
    A("  // tables while they are copied in.")
    A("  wire in_norm = (st == S_N1) || (st == S_N2);")
    A("  wire in_copy = (st == S_CPTOK) || (st == S_CPPOS);")
    A("  assign p_addr = in_norm ? pbase + rn_g_addr")
    A("                : in_copy ? pbase + fj")
    A("                : pbase + pj_w_addr;")
    A("")
    A("  // Registered reads, one edge, as every block expects.")
    A("  always @(posedge clk) begin")
    A("    pj_a_data <= act[src + pj_a_addr];")
    A("    rn_x_data <= act[src + rn_x_addr];")
    A("    at_k_data <= kc[at_k_addr[%d:0]];" % (kvw - 1))
    A("    at_v_data <= vc[at_v_addr[%d:0]];" % (kvw - 1))
    A("  end")
    A("")
    A("  wire run_proj = (st == S_Q) || (st == S_K) || (st == S_V) ||")
    A("                  (st == S_O) || (st == S_UP) || (st == S_DN) ||")
    A("                  (st == S_HD);")
    A("  wire blk_busy = run_proj ? pj_busy : in_norm ? rn_busy : at_busy;")
    A("  wire streaming = (st == S_EMB) || (st == S_R1) || (st == S_R2);")
    A("  reg signed [7:0] best;")
    A("  reg [%d:0] best_i;" % (tw - 1))
    A("")
    A("  always @(posedge clk) begin")
    A("    if (!rst_n) begin")
    A("      st <= S_IDLE; ph <= 1'b0; busy <= 1'b0; done <= 1'b0;")
    A("      tok_r <= 0; pos_r <= 0; next_tok <= 0; fj <= 0; ocnt <= 0;")
    A("      cp_v <= 1'b0; cp_i <= 0; pj_start <= 1'b0; rn_start <= 1'b0;")
    A("      at_start <= 1'b0; at_load_valid <= 1'b0; at_load_data <= 0;")
    A("      ra_a <= 0; ra_b <= 0; ra_v <= 1'b0; lg_valid <= 1'b0;")
    A("      lg_index <= 0; lg_data <= 0; best <= 0; best_i <= 0;")
    A("    end else begin")
    A("      done <= 1'b0; pj_start <= 1'b0; rn_start <= 1'b0;")
    A("      at_start <= 1'b0; at_load_valid <= 1'b0; ra_v <= 1'b0;")
    A("      lg_valid <= 1'b0; cp_v <= 1'b0;")
    A("      // Block outputs land in the state's destination.")
    A("      if (run_proj && pj_valid) begin")
    A("        ocnt <= ocnt + 1;")
    A("        if (st == S_K) kc[pos_r * %d + pj_index] <= pj_data;" % D)
    A("        else if (st == S_V) vc[pos_r * %d + pj_index] <= pj_data;" % D)
    A("        else if (st == S_UP)")
    A("          act[dst + pj_index] <= %s;" % relu)
    A("        else act[dst + pj_index] <= pj_data;")
    A("        if (st == S_HD) begin")
    A("          lg_valid <= 1'b1; lg_index <= pj_index[%d:0];" % (tw - 1))
    A("          lg_data <= pj_data;")
    A("        end")
    A("      end")
    A("      if (in_norm && rn_valid) begin")
    A("        act[dst + rn_index] <= rn_data; ocnt <= ocnt + 1;")
    A("      end")
    A("      if (st == S_ATT && at_valid) begin")
    A("        act[dst + at_index] <= at_data; ocnt <= ocnt + 1;")
    A("      end")
    A("      if (streaming && ra_vout) begin")
    A("        act[dst + ocnt] <= ra_y; ocnt <= ocnt + 1;")
    A("      end")
    A("      if (cp_v) act[dst + cp_i] <= p_data;")
    A("")
    A("      case (st)")
    A("        S_IDLE: if (start) begin")
    A("          tok_r <= tok; pos_r <= pos; busy <= 1'b1;")
    A("          st <= S_CPTOK; fj <= 0; ocnt <= 0; ph <= 1'b0;")
    A("        end")
    A("        // Embedding rows in from the parameter port, one a cycle; the")
    A("        // data is a cycle behind its address.")
    A("        S_CPTOK, S_CPPOS: begin")
    A("          if (fj < %d) begin" % D)
    A("            cp_v <= 1'b1; cp_i <= fj; fj <= fj + 1;")
    A("          end else if (!cp_v) begin")
    A("            fj <= 0;")
    A("            st <= (st == S_CPTOK) ? S_CPPOS : S_EMB;")
    A("          end")
    A("        end")
    A("        S_EMB, S_R1, S_R2: begin")
    A("          if (fj < %d) begin" % D)
    A("            ra_a <= act[ra_src_a + fj]; ra_b <= act[ra_src_b + fj];")
    A("            ra_v <= 1'b1; fj <= fj + 1;")
    A("          end")
    A("          if (ocnt == %d) begin" % D)
    A("            fj <= 0; ocnt <= 0; ph <= 1'b0;")
    A("            st <= (st == S_EMB) ? S_N1 : (st == S_R1) ? S_N2 : S_HD;")
    A("          end")
    A("        end")
    A("        S_LQ: begin")
    A("          if (fj < %d) begin" % D)
    A("            at_load_valid <= 1'b1; at_load_data <= act[src + fj];")
    A("            fj <= fj + 1;")
    A("          end else begin")
    A("            fj <= 0; ocnt <= 0; ph <= 1'b0; st <= S_ATT;")
    A("          end")
    A("        end")
    A("        S_ARG: begin")
    A("          if (fj < %d) begin" % V)
    A("            if (fj == 0 || act[src + fj] > best) begin")
    A("              best <= act[src + fj]; best_i <= fj[%d:0];" % (tw - 1))
    A("            end")
    A("            fj <= fj + 1;")
    A("          end else begin")
    A("            next_tok <= best_i; done <= 1'b1; busy <= 1'b0;")
    A("            fj <= 0; st <= S_IDLE;")
    A("          end")
    A("        end")
    A("        default: begin")
    A("          // A block state: pulse its start, then wait until it is")
    A("          // idle and every output it owes has landed.")
    A("          if (!ph) begin")
    A("            ph <= 1'b1; ocnt <= 0;")
    A("            if (run_proj) pj_start <= 1'b1;")
    A("            else if (in_norm) rn_start <= 1'b1;")
    A("            else at_start <= 1'b1;")
    A("          end else if (!blk_busy && ocnt == expect_n && !pj_start")
    A("                       && !rn_start && !at_start) begin")
    A("            ph <= 1'b0; ocnt <= 0; fj <= 0;")
    A("            case (st)")
    A("              S_N1: st <= S_Q;")
    A("              S_Q: st <= S_K;")
    A("              S_K: st <= S_V;")
    A("              S_V: st <= S_LQ;")
    A("              S_ATT: st <= S_O;")
    A("              S_O: st <= S_R1;")
    A("              S_N2: st <= S_UP;")
    A("              S_UP: st <= S_DN;")
    A("              S_DN: st <= S_R2;")
    A("              S_HD: st <= S_ARG;")
    A("              default: st <= S_IDLE;")
    A("            endcase")
    A("          end")
    A("        end")
    A("      endcase")
    A("    end")
    A("  end")
    A("endmodule")
    return "\n".join(lines) + "\n"


def write_deps(dec, build):
    """Every generated block the decoder instantiates, in their signed-off
    form, written once. Each is derived at the checkpoint's own size, so
    the submodules they share (the MAC, matvec, the requantizer) have one
    definition between them."""
    import agent
    from agent import RuleBasedAgent
    r = RuleBasedAgent()
    ms, sp = dec.ms, dec.sp
    e, rc = specgen.derive_exp_spec(ms), specgen.derive_recip_spec(ms)
    rs = specgen.derive_rsqrt_spec(ms)
    srcs = {
        "mac_dep.v": r.render_mac(specgen.derive_chiplet_spec(ms),
                                  {agent.FIX_WIDTH, agent.FIX_CLEAR}),
        "mv_dep.v": r.render_matvec(specgen.derive_matvec_spec(ms),
                                    {agent.FIX_CLRCOL, agent.FIX_MEMLAT}),
        "rq_dep.v": r.render_requant(specgen.derive_requant_spec(ms),
                                     {agent.FIX_SATURATE}),
        "sm_dep.v": r.render_softmax(sp["softmax"], {agent.FIX_SUBMAX}),
        "expu_dep.v": r.render_exp(e, {agent.FIX_LUT}),
        "recip_dep.v": r.render_recip(rc, {agent.FIX_NORM}),
        "exp_rom.v": specgen.exp_rom(e),
        "recip_rom.v": specgen.recip_rom(rc),
        "rs_dep.v": r.render_rsqrt(rs, {agent.FIX_EVEN}),
        "rsqrt_rom.v": specgen.rsqrt_rom(rs),
        "rn_dep.v": r.render_rmsnorm(sp["rmsnorm"], {agent.FIX_EPS}),
        "at_dep.v": r.render_attn(sp["attn"], {agent.FIX_VLAT}),
        "pj_dep.v": r.render_proj(sp["proj"], {agent.FIX_PIDX}),
        "ra_dep.v": r.render_resadd(sp["resadd"], {agent.FIX_RRND}),
    }
    os.makedirs(build, exist_ok=True)
    for fn, src in srcs.items():
        with open(os.path.join(build, fn), "w") as f:
            f.write(src)
    return list(srcs)


def expected_run(dec, prompt_ids, n_gen):
    """Every step of one greedy run: (token fed, position, logits, next)."""
    dec.reset()
    steps, feed = [], list(prompt_ids)
    total = len(prompt_ids) + n_gen - 1
    assert total <= dec.L, "one context window: no slide"
    for i in range(total):
        t_ = feed[i]
        nxt, lg, _ = dec.step(t_, i)
        steps.append((t_, i, lg, nxt))
        if i + 1 >= len(feed):
            feed.append(nxt)
    return steps


def render_tb(dec, runs, top="decoder", lay=None, img=None):
    """The testbench is the host: it feeds the prompt, then feeds back
    whatever token the hardware chose, and checks every logit of every
    step against the integer reference. It prints the text as it goes.
    The Qwen-shaped decoder reuses it with its own layout and image."""
    lay = lay or layout(dec)
    V = lay["V"]
    img = img or param_image(dec)
    chars = dec.ck["chars"]
    steps = []
    for prompt_ids, n_gen in runs:
        for j, s_ in enumerate(expected_run(dec, prompt_ids, n_gen)):
            fed_by_host = s_[1] < len(prompt_ids)
            steps.append(s_ + (fed_by_host, s_[1] == 0))
    ns = len(steps)
    tw, pw = _clog2(V), _clog2(lay["L"])
    L = []
    A = L.append
    A("`timescale 1ns/1ps")
    A("// GENERATED by decoder.py: do not edit by hand.")
    A("module tb_%s;" % top)
    A("  reg clk = 0, rst_n = 0, start = 0;")
    A("  reg [%d:0] tok = 0;" % (tw - 1))
    A("  reg [%d:0] pos = 0;" % (pw - 1))
    A("  wire [%d:0] p_addr;" % (_clog2(lay["size"]) - 1))
    A("  reg signed [7:0] p_data;")
    A("  reg signed [7:0] pmem [0:%d];" % (lay["size"] - 1))
    A("  wire lg_valid, done, busy;")
    A("  wire [%d:0] lg_index, next_tok;" % (tw - 1))
    A("  wire signed [7:0] lg_data;")
    A("  reg signed [7:0] exp_lg [0:%d];" % ((ns + 1) * V - 1))
    A("  reg [%d:0] exp_next [0:%d];" % (tw - 1, ns))
    A("  reg [%d:0] feed [0:%d];" % (tw - 1, ns - 1))
    A("  reg host [0:%d];" % (ns - 1))
    A("  reg [7:0] chr [0:%d];" % (V - 1))
    A("  integer checks = 0, step = 0, nlg = 0, i, cyc = 0, t0 = 0;")
    A("  integer span = 0, lat = 0;")
    A("  reg [%d:0] prev;" % (tw - 1))
    A("  always @(posedge clk) cyc = cyc + 1;")
    A("  always #5 clk = ~clk;")
    A("  always @(posedge clk) p_data <= pmem[p_addr];")
    A("")
    A("  %s dut (.clk(clk), .rst_n(rst_n), .start(start), .tok(tok)," % top)
    A("    .pos(pos), .p_addr(p_addr), .p_data(p_data), .lg_valid(lg_valid),")
    A("    .lg_index(lg_index), .lg_data(lg_data), .next_tok(next_tok),")
    A("    .done(done), .busy(busy));")
    A("")
    A("  always @(posedge clk) begin")
    A("    if (rst_n && lg_valid) begin")
    A("      checks = checks + 1; nlg = nlg + 1;")
    A("      if (lg_data !== exp_lg[step * %d + lg_index]) begin" % V)
    A('        $display("TB_FAIL test=step%0d pos=%0d logit=%0d expected_lg=%0d got_lg=%0d",')
    A("                 step, pos, lg_index, exp_lg[step * %d + lg_index], lg_data);" % V)
    A('        $display("TB_RESULT: FAIL");')
    A("        $finish;")
    A("      end")
    A("    end")
    A("  end")
    A("")
    A("  initial begin")
    for a_, v in enumerate(img):
        if v:
            A("    pmem[%d] = %d;" % (a_, v)) if v >= 0 else \
                A("    pmem[%d] = -8'sd%d;" % (a_, -v))
        else:
            A("    pmem[%d] = 0;" % a_)
    for j, c in enumerate(chars):
        A("    chr[%d] = 8'd%d;" % (j, ord(c)))
    for s_i, (t_, p_, lg, nxt, host, first) in enumerate(steps):
        for j, v in enumerate(lg):
            A("    exp_lg[%d] = %s;" % (s_i * V + j,
                                      str(v) if v >= 0 else "-8'sd%d" % -v))
        A("    exp_next[%d] = %d; feed[%d] = %d; host[%d] = %d;"
          % (s_i, nxt, s_i, t_, s_i, 1 if host else 0))
    # The tie step: every logit equal, so the argmax has to take the
    # lowest index. The model's own logits never tie, and a design that
    # took the last maximum passed every step of a real decode.
    for j in range(V):
        A("    exp_lg[%d] = 0;" % (ns * V + j))
    A("    exp_next[%d] = 0;" % ns)
    A("    repeat (3) @(negedge clk);")
    A("    checks = checks + 1;")
    A("    if (busy !== 1'b0 || done !== 1'b0 || lg_valid !== 1'b0) begin")
    A('      $display("TB_FAIL test=reset_init expected_lg=0 got_lg=1");')
    A('      $display("TB_RESULT: FAIL");')
    A("      $finish;")
    A("    end")
    A("    rst_n = 1;")
    A("    @(negedge clk);")
    A("    for (step = 0; step < %d; step = step + 1) begin" % ns)
    A("      // The prompt comes from the host; after it, the token the")
    A("      // hardware chose last step is fed straight back.")
    A("      tok = host[step] ? feed[step] : prev;")
    A("      pos = step_pos(step);")
    A("      if (pos == 0) $write(\"\\n  \");")
    A("      if (host[step]) $write(\"%c\", chr[tok]);")
    A("      nlg = 0;")
    A("      @(negedge clk); start = 1; t0 = cyc;")
    A("      @(negedge clk); start = 0;")
    A("      while (!done) @(negedge clk);")
    A("      span = span + (cyc - t0);")
    A("      if (cyc - t0 > lat) lat = cyc - t0;")
    A("      checks = checks + 2;")
    A("      if (nlg !== %d || next_tok !== exp_next[step]) begin" % V)
    A('        $display("TB_FAIL test=step%0d pos=%0d expected_next=%0d got_next=%0d logits=%0d",')
    A("                 step, pos, exp_next[step], next_tok, nlg);")
    A('        $display("TB_RESULT: FAIL");')
    A("        $finish;")
    A("      end")
    A("      prev = next_tok;")
    A("      // Print what the hardware chose whenever it is kept: fed back")
    A("      // as the next token, or the last of a run.")
    A("      if (step + 1 == %d || step_pos(step + 1) == 0 || host[step + 1] == 1'b0)"
      % ns)
    A('        $write("%c", chr[next_tok]);')
    A("    end")
    A('    $display("");')
    A("    // Head weights zeroed: all %d logits are 0, a tie." % V)
    A("    for (i = %d; i < %d; i = i + 1) pmem[i] = 0;"
      % (lay["off"]["head"], lay["off"]["head"] + lay["D"] * V))
    A("    step = %d; nlg = 0; tok = 0; pos = 0;" % ns)
    A("    @(negedge clk); start = 1;")
    A("    @(negedge clk); start = 0;")
    A("    while (!done) @(negedge clk);")
    A("    checks = checks + 2;")
    A("    if (nlg !== %d || next_tok !== 0) begin" % V)
    A('      $display("TB_FAIL test=argmax_tie pos=0 expected_next=0 got_next=%0d logits=%0d",')
    A("               next_tok, nlg);")
    A('      $display("TB_RESULT: FAIL");')
    A("      $finish;")
    A("    end")
    A('    $display("TB_PROFILE tokens=%0d span_cycles=%0d latency_cycles=%0d",')
    A("             %d, span, lat);" % ns)
    A('    $display("TB_PASS checks=%0d", checks);')
    A('    $display("TB_RESULT: PASS");')
    A("    $finish;")
    A("  end")
    A("")
    A("  function integer step_pos(input integer s);")
    A("    begin")
    A("      case (s)")
    for s_i, st_ in enumerate(steps):
        A("        %d: step_pos = %d;" % (s_i, st_[1]))
    A("        default: step_pos = 0;")
    A("      endcase")
    A("    end")
    A("  endfunction")
    A("endmodule")
    return "\n".join(L) + "\n"


def run_rtl(dec, runs, build, timeout=1200):
    """Compile the decoder over its generated blocks and decode."""
    import subprocess
    deps = write_deps(dec, build)
    with open(os.path.join(build, "decoder.v"), "w") as f:
        f.write(render_decoder(derive_spec(dec)))
    with open(os.path.join(build, "tb_decoder.v"), "w") as f:
        f.write(render_tb(dec, runs))
    r = subprocess.run(["iverilog", "-g2005", "-o", "dec.out",
                        "tb_decoder.v", "decoder.v"] + deps, cwd=build,
                       capture_output=True, text=True)
    if r.returncode:
        return 1, r.stdout + r.stderr
    r = subprocess.run(["vvp", "dec.out"], cwd=build, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


RUNS = (("the agent", 15), ("the tools", 15))


def load_decoder():
    with open(os.path.join(ROOT, CKPT)) as f:
        return IntDecoder(json.load(f))


def generate(spec_file="spec_decoder.json", tb_file="tb_decoder.v",
             build=None):
    """Spec and testbench for the flow, and the blocks it instantiates."""
    dec = load_decoder()
    stoi = {c: i for i, c in enumerate(dec.ck["chars"])}
    runs = [([stoi[c] for c in s_], n_) for s_, n_ in RUNS]
    spec = derive_spec(dec)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_tb(dec, runs))
    if build:
        write_deps(dec, build)
    return spec


DEPS = ("mac_dep.v", "mv_dep.v", "rq_dep.v", "sm_dep.v", "expu_dep.v",
        "recip_dep.v", "exp_rom.v", "recip_rom.v", "rs_dep.v",
        "rsqrt_rom.v", "rn_dep.v", "at_dep.v", "pj_dep.v", "ra_dep.v")


def float_generate(ck, prompt_ids, n):
    out = list(prompt_ids)
    for _ in range(n):
        lg = float_step(ck, out[-ck["seq"]:])["lg"]
        out.append(max(range(len(lg)), key=lambda k: lg[k]))
    return out


def agreement(ck, dec):
    """Teacher-forced next-token agreement with the float checkpoint over
    every corpus position the context window allows."""
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    same = total = 0
    for end in range(1, len(ids)):
        ctx = ids[max(0, end - ck["seq"]):end]
        lg = float_step(ck, ctx)["lg"]
        f = max(range(len(lg)), key=lambda k: lg[k])
        dec.reset()
        for i, t in enumerate(ctx):
            nxt, _, _ = dec.step(t, i)
        same += (nxt == f)
        total += 1
    return same, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="the agent")
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()
    with open(os.path.join(ROOT, CKPT)) as f:
        ck = json.load(f)
    dec = IntDecoder(ck)
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in a.prompt if c in stoi]
    txt = lambda s: "".join(ck["chars"][i] for i in s)
    print("shift_s", dec.shift_s, "constants", dec.k_)
    same, total = agreement(ck, dec)
    print("teacher-forced agreement with float: %d/%d" % (same, total))
    print("integer :", repr(txt(dec.generate(ids, a.tokens))))
    print("float   :", repr(txt(float_generate(ck, ids, a.tokens))))


if __name__ == "__main__":
    main()
