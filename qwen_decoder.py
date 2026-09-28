"""The Qwen-shaped checkpoint as the hardware runs it: integers only.

decoder.py does this for tiny_llm.json, which has learned positions, one
head, a ReLU MLP and one layer. tiny_qwen.json has Qwen's structure: RoPE
on q and k, two query heads sharing one key/value head, the gated SiLU
MLP, two layers and a final norm. Every stage here is a generated block's
golden model with calibrated constants, so this is the reference the
decoder RTL for that structure is checked against bit for bit.

One decode step, for token t at position i, per layer l:

    xn   = rmsnorm(x, g1)
    q, k, v = proj(Wq, xn), proj(Wk, xn), proj(Wv, xn)
    q, k = rope(q, i), rope(k, i)            each head's pairs turn
    K[l][i], V[l][i] = k, v
    c[h] = attn(q[h], K[l], V[l])            both heads read KV head 0
    x    = resadd(x, proj(Wo, c))
    g, u = proj(Wg, rmsnorm(x, g2)), proj(Wu, ...)
    m    = requant(silu(g << s) * u)
    x    = resadd(x, proj(Wd, m))

then next = argmax(proj(Whead, rmsnorm(x, gf))).

Run: python3 qwen_decoder.py [--prompt "the agent"] [--tokens 40]
"""
import argparse
import json
import math
import os

import specgen
from decoder import pick, qtensor, IntDecoder
from train_qwen import CKPT, rope

ROOT = os.path.dirname(os.path.abspath(__file__))


def model_spec(ck):
    base = specgen.load_model_spec()
    return dict(base, name="tiny_qwen", n_layer=ck["n_layer"],
                d_model=ck["d_model"], n_head=ck["n_head"],
                n_kv_head=ck["n_kv_head"], head_dim=ck["head_dim"],
                d_ff=ck["d_ff"], vocab=len(ck["chars"]), seq_len=ck["seq"],
                rope_theta=ck["rope_theta"], weight_bits=8,
                activation_bits=8)


def specs(ms):
    return {"rmsnorm": specgen.derive_rmsnorm_spec(ms),
            "attn": specgen.derive_attn_spec(ms),
            "resadd": specgen.derive_resadd_spec(ms),
            "proj": specgen.derive_proj_spec(ms),
            "softmax": specgen.derive_softmax_spec(ms),
            "rope": specgen.derive_rope_spec(ms),
            "silu": specgen.derive_silu_spec(ms),
            "requant": specgen.derive_requant_spec(ms)}


# ---------------------------------------------------------------- float

def _mv(x, w):
    return [sum(x[r] * w[r][c] for r in range(len(w)))
            for c in range(len(w[0]))]


def _rms(v, g):
    inv = (sum(u * u for u in v) / len(v) + 1e-6) ** -0.5
    return [v[i] * inv * g[i] for i in range(len(v))]


def _silu(x):
    return x / (1.0 + math.exp(-x))


def float_step(ck, ids):
    """The checkpoint's forward pass, in floats, returning the last
    position's logits and the largest magnitude of every tensor at every
    layer, for calibration."""
    w = ck["weights"]
    hd, H, KV = ck["head_dim"], ck["n_head"], ck["n_kv_head"]
    grp, base = H // KV, ck["rope_theta"]
    n = len(ids)
    mx = {}

    def see(key, vals):
        mx[key] = max(mx.get(key, 0.0), max(abs(v) for v in vals))

    x = [list(w["tok"][t]) for t in ids]
    for v in x:
        see("x0", v)
    for li in range(ck["n_layer"]):
        L = lambda k: w["l%d_%s" % (li, k)]
        P = "l%d_" % li
        hn = [_rms(v, L("g1")[0]) for v in x]
        q = [_mv(v, L("wq")) for v in hn]
        k = [_mv(v, L("wk")) for v in hn]
        val = [_mv(v, L("wv")) for v in hn]
        for i in range(n):
            see(P + "xn", hn[i]); see(P + "v", val[i])
            q[i] = sum((rope(q[i][h * hd:(h + 1) * hd], i, hd, base)
                        for h in range(H)), [])
            k[i] = sum((rope(k[i][h * hd:(h + 1) * hd], i, hd, base)
                        for h in range(KV)), [])
            see(P + "q", q[i]); see(P + "k", k[i])
        for i in range(n):
            ctx = []
            for h in range(H):
                kv = h // grp
                qh = q[i][h * hd:(h + 1) * hd]
                s = [sum(a * b for a, b in
                         zip(qh, k[j][kv * hd:(kv + 1) * hd])) / math.sqrt(hd)
                     for j in range(i + 1)]
                see(P + "score", s)
                m_ = max(s)
                e = [math.exp(z - m_) for z in s]
                tot = sum(e)
                ctx += [sum(e[j] / tot * val[j][kv * hd + d]
                            for j in range(i + 1)) for d in range(hd)]
            see(P + "ctx", ctx)
            a = _mv(ctx, L("wo"))
            see(P + "a", a)
            x[i] = [x[i][j] + a[j] for j in range(len(a))]
            see(P + "x1", x[i])
        for i in range(n):
            hn2 = _rms(x[i], L("g2")[0])
            g = _mv(hn2, L("wg"))
            u = _mv(hn2, L("wu"))
            m = [_silu(gg) * uu for gg, uu in zip(g, u)]
            dn = _mv(m, L("wd"))
            see(P + "xn2", hn2); see(P + "g", g); see(P + "u", u)
            see(P + "m", m); see(P + "dn", dn)
            x[i] = [x[i][j] + dn[j] for j in range(len(dn))]
            see(P + "x2", x[i])
    xf = [_rms(v, w["gf"][0]) for v in x]
    lg = [_mv(v, w["head"]) for v in xf]
    for i in range(n):
        see("xf", xf[i]); see("lg", lg[i])
    return lg[-1], mx


def calibrate(ck):
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    L, mx = ck["seq"], {}
    for end in range(1, len(ids) + 1):
        _, m = float_step(ck, ids[max(0, end - L):end])
        for k, v in m.items():
            mx[k] = max(mx.get(k, 0.0), v)
    return mx


class QwenIntDecoder:
    """Integer-only decode of the Qwen-shaped checkpoint."""

    def __init__(self, ck):
        self.ck = ck
        self.ms = model_spec(ck)
        self.sp = specs(self.ms)
        self.D, self.F, self.L = ck["d_model"], ck["d_ff"], ck["seq"]
        self.H, self.KV, self.hd = ck["n_head"], ck["n_kv_head"], \
            ck["head_dim"]
        self.NL = ck["n_layer"]
        w = ck["weights"]
        cal = self.cal = calibrate(ck)
        s = {k: v / 127.0 for k, v in cal.items()}
        self.W, self.G = {}, {}
        names = ["tok", "head"] + ["l%d_%s" % (li, n) for li in range(self.NL)
                                   for n in ("wq", "wk", "wv", "wo", "wg",
                                             "wu", "wd")]
        for n in names:
            self.W[n], s["w_" + n] = qtensor(w[n])
        for n in ["gf"] + ["l%d_%s" % (li, g) for li in range(self.NL)
                           for g in ("g1", "g2")]:
            q, s["w_" + n] = qtensor(w[n])
            self.G[n] = q[0]
        s["x0"] = s["w_tok"]

        rq = self.sp["proj"]["parameters"]
        rn = self.sp["rmsnorm"]["parameters"]
        at = self.sp["attn"]["parameters"]
        ra = self.sp["resadd"]["parameters"]
        sm = self.sp["softmax"]["parameters"]
        sl = self.sp["silu"]["parameters"]
        sf, hd = sm["score_frac"], self.hd
        self.k_, self.shift_s, self.gshift = {}, [], []
        for li in range(self.NL):
            P = "l%d_" % li
            # Scores: fold 1/sqrt(hd) and the score format into q's scale,
            # as decoder.py does, with q measured after its rotation.
            gd = at.get("score_guard", 0)     # scores are (t << gd) >> shs
            shs = 0
            while shs + 1 < (1 << at["shift_s_width"]):
                if (math.sqrt(hd) * 2.0 ** (gd - (shs + 1) - sf)
                        / s[P + "k"] < s[P + "q"]):
                    break
                shs += 1
            s[P + "q"] = math.sqrt(hd) * 2.0 ** (gd - shs - sf) / s[P + "k"]
            self.shift_s.append(shs)
            # The gate feeds SiLU's Q4.8 port as code << gs, so its scale
            # is the power of two 2**(gs - frac); gs is the smallest that
            # keeps the calibrated gate inside int8, capped by the port.
            gs = 0
            while (s[P + "g"] > 2.0 ** (gs - sl["frac"])
                   and 127 << (gs + 1) < 1 << (sl["width"] - 1)):
                gs += 1
            s[P + "g"] = 2.0 ** (gs - sl["frac"])
            self.gshift.append(gs)
            xin = "x0" if li == 0 else "l%d_x2" % (li - 1)
            r = lambda a, b: pick(a / s[b], rq)
            self.k_.update({
                P + "q": r(s[P + "xn"] * s["w_" + P + "wq"], P + "q"),
                P + "k": r(s[P + "xn"] * s["w_" + P + "wk"], P + "k"),
                P + "v": r(s[P + "xn"] * s["w_" + P + "wv"], P + "v"),
                P + "ctx": pick(2.0 ** -at["weight_frac"] * s[P + "v"]
                                / s[P + "ctx"], at),
                P + "a": r(s[P + "ctx"] * s["w_" + P + "wo"], P + "a"),
                P + "g": r(s[P + "xn2"] * s["w_" + P + "wg"], P + "g"),
                P + "u": r(s[P + "xn2"] * s["w_" + P + "wu"], P + "u"),
                # m = silu(g) * u: silu's output is Q4.frac
                P + "m": pick(2.0 ** -sl["frac"] * s[P + "u"] / s[P + "m"],
                              self.sp["requant"]["parameters"]),
                P + "dn": r(s[P + "m"] * s["w_" + P + "wd"], P + "dn"),
            })
            ow, kk = rn["rsqrt_out_width"], rn["norm_shift"]
            for nm, g, dst in (("n1", "g1", "xn"), ("n2", "g2", "xn2")):
                self.k_[P + nm] = pick(2.0 ** (kk - ow) * math.sqrt(self.D)
                                       * s["w_" + P + g] / s[P + dst], rn)
            self.k_[P + "r1"] = IntDecoder._resadd_consts(
                s[xin] / s[P + "x1"], s[P + "a"] / s[P + "x1"], ra)
            self.k_[P + "r2"] = IntDecoder._resadd_consts(
                s[P + "x1"] / s[P + "x2"], s[P + "dn"] / s[P + "x2"], ra)
        ow, kk = rn["rsqrt_out_width"], rn["norm_shift"]
        last = "l%d_x2" % (self.NL - 1)
        self.k_["nf"] = pick(2.0 ** (kk - ow) * math.sqrt(self.D)
                             * s["w_gf"] / s["xf"], rn)
        self.k_["lg"] = pick(s["xf"] * s["w_head"] / s["lg"], rq)
        self.s = s
        self.eps = 1
        self.reset()

    def reset(self):
        self.K = [[] for _ in range(self.NL)]
        self.V = [[] for _ in range(self.NL)]

    def _proj(self, x, W, key):
        sc, sh = self.k_[key]
        return [specgen.requant_golden(
            sum(x[r] * W[r][c] for r in range(len(W))), sc, sh, 8)[0]
            for c in range(len(W[0]))]

    def _rmsnorm(self, x, g, key):
        sc, sh = self.k_[key]
        p = self.sp["rmsnorm"]
        return specgen.rmsnorm_golden(x, g, self.eps, sc, sh, p["parameters"],
                                      p["derivation"]["rsqrt"])[4]

    def _resadd(self, a, b, key):
        sa, sb, sh = self.k_[key]
        return [specgen.resadd_golden(u, v, sa, sb, sh, 8)
                for u, v in zip(a, b)]

    def _rope(self, v, heads, pos):
        rp = self.sp["rope"]
        p, fr = rp["parameters"], rp["derivation"]["freqs"]
        hd, h2 = self.hd, self.hd // 2
        out = list(v)
        for h in range(heads):
            for i in range(h2):
                a, b = v[h * hd + i], v[h * hd + i + h2]
                out[h * hd + i], out[h * hd + i + h2] = \
                    specgen.rope_golden(a, b, i, pos, p, fr)
        return out

    def step(self, tok, i):
        assert all(len(k) == i for k in self.K), "positions arrive in order"
        sl = self.sp["silu"]["parameters"]
        sd = self.sp["silu"]["derivation"]
        at, smp = self.sp["attn"]["parameters"], \
            self.sp["softmax"]["parameters"]
        hd, grp = self.hd, self.H // self.KV
        x = list(self.W["tok"][tok])
        for li in range(self.NL):
            P = "l%d_" % li
            W = lambda n: self.W[P + n]
            xn = self._rmsnorm(x, self.G[P + "g1"], P + "n1")
            q = self._rope(self._proj(xn, W("wq"), P + "q"), self.H, i)
            k = self._rope(self._proj(xn, W("wk"), P + "k"), self.KV, i)
            v = self._proj(xn, W("wv"), P + "v")
            self.K[li].append(k)
            self.V[li].append(v)
            sc, sh = self.k_[P + "ctx"]
            ctx = []
            for h in range(self.H):
                kv = h // grp
                Kh = [kk[kv * hd:(kv + 1) * hd] for kk in self.K[li]]
                Vh = [vv[kv * hd:(kv + 1) * hd] for vv in self.V[li]]
                ctx += specgen.attn_golden(q[h * hd:(h + 1) * hd], Kh, Vh,
                                           i + 1, self.shift_s[li], sc, sh,
                                           at, smp)[4]
            x = self._resadd(x, self._proj(ctx, W("wo"), P + "a"), P + "r1")
            xn2 = self._rmsnorm(x, self.G[P + "g2"], P + "n2")
            g = self._proj(xn2, W("wg"), P + "g")
            u = self._proj(xn2, W("wu"), P + "u")
            gs = self.gshift[li]
            sg = [specgen.silu_golden(c << gs, sd) for c in g]
            msc, msh = self.k_[P + "m"]
            m = [specgen.requant_golden(a * b, msc, msh, 8)[0]
                 for a, b in zip(sg, u)]
            x = self._resadd(x, self._proj(m, W("wd"), P + "dn"), P + "r2")
        xf = self._rmsnorm(x, self.G["gf"], "nf")
        lg = self._proj(xf, self.W["head"], "lg")
        nxt = max(range(len(lg)), key=lambda k: (lg[k], -k))
        return nxt, lg, {}

    def generate(self, prompt_ids, n):
        out = list(prompt_ids)
        for _ in range(n):
            ctx = out[-self.L:]
            self.reset()
            for i, t in enumerate(ctx):
                nxt, _, _ = self.step(t, i)
            out.append(nxt)
        return out


# ---------------------------------------------------------------- RTL

REGIONS = ("x", "n", "q", "k", "c", "a", "g", "u", "m", "lg")
FIX_ROPEK = "rotate_the_keys_as_well_as_the_queries"


def _clog2(n):
    return max(1, (n - 1).bit_length())


def layout(dec):
    D, F, V, L = dec.D, dec.F, len(dec.ck["chars"]), dec.L
    QW, KW = dec.H * dec.hd, dec.KV * dec.hd
    off, p = {}, 0
    items = [("tok", V * D)]
    for li in range(dec.NL):
        P = "l%d_" % li
        items += [(P + "g1", D), (P + "g2", D), (P + "wq", D * QW),
                  (P + "wk", D * KW), (P + "wv", D * KW), (P + "wo", QW * D),
                  (P + "wg", D * F), (P + "wu", D * F), (P + "wd", F * D)]
    items += [("gf", D), ("head", D * V)]
    for name, size in items:
        off[name] = p
        p += size
    return {"D": D, "F": F, "V": V, "L": L, "QW": QW, "KW": KW,
            "B": max(D, F, V, QW), "off": off, "size": p}


def param_image(dec):
    lay = layout(dec)
    img = [0] * lay["size"]
    D = lay["D"]
    for t_, row in enumerate(dec.W["tok"]):
        for j, v in enumerate(row):
            img[lay["off"]["tok"] + t_ * D + j] = v
    for name, g in dec.G.items():
        for j, v in enumerate(g):
            img[lay["off"][name] + j] = v
    for name, W in dec.W.items():
        if name == "tok":
            continue
        depth = len(W)
        for c in range(len(W[0])):
            for r in range(depth):
                img[lay["off"][name] + c * depth + r] = W[r][c]
    return img


def _port_widths(sp):
    rn, at, pj, ro, sl, rq = (sp[b]["parameters"] for b in
                              ("rmsnorm", "attn", "proj", "rope", "silu",
                               "requant"))
    return {"rn_addr": rn["addr_width"], "rn_eps": rn["rsqrt_in_width"],
            "at_addr": at["addr_width"], "at_hd": at["head_dim_width"],
            "at_n": at["n_width"], "at_shs": at["shift_s_width"],
            "pj_depth": pj["depth_width"], "pj_col": pj["col_width"],
            "pj_addr": pj["addr_width"], "sc": pj["scale_width"],
            "sh": pj["shift_width"], "ro_idx": ro["index_width"],
            "ro_pos": ro["pos_width"], "si_w": sl["width"],
            "rq_acc": rq["acc_width"]}


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
    lay = layout(dec)
    return {
        "name": "qwen_decoder_%s" % dec.ms["name"],
        "description": "One decode step of the Qwen-shaped checkpoint: %d "
                       "layers, %d query heads over %d key/value head, "
                       "RoPE, the gated SiLU MLP, a final norm, embedding "
                       "lookup to argmax" % (dec.NL, dec.H, dec.KV),
        "top_module": "qwen_decoder",
        "unit": "token",
        "parameters": {
            "d_model": lay["D"], "d_ff": lay["F"], "vocab": lay["V"],
            "context": lay["L"], "n_head": dec.H, "n_kv_head": dec.KV,
            "head_dim": dec.hd, "n_layer": dec.NL, "region": lay["B"],
            "param_offsets": lay["off"], "param_words": lay["size"],
            "constants": {k_: list(v) for k_, v in dec.k_.items()},
            "shift_s": dec.shift_s, "gate_shift": dec.gshift,
            "eps": dec.eps, "ports": _port_widths(dec.sp),
            "data_width": 8, "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "ports": _ports(lay),
        "derivation": {
            "model": dec.ms["name"],
            "rule": "block specs derived at the checkpoint's own size; "
                    "static scales by calibration; the gate's scale a "
                    "power of two so its code shifts straight into SiLU",
        },
    }


def render_qwen_decoder(spec, fixes=frozenset((FIX_ROPEK,))):
    """One decode step of the Qwen-shaped model over the generated blocks.

    One instance each of the projection, RMSNorm, attention head, residual
    add, rotary unit, SiLU and requantizer serves every use of it, across
    both layers. The layer and the query head are loop counters; per-layer
    constants and weight bases are selected by the layer.

    Every activation has its own small memory with one write port, so it
    maps to LUT RAM, and the KV cache to block RAM. q, k and the key cache
    are split by rotary half, x[i] and x[i + d/2] in separate memories, so
    the rotary unit reads and writes one element of each per cycle. As one
    shared array with two writes a cycle, all of it was flip-flops and
    multiplexers: 36289 LUTs on 7-series, too big for the Basys 3.

    The seeded first cut stores keys in the cache without rotating them.
    At position 0 the rotation is the identity, so the first step passes.
    """
    p = spec["parameters"]
    D, F, V, L = p["d_model"], p["d_ff"], p["vocab"], p["context"]
    H, KV, hd, NL = p["n_head"], p["n_kv_head"], p["head_dim"], p["n_layer"]
    grp, h2 = H // KV, hd // 2
    QW, KW = H * hd, KV * hd
    assert h2 & (h2 - 1) == 0, "rotary pairs are indexed by shifts"
    assert hd & (hd - 1) == 0, "cache slots are indexed by shifts"
    off, k = p["param_offsets"], p["constants"]
    w = p["ports"]
    paw = _clog2(p["param_words"])
    tw, pw = _clog2(V), _clog2(L)
    kvn = NL * KV * L
    bw = _clog2(max(D, F, V, QW) + 1)
    lw = _clog2(NL + 1)
    hw = _clog2(H + 1)
    h2b, hdb = _clog2(h2), _clog2(hd)
    states = ["S_IDLE", "S_CPTOK", "S_N1", "S_Q", "S_K", "S_V", "S_RQ",
              "S_RK", "S_LQ", "S_ATT", "S_O", "S_R1", "S_N2", "S_G", "S_U",
              "S_GLU", "S_DN", "S_R2", "S_NF", "S_HD", "S_ARG"]
    sw = _clog2(len(states))
    rotate_k = FIX_ROPEK in fixes
    A = []
    a = A.append
    a("// GENERATED by qwen_decoder.py: do not edit by hand.")
    a("// One decode step of the Qwen-shaped checkpoint over the generated")
    a("// blocks: %d layers, %d query heads over %d KV head, RoPE, SwiGLU."
      % (NL, H, KV))
    a("module qwen_decoder (")
    a("  input                    clk,")
    a("  input                    rst_n,")
    a("  input                    start,")
    a("  input      [%d:0] tok," % (tw - 1))
    a("  input      [%d:0] pos," % (pw - 1))
    a("  output     [%d:0] p_addr," % (paw - 1))
    a("  input      signed [7:0] p_data,")
    a("  output reg               lg_valid,")
    a("  output reg [%d:0] lg_index," % (tw - 1))
    a("  output reg signed [7:0] lg_data,")
    a("  output reg [%d:0] next_tok," % (tw - 1))
    a("  output reg               done,")
    a("  output reg               busy")
    a(");")
    a("  localparam " + ", ".join("%s = %d'd%d" % (s_, sw, i)
                                  for i, s_ in enumerate(states)) + ";")
    a("  reg [%d:0] st;" % (sw - 1))
    a("  reg ph;")
    a("  reg [%d:0] lyr;" % (lw - 1))
    a("  reg [%d:0] hh;" % (hw - 1))
    a("  reg [%d:0] tok_r;" % (tw - 1))
    a("  reg [%d:0] pos_r;" % (pw - 1))
    a("  // One memory per activation, each with a single write port.")
    for name, size in (("xm", D), ("nm", D), ("qlo", QW // 2),
                       ("qhi", QW // 2), ("klo", KW // 2), ("khi", KW // 2),
                       ("cm", QW), ("am", D), ("gm", F), ("um", F),
                       ("mm", F), ("lgm", V)):
        a("  reg signed [7:0] %s [0:%d];" % (name, size - 1))
    a("  // KV cache, [layer][kv head][position][...]: keys split by rotary")
    a("  // half so each half takes one write a cycle.")
    a("  reg signed [7:0] kclo [0:%d];" % (kvn * h2 - 1))
    a("  reg signed [7:0] kchi [0:%d];" % (kvn * h2 - 1))
    a("  reg signed [7:0] vc [0:%d];" % (kvn * hd - 1))
    a("  reg [%d:0] fj, ocnt, gcnt;" % (bw - 1))
    a("  reg cp_v;")
    a("  reg [%d:0] cp_i;" % (bw - 1))
    a("")
    a("  reg pj_start;")
    a("  reg [%d:0] pj_depth, pj_cols;" % (w["pj_depth"] - 1))
    a("  reg [%d:0] pj_scale, rn_scale, ra_sa, ra_sb, rq_scale;" % (w["sc"] - 1))
    a("  reg [%d:0] pj_shift, rn_shift, ra_sh, rq_shift;" % (w["sh"] - 1))
    a("  wire [%d:0] pj_a_addr;" % (w["pj_depth"] - 1))
    a("  wire [%d:0] pj_w_addr;" % (w["pj_addr"] - 1))
    a("  reg signed [7:0] pj_a_data;")
    a("  wire pj_valid, pj_busy;")
    a("  wire [%d:0] pj_index;" % (w["pj_col"] - 1))
    a("  wire signed [7:0] pj_data;")
    a("  proj u_proj (.clk(clk), .rst_n(rst_n), .start(pj_start),")
    a("    .depth(pj_depth), .cols(pj_cols), .scale(pj_scale),")
    a("    .shift(pj_shift), .a_addr(pj_a_addr), .a_data(pj_a_data),")
    a("    .w_addr(pj_w_addr), .w_data(p_data), .o_valid(pj_valid),")
    a("    .o_index(pj_index), .o_data(pj_data), .busy(pj_busy));")
    a("")
    a("  reg rn_start;")
    a("  wire [%d:0] rn_x_addr, rn_g_addr, rn_index;" % (w["rn_addr"] - 1))
    a("  reg signed [7:0] rn_x_data;")
    a("  wire rn_valid, rn_busy;")
    a("  wire signed [7:0] rn_data;")
    a("  rmsnorm u_norm (.clk(clk), .rst_n(rst_n), .start(rn_start),")
    a("    .eps(%d'd%d), .scale_o(rn_scale), .shift_o(rn_shift),"
      % (w["rn_eps"], p["eps"]))
    a("    .x_addr(rn_x_addr), .x_data(rn_x_data), .g_addr(rn_g_addr),")
    a("    .g_data(p_data), .o_valid(rn_valid), .o_index(rn_index),")
    a("    .o_data(rn_data), .busy(rn_busy));")
    a("")
    a("  reg at_start, at_load_valid;")
    a("  reg signed [7:0] at_load_data;")
    a("  reg [%d:0] at_shs;" % (w["at_shs"] - 1))
    a("  reg [%d:0] at_scale;" % (w["sc"] - 1))
    a("  reg [%d:0] at_shift;" % (w["sh"] - 1))
    a("  wire [%d:0] at_k_addr, at_v_addr;" % (w["at_addr"] - 1))
    a("  reg signed [7:0] at_klo, at_khi, at_v_data;")
    a("  reg at_ksel;")
    a("  wire signed [7:0] at_k_data = at_ksel ? at_khi : at_klo;")
    a("  wire at_valid, at_busy;")
    a("  wire [%d:0] at_index;" % (w["at_hd"] - 1))
    a("  wire signed [7:0] at_data;")
    a("  attn u_attn (.clk(clk), .rst_n(rst_n), .load_valid(at_load_valid),")
    a("    .load_data(at_load_data), .start(at_start),")
    a("    .n({%d'd0, pos_r} + %d'd1), .shift_s(at_shs)," % (w["at_n"] - pw, w["at_n"]))
    a("    .scale_o(at_scale), .shift_o(at_shift),")
    a("    .k_addr(at_k_addr), .k_data(at_k_data), .v_addr(at_v_addr),")
    a("    .v_data(at_v_data), .o_valid(at_valid), .o_index(at_index),")
    a("    .o_data(at_data), .busy(at_busy));")
    a("")
    a("  reg signed [7:0] ra_a, ra_b;")
    a("  reg ra_v;")
    a("  wire signed [7:0] ra_y;")
    a("  wire ra_vout;")
    a("  resadd u_add (.clk(clk), .rst_n(rst_n), .a(ra_a), .b(ra_b),")
    a("    .scale_a(ra_sa), .scale_b(ra_sb), .shift(ra_sh),")
    a("    .valid_in(ra_v), .y(ra_y), .valid_out(ra_vout));")
    a("")
    a("  reg signed [7:0] ro_x1, ro_x2;")
    a("  reg [%d:0] ro_idx;" % (w["ro_idx"] - 1))
    a("  reg ro_v;")
    a("  wire signed [7:0] ro_y1, ro_y2;")
    a("  wire ro_vout;")
    a("  rope u_rope (.clk(clk), .rst_n(rst_n), .x1(ro_x1), .x2(ro_x2),")
    a("    .idx(ro_idx), .pos(pos_r), .valid_in(ro_v), .y1(ro_y1),")
    a("    .y2(ro_y2), .valid_out(ro_vout));")
    a("")
    a("  reg signed [%d:0] si_x;" % (w["si_w"] - 1))
    a("  reg si_v;")
    a("  wire signed [%d:0] si_y;" % (w["si_w"] - 1))
    a("  wire si_vout;")
    a("  silu u_silu (.clk(clk), .rst_n(rst_n), .x(si_x), .valid_in(si_v),")
    a("    .y(si_y), .valid_out(si_vout));")
    a("  reg signed [%d:0] rq_acc;" % (w["rq_acc"] - 1))
    a("  reg rq_v;")
    a("  wire signed [7:0] rq_q;")
    a("  wire rq_sat, rq_vout;")
    a("  requant u_rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc),")
    a("    .scale(rq_scale), .shift(rq_shift), .valid_in(rq_v),")
    a("    .q_out(rq_q), .sat(rq_sat), .valid_out(rq_vout));")
    a("")
    a("  // ---- per-state, per-layer constants and weight bases")
    a("  reg [%d:0] pbase;" % (paw - 1))
    a("  reg [%d:0] expect_n;" % (bw - 1))
    a("  reg [3:0] gsh;")
    a("  always @(*) begin")
    a("    pbase = 0; expect_n = %d;" % D)
    a("    pj_depth = 0; pj_cols = 0; pj_scale = 0; pj_shift = 0;")
    a("    rn_scale = 0; rn_shift = 0; ra_sa = 0; ra_sb = 0; ra_sh = 0;")
    a("    rq_scale = 0; rq_shift = 0; at_shs = 0; at_scale = 0;")
    a("    at_shift = 0; gsh = 0;")
    a("    case (st)")
    a("      S_CPTOK: pbase = %d + tok_r * %d;" % (off["tok"], D))
    a("      S_NF: begin pbase = %d; rn_scale = %d; rn_shift = %d; end"
      % ((off["gf"],) + tuple(k["nf"])))
    a("      S_HD: begin pbase = %d; pj_depth = %d; pj_cols = %d;"
      % (off["head"], D, V))
    a("        pj_scale = %d; pj_shift = %d; expect_n = %d; end"
      % (k["lg"][0], k["lg"][1], V))
    a("      S_ARG: expect_n = %d;" % V)
    per = {}
    for li in range(NL):
        P = "l%d_" % li
        c = lambda key: k[P + key]
        per.setdefault("S_N1", []).append(
            "pbase = %d; rn_scale = %d; rn_shift = %d;"
            % ((off[P + "g1"],) + tuple(c("n1"))))
        per.setdefault("S_N2", []).append(
            "pbase = %d; rn_scale = %d; rn_shift = %d;"
            % ((off[P + "g2"],) + tuple(c("n2"))))
        for s_, w_, dep, cols, key in (
                ("S_Q", "wq", D, QW, "q"), ("S_K", "wk", D, KW, "k"),
                ("S_V", "wv", D, KW, "v"), ("S_O", "wo", QW, D, "a"),
                ("S_G", "wg", D, F, "g"), ("S_U", "wu", D, F, "u"),
                ("S_DN", "wd", F, D, "dn")):
            per.setdefault(s_, []).append(
                "pbase = %d; pj_depth = %d; pj_cols = %d; pj_scale = %d; "
                "pj_shift = %d; expect_n = %d;"
                % ((off[P + w_], dep, cols) + tuple(c(key)) + (cols,)))
        per.setdefault("S_R1", []).append(
            "ra_sa = %d; ra_sb = %d; ra_sh = %d;" % tuple(c("r1")))
        per.setdefault("S_R2", []).append(
            "ra_sa = %d; ra_sb = %d; ra_sh = %d;" % tuple(c("r2")))
        per.setdefault("S_ATT", []).append(
            "at_shs = %d; at_scale = %d; at_shift = %d; expect_n = %d;"
            % ((p["shift_s"][li],) + tuple(c("ctx")) + (hd,)))
        per.setdefault("S_GLU", []).append(
            "rq_scale = %d; rq_shift = %d; gsh = %d; expect_n = %d;"
            % (tuple(c("m")) + (p["gate_shift"][li], F)))
        per.setdefault("S_RQ", []).append("expect_n = %d;" % (H * h2))
        per.setdefault("S_RK", []).append("expect_n = %d;" % (KV * h2))
    for s_, arms in per.items():
        a("      %s: case (lyr)" % s_)
        for li, body in enumerate(arms):
            a("        %d: begin %s end" % (li, body))
        a("        default: ;")
        a("      endcase")
    a("      default: ;")
    a("    endcase")
    a("  end")
    a("")
    a("  wire in_norm = (st == S_N1) || (st == S_N2) || (st == S_NF);")
    a("  assign p_addr = in_norm ? pbase + rn_g_addr")
    a("                : (st == S_CPTOK) ? pbase + fj")
    a("                : pbase + pj_w_addr;")
    a("  // Where this layer's cache starts, per KV head.")
    a("  wire [%d:0] kvrow = (lyr * %d + hh / %d) * %d;"
      % (_clog2(kvn) - 1, KV, grp, L))
    a("  wire [%d:0] j_at = at_k_addr >> %d;" % (w["at_addr"] - 1, hdb))
    a("  wire [%d:0] d_at = at_k_addr & %d;" % (hdb - 1, hd - 1))
    a("  wire [%d:0] kslot = (lyr * %d) * %d + pos_r;" % (_clog2(kvn) - 1, KV, L))
    a("  always @(posedge clk) begin")
    a("    rn_x_data <= xm[rn_x_addr];")
    a("    pj_a_data <= (st == S_O) ? cm[pj_a_addr]")
    a("               : (st == S_DN) ? mm[pj_a_addr] : nm[pj_a_addr];")
    a("    at_klo <= kclo[(kvrow + j_at) * %d + d_at[%d:0]];" % (h2, h2b - 1))
    a("    at_khi <= kchi[(kvrow + j_at) * %d + d_at[%d:0]];" % (h2, h2b - 1))
    a("    at_ksel <= d_at[%d];" % h2b)
    a("    at_v_data <= vc[kvrow * %d + at_v_addr];" % hd)
    a("  end")
    a("")
    a("  wire run_proj = (st == S_Q) || (st == S_K) || (st == S_V) ||")
    a("                  (st == S_O) || (st == S_G) || (st == S_U) ||")
    a("                  (st == S_DN) || (st == S_HD);")
    a("  wire blk_busy = run_proj ? pj_busy : in_norm ? rn_busy : at_busy;")
    a("  // A projection output's place in a rotary-split vector.")
    a("  wire [%d:0] pj_half = pj_index[%d];" % (0, h2b))
    a("  wire [%d:0] pj_hadr = ((pj_index >> %d) << %d) | (pj_index & %d);"
      % (bw - 1, hdb, h2b, h2 - 1))
    a("  // The cache row of a key pair the rotary unit hands back.")
    a("  wire [%d:0] ok_row = kslot + (ocnt >> %d) * %d;"
      % (_clog2(kvn) - 1, h2b, L))
    a("  wire [%d:0] ok_p = ocnt & %d;" % (h2b - 1, h2 - 1))
    a("  reg signed [7:0] best;")
    a("  reg [%d:0] best_i;" % (tw - 1))
    a("")
    a("  always @(posedge clk) begin")
    a("    if (!rst_n) begin")
    a("      st <= S_IDLE; ph <= 1'b0; busy <= 1'b0; done <= 1'b0;")
    a("      lyr <= 0; hh <= 0; tok_r <= 0; pos_r <= 0; next_tok <= 0;")
    a("      fj <= 0; ocnt <= 0; gcnt <= 0; cp_v <= 1'b0; cp_i <= 0;")
    a("      pj_start <= 1'b0; rn_start <= 1'b0; at_start <= 1'b0;")
    a("      at_load_valid <= 1'b0; at_load_data <= 0; ra_a <= 0; ra_b <= 0;")
    a("      ra_v <= 1'b0; ro_x1 <= 0; ro_x2 <= 0; ro_idx <= 0; ro_v <= 1'b0;")
    a("      si_x <= 0; si_v <= 1'b0; rq_acc <= 0; rq_v <= 1'b0;")
    a("      lg_valid <= 1'b0; lg_index <= 0; lg_data <= 0; best <= 0;")
    a("      best_i <= 0;")
    a("    end else begin")
    a("      done <= 1'b0; pj_start <= 1'b0; rn_start <= 1'b0;")
    a("      at_start <= 1'b0; at_load_valid <= 1'b0; ra_v <= 1'b0;")
    a("      ro_v <= 1'b0; si_v <= 1'b0; rq_v <= 1'b0; lg_valid <= 1'b0;")
    a("      cp_v <= 1'b0;")
    a("      if (run_proj && pj_valid) begin")
    a("        ocnt <= ocnt + 1;")
    a("        case (st)")
    a("          S_Q: if (pj_half) qhi[pj_hadr] <= pj_data;")
    a("               else qlo[pj_hadr] <= pj_data;")
    a("          S_K: if (pj_half) khi[pj_hadr] <= pj_data;")
    a("               else klo[pj_hadr] <= pj_data;")
    a("          S_V: vc[(kslot + (pj_index >> %d) * %d) * %d + (pj_index & %d)] <= pj_data;"
      % (hdb, L, hd, hd - 1))
    a("          S_O, S_DN: am[pj_index] <= pj_data;")
    a("          S_G: gm[pj_index] <= pj_data;")
    a("          S_U: um[pj_index] <= pj_data;")
    a("          S_HD: begin")
    a("            lgm[pj_index] <= pj_data;")
    a("            lg_valid <= 1'b1; lg_index <= pj_index[%d:0];" % (tw - 1))
    a("            lg_data <= pj_data;")
    a("          end")
    a("          default: ;")
    a("        endcase")
    a("      end")
    a("      if (in_norm && rn_valid) begin")
    a("        nm[rn_index] <= rn_data; ocnt <= ocnt + 1;")
    a("      end")
    a("      if (st == S_ATT && at_valid) begin")
    a("        cm[hh * %d + at_index] <= at_data; ocnt <= ocnt + 1;" % hd)
    a("      end")
    a("      if ((st == S_R1 || st == S_R2) && ra_vout) begin")
    a("        xm[ocnt] <= ra_y; ocnt <= ocnt + 1;")
    a("      end")
    a("      // Rotated pairs go back where they came from; keys to the cache.")
    a("      if (st == S_RQ && ro_vout) begin")
    a("        qlo[ocnt] <= ro_y1; qhi[ocnt] <= ro_y2; ocnt <= ocnt + 1;")
    a("      end")
    a("      if (st == S_RK && ro_vout) begin")
    if rotate_k:
        a("        kclo[ok_row * %d + ok_p] <= ro_y1;" % h2)
        a("        kchi[ok_row * %d + ok_p] <= ro_y2;" % h2)
    else:
        a("        kclo[ok_row * %d + ok_p] <= klo[ocnt];" % h2)
        a("        kchi[ok_row * %d + ok_p] <= khi[ocnt];" % h2)
    a("        ocnt <= ocnt + 1;")
    a("      end")
    a("      // SiLU of the gate, times up, requantized: one stream.")
    a("      if (st == S_GLU && si_vout) begin")
    a("        rq_acc <= si_y * um[gcnt]; rq_v <= 1'b1; gcnt <= gcnt + 1;")
    a("      end")
    a("      if (st == S_GLU && rq_vout) begin")
    a("        mm[ocnt] <= rq_q; ocnt <= ocnt + 1;")
    a("      end")
    a("      if (cp_v) xm[cp_i] <= p_data;")
    a("")
    a("      case (st)")
    a("        S_IDLE: if (start) begin")
    a("          tok_r <= tok; pos_r <= pos; busy <= 1'b1; lyr <= 0; hh <= 0;")
    a("          st <= S_CPTOK; fj <= 0; ocnt <= 0; ph <= 1'b0;")
    a("        end")
    a("        S_CPTOK: begin")
    a("          if (fj < %d) begin" % D)
    a("            cp_v <= 1'b1; cp_i <= fj; fj <= fj + 1;")
    a("          end else if (!cp_v) begin")
    a("            fj <= 0; ocnt <= 0; st <= S_N1;")
    a("          end")
    a("        end")
    a("        S_R1, S_R2: begin")
    a("          if (fj < %d) begin" % D)
    a("            ra_a <= xm[fj]; ra_b <= am[fj]; ra_v <= 1'b1; fj <= fj + 1;")
    a("          end")
    a("          if (ocnt == %d) begin" % D)
    a("            fj <= 0; ocnt <= 0; ph <= 1'b0;")
    a("            if (st == S_R1) st <= S_N2;")
    a("            else if (lyr + 1 < %d) begin lyr <= lyr + 1; st <= S_N1; end" % NL)
    a("            else st <= S_NF;")
    a("          end")
    a("        end")
    a("        S_RQ, S_RK: begin")
    a("          if (fj < expect_n) begin")
    a("            ro_x1 <= (st == S_RQ) ? qlo[fj] : klo[fj];")
    a("            ro_x2 <= (st == S_RQ) ? qhi[fj] : khi[fj];")
    a("            ro_idx <= fj[%d:0]; ro_v <= 1'b1; fj <= fj + 1;" % (w["ro_idx"] - 1))
    a("          end")
    a("          if (ocnt == expect_n) begin")
    a("            fj <= 0; ocnt <= 0; ph <= 1'b0;")
    a("            st <= (st == S_RQ) ? S_RK : S_LQ;")
    a("          end")
    a("        end")
    a("        S_LQ: begin")
    a("          if (fj < %d) begin" % hd)
    a("            at_load_valid <= 1'b1;")
    a("            at_load_data <= fj[%d] ? qhi[hh * %d + fj[%d:0]]"
      % (h2b, h2, h2b - 1))
    a("                                : qlo[hh * %d + fj[%d:0]];" % (h2, h2b - 1))
    a("            fj <= fj + 1;")
    a("          end else begin")
    a("            fj <= 0; ocnt <= 0; ph <= 1'b0; st <= S_ATT;")
    a("          end")
    a("        end")
    a("        S_GLU: begin")
    a("          if (fj < %d) begin" % F)
    a("            si_x <= gm[fj] <<< gsh; si_v <= 1'b1; fj <= fj + 1;")
    a("          end")
    a("          if (ocnt == %d) begin" % F)
    a("            fj <= 0; ocnt <= 0; gcnt <= 0; ph <= 1'b0; st <= S_DN;")
    a("          end")
    a("        end")
    a("        S_ARG: begin")
    a("          if (fj < %d) begin" % V)
    a("            if (fj == 0 || lgm[fj] > best) begin")
    a("              best <= lgm[fj]; best_i <= fj[%d:0];" % (tw - 1))
    a("            end")
    a("            fj <= fj + 1;")
    a("          end else begin")
    a("            next_tok <= best_i; done <= 1'b1; busy <= 1'b0;")
    a("            fj <= 0; st <= S_IDLE;")
    a("          end")
    a("        end")
    a("        default: begin")
    a("          if (!ph) begin")
    a("            ph <= 1'b1; ocnt <= 0;")
    a("            if (run_proj) pj_start <= 1'b1;")
    a("            else if (in_norm) rn_start <= 1'b1;")
    a("            else at_start <= 1'b1;")
    a("          end else if (!blk_busy && ocnt == expect_n && !pj_start")
    a("                       && !rn_start && !at_start) begin")
    a("            ph <= 1'b0; ocnt <= 0; fj <= 0;")
    a("            case (st)")
    a("              S_N1: st <= S_Q;")
    a("              S_Q: st <= S_K;")
    a("              S_K: st <= S_V;")
    a("              S_V: st <= S_RQ;")
    a("              S_ATT: if (hh + 1 < %d) begin hh <= hh + 1; st <= S_LQ; end" % H)
    a("                     else begin hh <= 0; st <= S_O; end")
    a("              S_O: st <= S_R1;")
    a("              S_N2: st <= S_G;")
    a("              S_G: st <= S_U;")
    a("              S_U: st <= S_GLU;")
    a("              S_DN: st <= S_R2;")
    a("              S_NF: st <= S_HD;")
    a("              S_HD: st <= S_ARG;")
    a("              default: st <= S_IDLE;")
    a("            endcase")
    a("          end")
    a("        end")
    a("      endcase")
    a("    end")
    a("  end")
    a("endmodule")
    return "\n".join(A) + "\n"


RUNS = (("the agent", 15), ("the tools", 15))
DEPS = ("mac_dep.v", "mv_dep.v", "rq_dep.v", "sm_dep.v", "expu_dep.v",
        "recip_dep.v", "exp_rom.v", "recip_rom.v", "rs_dep.v",
        "rsqrt_rom.v", "rn_dep.v", "at_dep.v", "pj_dep.v", "ra_dep.v",
        "ro_dep.v", "rope_rom.v", "si_dep.v")


def write_deps(dec, build):
    """Every block the decoder instantiates, at the checkpoint's size."""
    import agent
    import decoder
    decoder.write_deps(dec, build)
    r = agent.RuleBasedAgent()
    srcs = {"ro_dep.v": r.render_rope(dec.sp["rope"], {agent.FIX_ROTDIR}),
            "rope_rom.v": specgen.rope_roms(dec.sp["rope"]),
            "si_dep.v": r.render_silu(dec.sp["silu"], {agent.FIX_SIGN})}
    for fn, src in srcs.items():
        with open(os.path.join(build, fn), "w") as f:
            f.write(src)
    return list(DEPS)


def render_tb(dec, runs):
    import decoder
    return decoder.render_tb(dec, runs, top="qwen_decoder",
                             lay=layout(dec), img=param_image(dec))


def generate(spec_file="spec_qwen_decoder.json",
             tb_file="tb_qwen_decoder.v", build=None):
    dec = load()
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


def run_rtl(dec, runs, build, fixes=frozenset((FIX_ROPEK,)), timeout=1800):
    import subprocess
    deps = write_deps(dec, build)
    with open(os.path.join(build, "qwen_decoder.v"), "w") as f:
        f.write(render_qwen_decoder(derive_spec(dec), fixes))
    with open(os.path.join(build, "tb_qwen_decoder.v"), "w") as f:
        f.write(render_tb(dec, runs))
    r = subprocess.run(["iverilog", "-g2005", "-o", "qd.out",
                        "tb_qwen_decoder.v", "qwen_decoder.v"] + deps,
                       cwd=build, capture_output=True, text=True)
    if r.returncode:
        return 1, r.stdout + r.stderr
    r = subprocess.run(["vvp", "qd.out"], cwd=build, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def float_generate(ck, prompt_ids, n):
    out = list(prompt_ids)
    for _ in range(n):
        lg, _ = float_step(ck, out[-ck["seq"]:])
        out.append(max(range(len(lg)), key=lambda k: lg[k]))
    return out


def agreement(ck, dec):
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    same = total = 0
    for end in range(1, len(ids)):
        ctx = ids[max(0, end - ck["seq"]):end]
        lg, _ = float_step(ck, ctx)
        f = max(range(len(lg)), key=lambda k: lg[k])
        dec.reset()
        for i, t in enumerate(ctx):
            nxt, _, _ = dec.step(t, i)
        same += (nxt == f)
        total += 1
    return same, total


def load():
    with open(os.path.join(ROOT, CKPT)) as f:
        return QwenIntDecoder(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="the agent")
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()
    dec = load()
    ck = dec.ck
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in a.prompt if c in stoi]
    txt = lambda s_: "".join(ck["chars"][i] for i in s_)
    print("shift_s", dec.shift_s, "gate shifts", dec.gshift)
    same, total = agreement(ck, dec)
    print("teacher-forced agreement with float: %d/%d" % (same, total))
    print("integer :", repr(txt(dec.generate(ids, a.tokens))))
    print("float   :", repr(txt(float_generate(ck, ids, a.tokens))))


if __name__ == "__main__":
    main()
