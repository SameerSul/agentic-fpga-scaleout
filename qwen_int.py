"""The real Qwen2.5-0.5B, or Qwen3-0.6B, on the generated blocks' arithmetic.

qwen_real.py runs the checkpoint in float. This runs it the way the
hardware would: int8 weights, activations at a fixed width with scales
fixed by calibration, and every stage one of the generated blocks' own
golden models, derived at Qwen's size (RMSNorm over 896, attention heads
of 64 with the wide score path, RoPE with theta 1e6, SiLU, the residual
add, the requantizer). The q, k and v projections carry biases, which are
added to the accumulator before requantization, in the accumulator's
own units. Qwen3 (FPGAI_QWEN=qwen3) has no biases; instead each head of q
and k goes through its own RMSNorm, the same block derived for a row of
head_dim, before RoPE.

Run: python3 qwen_int.py [--act-bits 16] [--per-channel] [--eval 24]
"""
import argparse
import array
import json
import math
import operator
import os
import sys
import time

import specgen
import qwen_real as qr

ROOT = os.path.dirname(os.path.abspath(__file__))
CALIB = ("The quick brown fox jumps over the lazy dog. In 1492, Columbus "
         "sailed across the Atlantic Ocean, and the history of two "
         "continents changed.")
EVAL = ("Paris is the capital and largest city of France. The city is "
        "known for its museums, its architecture and the river that runs "
        "through it.")


def model_spec(cfg, act_bits):
    base = specgen.load_model_spec()
    name = {"qwen2.5": "qwen2_5_0p5b", "qwen3": "qwen3_0p6b"}[qr.MODEL]
    return dict(base, name="%s_a%d" % (name, act_bits),
                d_model=cfg["hidden_size"], d_ff=cfg["intermediate_size"],
                n_head=cfg["num_attention_heads"],
                n_kv_head=cfg["num_key_value_heads"],
                head_dim=qr.head_dim(cfg),
                n_layer=cfg["num_hidden_layers"], vocab=cfg["vocab_size"],
                seq_len=256, rope_theta=cfg["rope_theta"], weight_bits=8,
                activation_bits=act_bits)


def pick(ratio, mw, sw):
    best = (1, 1)
    for sh in range(1, min(1 << sw, 80)):
        sc = int(round(ratio * (1 << sh)))
        if sc >= 1 << mw:
            break
        if sc >= 1:
            best = (sc, sh)
    return best


def quantize_rows(Wf, rows, cols, per_channel):
    """int8 weights, one scale per output row or one for the tensor."""
    q = array.array("b", bytes(rows * cols))
    scales = []
    if not per_channel:
        s = max(abs(min(Wf)), abs(max(Wf))) / 127.0 or 1.0
    for r in range(rows):
        row = Wf[r * cols:(r + 1) * cols]
        if per_channel:
            s = max(abs(min(row)), abs(max(row))) / 127.0 or 1.0
        inv = 1.0 / s
        q[r * cols:(r + 1) * cols] = array.array(
            "b", [max(-127, min(127, int(round(v * inv)))) for v in row])
        scales.append(s)
    return q, scales


class IntQwen:
    def __init__(self, cfg, W, act_bits=8, per_channel=True, calib=None,
                 log=print, probe=None, exact_io=False, wide=None):
        self.probe = probe or (lambda *a: None)
        # exact_io: the embedding lookup and the head as the hardware does
        # them, a per-token requantize in and a per-column requantize to
        # one logit scale out, instead of float rescaling at the edges.
        self.exact_io = exact_io
        # wide: {tensor name: bits} for activations held wider than
        # act_bits, e.g. {"m": 32} for Qwen3's MLP product, whose
        # massive activations leave its ordinary values below one code.
        self.wide = wide or {}
        self.c = cfg
        self.A = act_bits
        self.pc = per_channel
        self.D = cfg["hidden_size"]
        self.F = cfg["intermediate_size"]
        self.H = cfg["num_attention_heads"]
        self.KV = cfg["num_key_value_heads"]
        self.hd = qr.head_dim(cfg)
        self.qkn = "model.layers.0.self_attn.q_norm.weight" in W
        self.NL = cfg["num_hidden_layers"]
        self.V = cfg["vocab_size"]
        self.ms = model_spec(cfg, act_bits)
        ms = self.ms
        self.sp = {"rmsnorm": specgen.derive_rmsnorm_spec(ms),
                   "attn": specgen.derive_attn_spec(ms),
                   "softmax": specgen.derive_softmax_spec(ms),
                   "rope": specgen.derive_rope_spec(ms),
                   "silu": specgen.derive_silu_spec(ms),
                   "resadd": specgen.derive_resadd_spec(ms),
                   "requant": specgen.derive_requant_spec(ms)}
        if self.qkn:
            # The same norm over a row of head_dim: only the row length and
            # its address change, so the rsqrt and requantizer inside are the
            # hidden-size norm's, shared by both instances in the RTL. Derived
            # again, not copied with two parameters changed: the copy kept
            # "sum over i in 0..63" and 6-bit addresses for a 32-element row,
            # and Sonnet summed 64 elements through the wrapping address.
            hn = specgen.derive_rmsnorm_spec(ms, row=self.hd)
            hn["top_module"] = "rmsnorm_hd"
            self.sp["headnorm"] = hn
        rq = self.sp["requant"]["parameters"]
        self.mw, self.sw = rq["scale_width"], rq["shift_width"]
        self.hi = (1 << (act_bits - 1)) - 1
        t0 = time.time()
        self.Q, self.WS = {}, {}
        D, F, H, KV, hd = self.D, self.F, self.H, self.KV, self.hd
        shapes = {"q_proj": (H * hd, D), "k_proj": (KV * hd, D),
                  "v_proj": (KV * hd, D), "o_proj": (D, H * hd),
                  "gate_proj": (F, D), "up_proj": (F, D),
                  "down_proj": (D, F)}
        for li in range(self.NL):
            for mname, (r, c) in shapes.items():
                sub = "self_attn" if mname[0] in "qkvo" else "mlp"
                key = "model.layers.%d.%s.%s.weight" % (li, sub, mname)
                self.Q[key], self.WS[key] = quantize_rows(W[key][0], r, c,
                                                          per_channel)
        emb = "model.embed_tokens.weight"
        self.Q[emb], self.WS[emb] = quantize_rows(W[emb][0], self.V, D,
                                                  per_channel)
        self.Wf = W
        log("quantized in %.0f s" % (time.time() - t0))
        self.cal = calib
        self._constants()
        self.reset()

    # -- scales ----------------------------------------------------------
    def _constants(self):
        cal, A = self.cal, self.A
        hi_of = lambda k: (1 << (self.wide.get(k[0] if isinstance(k, tuple) else k, A) - 1)) - 1
        s = {k: v / hi_of(k) for k, v in cal.items()}
        if "lg" in s:
            # Headroom: a logit above every calibrated one must not
            # saturate, or several would tie at the top code.
            s["lg"] *= 1.5
        self.s = s
        at = self.sp["attn"]["parameters"]
        sm = self.sp["softmax"]["parameters"]
        sl = self.sp["silu"]["parameters"]
        g = at["score_guard"]
        self.shs, self.gsh = [], []
        for li in range(self.NL):
            # Fold 1/sqrt(hd) and the score format into q's scale.
            shs = 0
            while shs + 1 < (1 << at["shift_s_width"]):
                if (math.sqrt(self.hd) * 2.0 ** (g - (shs + 1) - sm["score_frac"])
                        / s[("k", li)] < s[("q", li)]):
                    break
                shs += 1
            s[("q", li)] = (math.sqrt(self.hd)
                            * 2.0 ** (g - shs - sm["score_frac"]) / s[("k", li)])
            self.shs.append(shs)
            # The gate reaches SiLU's port as code << gs (or >> for wide
            # activations), so its scale is a power of two.
            want = s[("g", li)]
            e = math.ceil(math.log2(want * (1 << sl["frac"])))
            e = min(e, sl["width"] - 1 - (A - 1))
            s[("g", li)] = 2.0 ** e / (1 << sl["frac"])
            self.gsh.append(e)

    def reset(self):
        self.K = [[] for _ in range(self.NL)]
        self.Vc = [[] for _ in range(self.NL)]

    # -- blocks ----------------------------------------------------------
    def proj_consts(self, key, sx, dst_s, bias=None):
        """Each output column's (bias in accumulator units, scale, shift):
        what the per-column projection reads from its constant memory."""
        ws = self.WS[key]
        out, cache = [], {}
        for r, w in enumerate(ws):
            b = int(round(bias[r] / (sx * w))) if bias is not None else 0
            ratio = sx * w / dst_s
            k = cache.get(ratio)
            if k is None:
                k = cache[ratio] = pick(ratio, self.mw, self.sw)
            out.append((b, k[0], k[1]))
        return out

    def proj(self, x, key, sx, dst_s, bias=None):
        """requant(W x + b) at dst_s; per-channel weights give each output
        its own requantizer scale."""
        Qw = self.Q[key]
        cols = len(x)
        mul = operator.mul
        out = []
        for r, (b, sc, sh) in enumerate(self.proj_consts(key, sx, dst_s, bias)):
            acc = sum(map(mul, x, Qw[r * cols:(r + 1) * cols])) + b
            out.append(specgen.requant_golden(acc, sc, sh, self.A)[0])
        return out

    def norm_consts(self, gname, dst, block="rmsnorm"):
        """The gains as int8 codes, and the output (scale, shift). The
        row's 1/n is folded in, n the gains' length: the hidden size, or
        head_dim for Qwen3's norms on q and k."""
        pp = self.sp[block]["parameters"]
        gf = self.Wf[gname][0]
        gs = max(abs(v) for v in gf) / 127.0
        gq = [int(round(v / gs)) for v in gf]
        ow, kk = pp["rsqrt_out_width"], pp["norm_shift"]
        sc, sh = pick(2.0 ** (kk - ow) * math.sqrt(len(gf)) * gs / dst,
                      pp["scale_width"], pp["shift_width"])
        return gq, sc, sh

    def norm(self, x, gname, dst, block="rmsnorm"):
        p = self.sp[block]
        gq, sc, sh = self.norm_consts(gname, dst, block)
        return specgen.rmsnorm_golden(x, gq, 1, sc, sh, p["parameters"],
                                      p["derivation"]["rsqrt"])[4]

    def headnorm(self, v, heads, gname, dst):
        hd = self.hd
        return sum((self.norm(v[h * hd:(h + 1) * hd], gname, dst, "headnorm")
                    for h in range(heads)), [])

    def add_consts(self, sa, sb, dst):
        rp = self.sp["resadd"]["parameters"]
        best = None
        for sh in range(1, 60):
            ka, kb = int(round(sa / dst * (1 << sh))), int(round(sb / dst * (1 << sh)))
            if max(ka, kb) >= 1 << rp["scale_width"]:
                break
            best = (ka, kb, sh)
        return best

    def add(self, a, b, sa, sb, dst):
        ka, kb, sh = self.add_consts(sa, sb, dst)
        return [specgen.resadd_golden(u, v, ka, kb, sh, self.A)
                for u, v in zip(a, b)]

    def embed_consts(self, tok):
        """The per-token (scale, shift) that requantizes an int8
        embedding row, stored at its own per-channel scale, to x0's."""
        return pick(self.WS["model.embed_tokens.weight"][tok] / self.s["x0"],
                    self.mw, self.sw)

    def embed(self, tok):
        emb = "model.embed_tokens.weight"
        row = self.Q[emb][tok * self.D:(tok + 1) * self.D]
        if self.exact_io:
            sc, sh = self.embed_consts(tok)
            return [specgen.requant_golden(v, sc, sh, self.A)[0] for v in row]
        es = self.WS[emb][tok]
        return [max(-self.hi - 1, min(self.hi, int(round(v * es / self.s["x0"]))))
                for v in row]

    def rope(self, v, heads, pos):
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

    # -- one position ----------------------------------------------------
    def step(self, tok, pos, logits=True):
        s, A = self.s, self.A
        D, H, KV, hd = self.D, self.H, self.KV, self.hd
        grp = H // KV
        at, smp = self.sp["attn"]["parameters"], \
            self.sp["softmax"]["parameters"]
        sd = self.sp["silu"]["derivation"]
        emb = "model.embed_tokens.weight"
        # The embedding row, requantized from its int8 codes to x0's scale.
        x = self.embed(tok)
        sx = s["x0"]
        pr = self.probe
        pr("x0", x, sx, None)
        for li in range(self.NL):
            P = "model.layers.%d." % li
            xn = self.norm(x, P + "input_layernorm.weight", s[("xn", li)])
            sn = s[("xn", li)]
            b = lambda n: (self.Wf[P + "self_attn.%s.bias" % n][0]
                           if P + "self_attn.%s.bias" % n in self.Wf else None)
            if self.qkn:
                # Qwen3: q and k at their own calibrated scales, then each
                # head normalised to the scale the scores need.
                q = self.proj(xn, P + "self_attn.q_proj.weight", sn, s[("qp", li)])
                k = self.proj(xn, P + "self_attn.k_proj.weight", sn, s[("kp", li)])
                q = self.headnorm(q, H, P + "self_attn.q_norm.weight", s[("q", li)])
                k = self.headnorm(k, KV, P + "self_attn.k_norm.weight", s[("k", li)])
            else:
                q = self.proj(xn, P + "self_attn.q_proj.weight", sn, s[("q", li)], b("q_proj"))
                k = self.proj(xn, P + "self_attn.k_proj.weight", sn, s[("k", li)], b("k_proj"))
            v = self.proj(xn, P + "self_attn.v_proj.weight", sn, s[("v", li)], b("v_proj"))
            q = self.rope(q, H, pos)
            k = self.rope(k, KV, pos)
            pr("xn", xn, sn, li); pr("q", q, s[("q", li)], li)
            pr("k", k, s[("k", li)], li); pr("v", v, s[("v", li)], li)
            self.K[li].append(k)
            self.Vc[li].append(v)
            sc, sh = pick(2.0 ** -at["weight_frac"] * s[("v", li)] / s[("ctx", li)],
                          at["scale_width"], at["shift_width"])
            ctx = []
            for h in range(H):
                kv = h // grp
                Kh = [kk[kv * hd:(kv + 1) * hd] for kk in self.K[li]]
                Vh = [vv[kv * hd:(kv + 1) * hd] for vv in self.Vc[li]]
                ctx += specgen.attn_golden(q[h * hd:(h + 1) * hd], Kh, Vh,
                                           pos + 1, self.shs[li], sc, sh,
                                           at, smp)[4]
            a = self.proj(ctx, P + "self_attn.o_proj.weight", s[("ctx", li)],
                          s[("a", li)])
            pr("ctx", ctx, s[("ctx", li)], li); pr("a", a, s[("a", li)], li)
            x = self.add(x, a, sx, s[("a", li)], s[("x1", li)])
            sx = s[("x1", li)]
            pr("x1", x, sx, li)
            xn2 = self.norm(x, P + "post_attention_layernorm.weight",
                            s[("xn2", li)])
            g = self.proj(xn2, P + "mlp.gate_proj.weight", s[("xn2", li)],
                          s[("g", li)])
            u = self.proj(xn2, P + "mlp.up_proj.weight", s[("xn2", li)],
                          s[("u", li)])
            e = self.gsh[li]
            sg = [specgen.silu_golden(c << e if e >= 0 else c >> -e, sd)
                  for c in g]
            msc, msh = pick(2.0 ** -8 * s[("u", li)] / s[("m", li)],
                            self.mw, self.sw)
            m = [specgen.requant_golden(p_ * u_, msc, msh, self.wide.get("m", A))[0]
                 for p_, u_ in zip(sg, u)]
            dn = self.proj(m, P + "mlp.down_proj.weight", s[("m", li)],
                           s[("dn", li)])
            pr("xn2", xn2, s[("xn2", li)], li); pr("g", g, s[("g", li)], li)
            pr("u", u, s[("u", li)], li); pr("m", m, s[("m", li)], li)
            pr("dn", dn, s[("dn", li)], li)
            x = self.add(x, dn, sx, s[("dn", li)], s[("x2", li)])
            sx = s[("x2", li)]
            pr("x2", x, sx, li)
        if not logits:
            return None
        xf = self.norm(x, "model.norm.weight", s["xf"])
        self.last_xf = xf
        if self.exact_io:
            # Every logit requantized to one calibrated scale, as the
            # per-column projection does it; the argmax is over codes.
            return self.proj(xf, emb, s["xf"], s["lg"])
        Qe, ws = self.Q[emb], self.WS[emb]
        mul = operator.mul
        # The head only needs the argmax, so its accumulators are compared
        # in one unit, the per-row weight scale folded in.
        return [sum(map(mul, xf, Qe[r * D:(r + 1) * D])) * ws[r]
                for r in range(self.V)]


def calibrate(cfg, W, tok, text=CALIB, log=print):
    """Largest magnitude of every tensor, per layer, over a calibration
    text decoded in float: static scales, fixed before the model runs."""
    mx = {}

    def probe(name, vec, li):
        key = name if li is None else (name, li)
        m = max(abs(v) for v in vec)
        if m > mx.get(key, 0.0):
            mx[key] = m
    fm = qr.FloatQwen(cfg, W, probe)
    ids = tok.encode(text)
    t0 = time.time()
    for i, t in enumerate(ids):
        # Logits too, so the head's requantizer has a calibrated scale.
        fm.step(t, i, logits=True)
    log("calibrated on %d tokens in %.0f s" % (len(ids), time.time() - t0))
    return mx


def agreement(fm, im, ids, n, log=print):
    """Teacher-forced next-token agreement with the float model over the
    first n positions of ids."""
    fm.reset()
    im.reset()
    same = 0
    for i in range(n):
        lf = fm.step(ids[i], i)
        li = im.step(ids[i], i)
        a = max(range(len(lf)), key=lf.__getitem__)
        b = max(range(len(li)), key=li.__getitem__)
        same += (a == b)
        log("  pos %2d  float %6d  int %6d  %s" % (i, a, b, "=" if a == b else "x"))
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=4)
    a = ap.parse_args()
    cfg, W = qr.load()
    tok = qr.Tokenizer()
    cal = calibrate(cfg, W, tok)
    im = IntQwen(cfg, W, a.act_bits, a.per_channel, cal)
    fm = qr.FloatQwen(cfg, W)
    ids = tok.encode(EVAL)
    same = agreement(fm, im, ids, min(a.eval, len(ids) - 1))
    print("teacher-forced agreement: %d/%d (a%d, %s weights)"
          % (same, min(a.eval, len(ids) - 1), a.act_bits,
             "per-channel" if a.per_channel else "per-tensor"))
    out = qr.greedy(im, tok.encode(a.prompt), a.tokens)
    print("integer: %r" % tok.decode(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
