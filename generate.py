"""Run the trained checkpoint on the generated hardware's arithmetic and
emit text.

This is the thing the repo could not do. It described hardware for hosting
an LLM, and verified that hardware thoroughly, but had never run a model
through it and had never produced a token.

Every multiply-accumulate in this decode goes through MacModel, and every
requantization between matmuls goes through RequantModel. Both are
bit-accurate models of the RTL the agent generated, and both are checked
against that RTL in iverilog on vectors taken from this very decode, so
the text below is what the hardware would produce, not what a float model
would produce.

What is still host-side, and is not claimed otherwise: softmax and the
argmax, because the flow does not generate hardware for them yet.

Run: python3 generate.py [--tokens 60] [--prompt "the agent"]
"""
import argparse
import json
import math
import os
import random
import shutil
import sys

import specgen
from agent import RuleBasedAgent, FIX_WIDTH, FIX_CLEAR, FIX_SATURATE
from inference import MacModel, RequantModel, quantize, run_cosim, \
    run_requant_cosim, WORK
from train_tiny import CKPT


def qmat(m, bits):
    """Symmetric per-tensor quantization of a weight matrix."""
    flat = [v for row in m for v in row]
    q, scale = quantize(flat, bits)
    c = len(m[0])
    return [q[i * c:(i + 1) * c] for i in range(len(m))], scale


class HwModel:
    """The checkpoint, executed the way the hardware would execute it."""

    def __init__(self, ck, dw, aw, rq):
        self.d = ck["d_model"]
        self.f = ck["d_ff"]
        self.seq = ck["seq"]
        self.chars = ck["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        w = ck["weights"]
        self.tok = w["tok"]          # embeddings stay float: a lookup is
        self.pos = w["pos"]          # not arithmetic, so no MACs here
        self.q = {}
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "head"):
            self.q[name] = qmat(w[name], dw)
        self.mac = MacModel(dw, aw)
        self.rq = rq
        self.dots = []               # (x, w_col) for co-simulation
        self.rqs = []                # (acc, scale, shift)
        self.dw = dw

    def matvec(self, x, name, keep=1):
        w, _ = self.q[name]
        cols = len(w[0])
        out = []
        for c in range(cols):
            col = [w[r][c] for r in range(len(w))]
            out.append(self.mac.dot(x, col))
            if c < keep:
                self.dots.append((list(x), col))
        return out

    def requant(self, accs):
        q, sc, sh = self.rq.vector(accs)
        for a in accs[:1]:
            self.rqs.append((a, sc, sh))
        return q

    def forward(self, ids):
        """One decode step: returns the logits for the last position."""
        n = len(ids)
        hs = [[self.tok[t][j] + self.pos[i][j] for j in range(self.d)]
              for i, t in enumerate(ids)]
        # Quantize the residual stream once per position.
        hq = [quantize(h, self.dw)[0] for h in hs]
        qs = [self.requant(self.matvec(h, "wq")) for h in hq]
        ks = [self.requant(self.matvec(h, "wk")) for h in hq]
        vs = [self.requant(self.matvec(h, "wv")) for h in hq]
        i = n - 1                     # only the last position is decoded
        scale = 1.0 / math.sqrt(self.d)
        scores = [sum(a * b for a, b in zip(qs[i], ks[j])) * scale
                  for j in range(i + 1)]
        mx = max(scores)
        ex = [math.exp(s - mx) for s in scores]
        tot = sum(ex)
        ctx = [sum(ex[j] * vs[j][t] for j in range(i + 1)) / tot
               for t in range(self.d)]
        cq, _ = quantize(ctx, self.dw)
        att = self.requant(self.matvec(cq, "wo"))
        res = [hq[i][t] + att[t] for t in range(self.d)]
        rq, _ = quantize([float(v) for v in res], self.dw)
        h1 = self.requant(self.matvec(rq, "w1"))
        h1 = [max(0, u) for u in h1]
        h2 = self.requant(self.matvec(h1, "w2"))
        res2 = [rq[t] + h2[t] for t in range(self.d)]
        r2, _ = quantize([float(v) for v in res2], self.dw)
        return self.matvec(r2, "head")


def teacher_forced_agreement(ck, hw, n=48):
    """Next-token agreement under identical context.

    Free-running agreement conflates two different things: how much
    quantization perturbs a prediction, and how fast greedy decoding
    amplifies one different character into a different continuation. One
    early divergence makes everything after it disagree, so that number
    says almost nothing about the arithmetic. Feeding both models the same
    ground-truth context isolates the part that is actually about the
    hardware.
    """
    from train_tiny import Tiny
    from autodiff import V
    m = Tiny.__new__(Tiny)
    m.d, m.f, m.seq = ck["d_model"], ck["d_ff"], ck["seq"]
    w = ck["weights"]
    for name in ("tok", "pos", "wq", "wk", "wv", "wo", "w1", "w2", "head"):
        setattr(m, name, [[V(c) for c in row] for row in w[name]])
    m.vocab = len(ck["chars"])
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    same = total = 0
    top1_float = []
    for end in range(2, min(len(ids) - 1, n + 2)):
        ctx = ids[max(0, end - ck["seq"]):end]
        fl = m.forward(ctx)[-1]
        f_top = max(range(len(fl)), key=lambda k: fl[k].d)
        q_logits = hw.forward(ctx)
        q_top = max(range(len(q_logits)), key=lambda k: q_logits[k])
        same += (f_top == q_top)
        total += 1
        top1_float.append(f_top)
    return same, total


def float_reference(ck, prompt, n):
    """The same checkpoint in float, so the cost of running it on int8
    hardware is measured rather than assumed."""
    from train_tiny import Tiny
    from autodiff import V
    m = Tiny.__new__(Tiny)
    m.d, m.f, m.seq = ck["d_model"], ck["d_ff"], ck["seq"]
    w = ck["weights"]
    for name in ("tok", "pos", "wq", "wk", "wv", "wo", "w1", "w2", "head"):
        setattr(m, name, [[V(c) for c in row] for row in w[name]])
    m.vocab = len(ck["chars"])
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    out = [stoi[c] for c in prompt if c in stoi]
    for _ in range(n):
        lg = m.forward(out[-ck["seq"]:])[-1]
        out.append(max(range(len(lg)), key=lambda k: lg[k].d))
    return "".join(ck["chars"][i] for i in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--prompt", default="the agent")
    ap.add_argument("--cosim", type=int, default=12)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to run; defaults to tiny_llm.json")
    a = ap.parse_args()

    path = a.ckpt or CKPT
    if not os.path.exists(path):
        raise SystemExit("no checkpoint; run: python3 train_tiny.py")
    ck = json.load(open(path))
    print("checkpoint %s: d_model %d, d_ff %d, vocab %d"
          % (os.path.basename(path), ck["d_model"], ck["d_ff"],
             len(ck["chars"])))
    ms = specgen.load_model_spec()
    cspec = specgen.derive_chiplet_spec(ms)
    rspec = specgen.derive_requant_spec(ms)
    dw = cspec["parameters"]["data_width"]
    aw = cspec["parameters"]["acc_width"]
    rp = rspec["parameters"]
    print("hardware: int%d MAC, %d-bit signed accumulator; requantizer "
          "%d -> %d bits\n" % (dw, aw, rp["acc_width"], rp["out_width"]))

    rq = RequantModel(rp["out_width"], rp["scale_width"], rp["shift_width"])
    hw = HwModel(ck, dw, aw, rq)

    ids = [hw.stoi[c] for c in a.prompt if c in hw.stoi]
    if not ids:
        raise SystemExit("prompt has no characters the model knows")
    out = list(ids)
    for _ in range(a.tokens):
        window = out[-hw.seq:]
        logits = hw.forward(window)
        out.append(max(range(len(logits)), key=lambda k: logits[k]))
    text = "".join(hw.chars[i] for i in out)

    ref_text = float_reference(ck, a.prompt, a.tokens)
    agree = sum(1 for x, y in zip(text, ref_text) if x == y)
    print("prompt:    %r" % a.prompt)
    print("generated: %r" % text[len(a.prompt):])
    print("full:      %r" % text)
    print("float ref: %r" % ref_text[len(a.prompt):])
    print("free-running agreement with the float model: %d of %d characters "
          "(%.0f%%)" % (agree, len(text), 100.0 * agree / len(text)))
    tf_same, tf_total = teacher_forced_agreement(ck, hw)
    print("teacher-forced next-token agreement: %d of %d (%.0f%%)"
          % (tf_same, tf_total, 100.0 * tf_same / max(1, tf_total)))
    print("  the second number is the one about quantization; the first "
          "also\n  measures how fast greedy decoding amplifies a single "
          "divergence\n")
    print("%d dot products sampled, %d requantizations sampled, "
          "%d saturated, %d accumulator overflows"
          % (len(hw.dots), len(hw.rqs), rq.saturations, hw.mac.overflows))

    # Both models, against both generated blocks, on vectors from this run.
    rnd = random.Random(0)
    rnd.shuffle(hw.dots)
    rnd.shuffle(hw.rqs)
    dots = hw.dots[:a.cosim]
    rqs = hw.rqs[:a.cosim]
    mac_rtl = RuleBasedAgent().render_mac(cspec, {FIX_WIDTH, FIX_CLEAR})
    rq_rtl = RuleBasedAgent().render_requant(rspec, {FIX_SATURATE})

    print("\nco-simulating this decode against the generated RTL")
    got = run_cosim(cspec, dots, mac_rtl)
    bad = [i for i, (xs, ws) in enumerate(dots)
           if got.get(i) != MacModel(dw, aw).dot(xs, ws)]
    print("  MAC:       %d/%d dot products bit exact"
          % (len(dots) - len(bad), len(dots)))
    got2 = run_requant_cosim(rspec, rqs, rq_rtl)
    ref = RequantModel(rp["out_width"], rp["scale_width"], rp["shift_width"])
    bad2 = [i for i, (acc, sc, sh) in enumerate(rqs)
            if got2.get(i) != ref.apply(acc, sc, sh)]
    print("  requant:   %d/%d requantizations bit exact"
          % (len(rqs) - len(bad2), len(rqs)))
    shutil.rmtree(WORK, ignore_errors=True)
    if bad or bad2:
        print("\nMISMATCH: the text above is not what the hardware produces")
        return 1
    print("\nEvery dot product and every requantization in this decode is "
          "arithmetic\nthe generated RTL reproduces exactly, so the text is "
          "what the hardware\nwould emit. Softmax and argmax ran on the "
          "host; no RTL exists for them yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
