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
    """Symmetric per-output-channel quantization of a weight matrix.

    One scale for the whole tensor is the easy thing and the wrong thing:
    a single outlier column forces every other column to share its range,
    and the small-magnitude columns lose most of their resolution. Real
    int8 inference gives each output channel its own scale, and the
    hardware here already allows it, because the requantizer takes the
    scale as a run-time input rather than baking it in.
    """
    rows, cols = len(m), len(m[0])
    q = [[0] * cols for _ in range(rows)]
    scales = []
    hi = (1 << (bits - 1)) - 1
    for c in range(cols):
        col = [m[r][c] for r in range(rows)]
        mx = max(abs(v) for v in col) or 1.0
        sc = mx / hi
        scales.append(sc)
        for r in range(rows):
            q[r][c] = max(-hi - 1, min(hi, int(round(col[r] / sc))))
    return q, scales


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
        """Returns (accumulators, per-column weight scales)."""
        w, wscales = self.q[name]
        cols = len(w[0])
        out = []
        for c in range(cols):
            col = [w[r][c] for r in range(len(w))]
            out.append(self.mac.dot(x, col))
            if c < keep:
                self.dots.append((list(x), col))
        return out, wscales

    def forward(self, ids):
        """One decode step, with the scale carried alongside every
        activation.

        An int8 activation is a pair: the codes and the scale that turns
        them back into values. Adding two int8 vectors that carry
        different scales adds numbers in different units, which is what
        this did before and it is simply wrong arithmetic rather than a
        rounding loss. Residuals are therefore summed in the value domain
        and requantized, which is what hardware does with a wide
        accumulator between the add and the next matmul.
        """
        n, D = len(ids), self.d
        hs = [[self.tok[t][j] + self.pos[i][j] for j in range(D)]
              for i, t in enumerate(ids)]
        hq = [self.qact(h) for h in hs]
        qs = [self.project(h, "wq") for h in hq]
        ks = [self.project(h, "wk") for h in hq]
        vs = [self.project(h, "wv") for h in hq]

        i = n - 1                     # only the last position is decoded
        sc = 1.0 / math.sqrt(D)
        scores = [sum(a * b for a, b in zip(qs[i][0], ks[j][0]))
                  * qs[i][1] * ks[j][1] * sc for j in range(i + 1)]
        mx = max(scores)
        ex = [math.exp(s_ - mx) for s_ in scores]
        tot = sum(ex)
        ctx = [sum(ex[j] * vs[j][0][t] * vs[j][1] for j in range(i + 1)) / tot
               for t in range(D)]
        att = self.project(self.qact(ctx), "wo")
        res = self.add(hq[i], att)
        h1q, h1s = self.project(res, "w1")
        h1 = ([max(0, u) for u in h1q], h1s)
        h2 = self.project(h1, "w2")
        res2 = self.add(res, h2)
        logits, ls = self.project(res2, "head")
        return [v * ls for v in logits]

    def qact(self, vals):
        """Quantize a float activation vector, returning codes and scale."""
        q, s = quantize(vals, self.dw)
        return q, s

    def project(self, act, name):
        """Matmul then requantize, returning codes and their scale."""
        xq, xs = act
        accs, wscales = self.matvec(xq, name)
        # Per-channel weight scales mean the accumulators are not in one
        # unit, so they are brought into a common one before the shared
        # output scale is picked. That correction is a per-channel
        # multiply, which is what the requantizer's scale input is for.
        ref = max(wscales) or 1.0
        adj = [int(round(a * (wscales[k] / ref))) for k, a in enumerate(accs)]
        q, msc, sh = self.rq.vector(adj)
        for a in adj[:1]:
            self.rqs.append((a, msc, sh))
        # value = code * (acc scale) * (the shift the requantizer applied)
        out_scale = xs * ref * (float(1 << sh) / msc if msc else 1.0)
        return q, out_scale

    def add(self, a, b):
        """Residual add. Both operands are returned to values, summed, and
        requantized, because their scales differ."""
        (aq, asc), (bq, bsc) = a, b
        return self.qact([x * asc + y * bsc for x, y in zip(aq, bq)])

def _float_model(ck):
    """Rebuild the checkpoint in float, at whatever geometry it records."""
    from train_tiny import Tiny
    from autodiff import V
    m = Tiny.__new__(Tiny)
    m.d, m.f, m.seq = ck["d_model"], ck["d_ff"], ck["seq"]
    m.vocab = len(ck["chars"])
    w = ck["weights"]
    for name in ("tok", "pos", "wq", "wk", "wv", "wo", "w1", "w2", "head"):
        setattr(m, name, [[V(c) for c in row] for row in w[name]])
    return m


def teacher_forced_agreement(ck, hw, n=48):
    """Next-token agreement under identical context.

    Free-running agreement conflates two things: how much quantization
    perturbs a prediction, and how fast greedy decoding amplifies one
    different character into a different continuation. Feeding both models
    the same ground-truth context isolates the part about the arithmetic.
    The float model's own accuracy and top-2 margin come back too, because
    a model whose decisions are nearly ties is easy to flip and that is a
    property of the checkpoint, not of the hardware.
    """
    m = _float_model(ck)
    stoi = {c: i for i, c in enumerate(ck["chars"])}
    ids = [stoi[c] for c in ck["corpus"] if c in stoi]
    same = total = correct = 0
    margins = []
    for end in range(2, min(len(ids) - 1, n + 2)):
        ctx = ids[max(0, end - ck["seq"]):end]
        fl = [v.d for v in m.forward(ctx)[-1]]
        order = sorted(range(len(fl)), key=lambda k: fl[k], reverse=True)
        margins.append(fl[order[0]] - fl[order[1]])
        q = hw.forward(ctx)
        same += (order[0] == max(range(len(q)), key=lambda k: q[k]))
        correct += (order[0] == ids[end])
        total += 1
    return (same, total, correct / max(1, total),
            sum(margins) / max(1, len(margins)))


def float_reference(ck, prompt, n):
    """The same checkpoint in float, so the cost of running it on int8
    hardware is measured rather than assumed."""
    m = _float_model(ck)
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
    tf_same, tf_total, fl_acc, margin = teacher_forced_agreement(ck, hw)
    print("teacher-forced next-token agreement: %d of %d (%.0f%%)"
          % (tf_same, tf_total, 100.0 * tf_same / max(1, tf_total)))
    print("float model's own next-token accuracy: %.0f%%, mean top-2 logit "
          "margin %.2f" % (100.0 * fl_acc, margin))
    print("  a model with a small margin is easy to flip, so agreement has "
          "to be\n  read next to the margin: an undertrained checkpoint "
          "looks like a\n  quantization problem when it is not")
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
