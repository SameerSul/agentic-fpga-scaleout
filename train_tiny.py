"""Train a small character-level transformer, pure stdlib, and write a
checkpoint.

The repo could describe hardware for an LLM but had never run one, because
there was no checkpoint and no tokenizer. Downloading Qwen3-0.6B is not an
option here (no numeric stack, no network dependency wanted in a capstone
repo), so this trains a real, if small, transformer from scratch and
commits the weights. It is a genuine decoder: learned token and position
embeddings, causal single-head attention with softmax, an MLP with a
residual around each, and a cross-entropy objective over next characters.

The point is not the model's quality. It is that inference.py can then run
a real trained network, with real learned weights, through arithmetic that
has been checked bit for bit against the generated RTL, and emit text.

Run: python3 train_tiny.py [--steps 400]
"""
import argparse
import json
import math
import os
import random
import sys

from autodiff import V

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "tiny_llm.json")

CORPUS = ("the agent writes the rtl and the tools decide. "
          "the agent writes the rtl and the tools decide. ")

D_MODEL, D_FF, SEQ = 16, 32, 24


def softmax(xs):
    m = max(x.d for x in xs)
    ex = [(x - V(m)).exp() for x in xs]
    s = ex[0]
    for e in ex[1:]:
        s = s + e
    return [e / s for e in ex]


class Tiny:
    """Geometry is per instance, not module level, so a checkpoint of any
    size can be rebuilt from its own fields rather than from whatever the
    module happens to be set to."""

    def __init__(self, vocab, seed=0, d_model=None, d_ff=None, seq=None):
        rnd = random.Random(seed)
        self.vocab = vocab
        self.d = d = d_model or D_MODEL
        self.f = f = d_ff or D_FF
        self.seq = seq or SEQ

        def mat(r, c, g=None):
            g = g or (1.0 / math.sqrt(r))
            return [[V(rnd.gauss(0, g)) for _ in range(c)] for _ in range(r)]
        self.tok = mat(vocab, d, 0.3)
        self.pos = mat(self.seq, d, 0.3)
        self.wq, self.wk, self.wv, self.wo = (mat(d, d) for _ in range(4))
        self.w1, self.w2 = mat(d, f), mat(f, d)
        self.head = mat(d, vocab)
        # RMSNorm gains, one per position in the residual stream, before
        # attention and before the MLP. This is what the target model
        # family normalises with, and it is what exercises the generated
        # inverse square root.
        self.g1 = [[V(1.0) for _ in range(d)]]
        self.g2 = [[V(1.0) for _ in range(d)]]

    def params(self):
        out = []
        for m in (self.tok, self.pos, self.wq, self.wk, self.wv, self.wo,
                  self.w1, self.w2, self.head, self.g1, self.g2):
            for row in m:
                out.extend(row)
        return out

    def rmsnorm(self, v, g):
        """x / sqrt(mean(x^2) + eps) * g, the normalisation the target
        models use. No mean subtraction, which is what makes it cheaper
        than LayerNorm and why only the inverse square root needs
        hardware."""
        n = len(v)
        ss = v[0] * v[0]
        for u in v[1:]:
            ss = ss + u * u
        inv = (ss * (1.0 / n) + V(1e-6)) ** -0.5
        return [v[i] * inv * g[0][i] for i in range(n)]

    @staticmethod
    def mv(x, w):
        cols = len(w[0])
        out = []
        for c in range(cols):
            s = x[0] * w[0][c]
            for r in range(1, len(w)):
                s = s + x[r] * w[r][c]
            out.append(s)
        return out

    def forward(self, ids):
        n, D = len(ids), self.d
        h = [[self.tok[t][j] + self.pos[i][j] for j in range(D)]
             for i, t in enumerate(ids)]
        hn = [self.rmsnorm(v, self.g1) for v in h]
        q = [self.mv(v, self.wq) for v in hn]
        k = [self.mv(v, self.wk) for v in hn]
        val = [self.mv(v, self.wv) for v in hn]
        scale = 1.0 / math.sqrt(D)
        ctx = []
        for i in range(n):
            # Causal: position i attends to 0..i only.
            scores = []
            for j in range(i + 1):
                s = q[i][0] * k[j][0]
                for t in range(1, D):
                    s = s + q[i][t] * k[j][t]
                scores.append(s * scale)
            w = softmax(scores)
            acc = [V(0.0)] * D
            for j in range(i + 1):
                acc = [acc[t] + w[j] * val[j][t] for t in range(D)]
            ctx.append(acc)
        a = [self.mv(c, self.wo) for c in ctx]
        h = [[h[i][j] + a[i][j] for j in range(D)] for i in range(n)]
        for i in range(n):
            hn2 = self.rmsnorm(h[i], self.g2)
            m1 = [u.relu() for u in self.mv(hn2, self.w1)]
            m2 = self.mv(m1, self.w2)
            h[i] = [h[i][j] + m2[j] for j in range(D)]
        return [self.mv(h[i], self.head) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--d-ff", type=int, default=D_FF)
    ap.add_argument("--out", default=CKPT)
    a = ap.parse_args()

    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    ids_all = [stoi[c] for c in CORPUS]
    model = Tiny(len(chars), d_model=a.d_model, d_ff=a.d_ff)
    ps = model.params()
    m = [0.0] * len(ps)
    v = [0.0] * len(ps)
    print("vocab %d, d_model %d, d_ff %d, seq %d, %d parameters"
          % (len(chars), a.d_model, a.d_ff, SEQ, len(ps)))

    rnd = random.Random(1)
    for step in range(1, a.steps + 1):
        start = rnd.randrange(0, len(ids_all) - SEQ - 1)
        ids = ids_all[start:start + SEQ]
        tgt = ids_all[start + 1:start + SEQ + 1]
        logits = model.forward(ids)
        loss = V(0.0)
        for i, lg in enumerate(logits):
            p = softmax(lg)
            loss = loss - p[tgt[i]].log()
        loss = loss * (1.0 / len(logits))
        for p in ps:
            p.g = 0.0
        loss.backward()
        # Adam, which reaches a usable model in a few hundred steps where
        # plain SGD at this size does not.
        b1, b2, eps = 0.9, 0.999, 1e-8
        # Cosine decay. At a fixed rate the loss bounced between 1.0 and
        # 1.5 late in training instead of settling.
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps))
        for i, p in enumerate(ps):
            g = max(-5.0, min(5.0, p.g))
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** step)
            vh = v[i] / (1 - b2 ** step)
            p.d -= lr * mh / (math.sqrt(vh) + eps)
        if step % 25 == 0 or step == 1:
            print("  step %4d  loss %.4f  lr %.4f" % (step, loss.d, lr))
            sys.stdout.flush()

    ck = {
        "corpus": CORPUS, "chars": chars,
        "d_model": a.d_model, "d_ff": a.d_ff, "seq": SEQ,
        "weights": {
            name: [[c.d for c in row] for row in getattr(model, name)]
            for name in ("tok", "pos", "wq", "wk", "wv", "wo", "w1", "w2",
                         "head", "g1", "g2")
        },
    }
    with open(a.out, "w") as f:
        json.dump(ck, f)
    print("wrote %s (%.0f KB)" % (os.path.basename(a.out),
                                  os.path.getsize(a.out) / 1024.0))


if __name__ == "__main__":
    main()
