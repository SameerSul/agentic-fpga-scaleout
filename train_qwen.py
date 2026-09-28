"""Train a small transformer with Qwen's structure, pure stdlib.

tiny_llm.json is a real trained decoder, but not a Qwen-shaped one: it has
learned position vectors, one head, a ReLU MLP and one layer. Qwen rotates
q and k by position instead (RoPE), shares each key/value head between
several query heads (grouped-query attention), gates its MLP through SiLU
(SwiGLU), stacks layers, and normalises once more before the head. This
trains that structure at a size a testbench can run, so the generated
decoder can be checked on Qwen's arithmetic and not only on a simpler
model's.

Run: python3 train_qwen.py [--steps 600]
"""
import argparse
import json
import math
import os
import random
import sys

from autodiff import V, dot
from train_tiny import CORPUS, softmax

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "tiny_qwen.json")

GEOM = dict(d_model=32, n_head=2, n_kv_head=1, head_dim=16, d_ff=64,
            n_layer=2, seq=24, rope_theta=1000000.0)


def rope_angles(pos, hd, base):
    """Qwen's rotate-half pairing: pair i is (x[i], x[i + hd/2]), turned by
    pos * base**(-2i/hd)."""
    return [pos * base ** (-2.0 * i / hd) for i in range(hd // 2)]


def rope(v, pos, hd, base):
    """Rotate one head's vector. Works on floats and on autodiff nodes."""
    h = hd // 2
    out = list(v)
    for i, a in enumerate(rope_angles(pos, hd, base)):
        c, s = math.cos(a), math.sin(a)
        x1, x2 = v[i], v[i + h]
        out[i] = x1 * c - x2 * s
        out[i + h] = x2 * c + x1 * s
    return out


class TinyQwen:
    def __init__(self, vocab, seed=0, **geom):
        g = dict(GEOM, **geom)
        self.__dict__.update(g)
        self.vocab = vocab
        rnd = random.Random(seed)

        def mat(r, c, gain=None):
            gain = gain or (1.0 / math.sqrt(r))
            return [[V(rnd.gauss(0, gain)) for _ in range(c)]
                    for _ in range(r)]
        D, F, hd = self.d_model, self.d_ff, self.head_dim
        self.tok = mat(vocab, D, 0.3)
        self.layers = []
        for _ in range(self.n_layer):
            self.layers.append({
                "g1": [[V(1.0) for _ in range(D)]],
                "wq": mat(D, self.n_head * hd),
                "wk": mat(D, self.n_kv_head * hd),
                "wv": mat(D, self.n_kv_head * hd),
                "wo": mat(self.n_head * hd, D),
                "g2": [[V(1.0) for _ in range(D)]],
                "wg": mat(D, F), "wu": mat(D, F), "wd": mat(F, D),
            })
        self.gf = [[V(1.0) for _ in range(D)]]
        self.head = mat(D, vocab)

    def tensors(self):
        yield "tok", self.tok
        for li, L in enumerate(self.layers):
            for k, m in L.items():
                yield "l%d_%s" % (li, k), m
        yield "gf", self.gf
        yield "head", self.head

    def params(self):
        return [p for _, m in self.tensors() for row in m for p in row]

    @staticmethod
    def mv(x, w):
        cols = [[w[r][c] for r in range(len(w))] for c in range(len(w[0]))]
        return [dot(x, col) for col in cols]

    @staticmethod
    def rms(v, g):
        ss = dot(v, v)
        inv = (ss * (1.0 / len(v)) + 1e-6) ** -0.5
        return [v[i] * inv * g[0][i] for i in range(len(v))]

    def forward(self, ids):
        n, hd = len(ids), self.head_dim
        grp = self.n_head // self.n_kv_head
        x = [list(self.tok[t]) for t in ids]
        for L in self.layers:
            hn = [self.rms(v, L["g1"]) for v in x]
            q = [self.mv(v, L["wq"]) for v in hn]
            k = [self.mv(v, L["wk"]) for v in hn]
            val = [self.mv(v, L["wv"]) for v in hn]
            for i in range(n):
                q[i] = sum((rope(q[i][h * hd:(h + 1) * hd], i, hd,
                                 self.rope_theta)
                            for h in range(self.n_head)), [])
                k[i] = sum((rope(k[i][h * hd:(h + 1) * hd], i, hd,
                                 self.rope_theta)
                            for h in range(self.n_kv_head)), [])
            sc = 1.0 / math.sqrt(hd)
            for i in range(n):
                ctx = []
                for h in range(self.n_head):
                    kv = h // grp
                    qh = q[i][h * hd:(h + 1) * hd]
                    s = [dot(qh, k[j][kv * hd:(kv + 1) * hd]) * sc
                         for j in range(i + 1)]
                    w = softmax(s)
                    for d in range(hd):
                        ctx.append(dot(w, [val[j][kv * hd + d]
                                           for j in range(i + 1)]))
                a = self.mv(ctx, L["wo"])
                x[i] = [x[i][j] + a[j] for j in range(self.d_model)]
            for i in range(n):
                hn2 = self.rms(x[i], L["g2"])
                g = self.mv(hn2, L["wg"])
                u = self.mv(hn2, L["wu"])
                m = [gg * ((1.0 + (-gg).exp()) ** -1) * uu
                     for gg, uu in zip(g, u)]
                dn = self.mv(m, L["wd"])
                x[i] = [x[i][j] + dn[j] for j in range(self.d_model)]
        return [self.mv(self.rms(v, self.gf), self.head) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--out", default=CKPT)
    a = ap.parse_args()
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    ids_all = [stoi[c] for c in CORPUS]
    model = TinyQwen(len(chars))
    ps = model.params()
    m = [0.0] * len(ps)
    v = [0.0] * len(ps)
    S = model.seq
    print("vocab %d, %s, %d parameters" % (len(chars), GEOM, len(ps)))
    rnd = random.Random(1)
    for step in range(1, a.steps + 1):
        start = rnd.randrange(0, len(ids_all) - S - 1)
        ids = ids_all[start:start + S]
        tgt = ids_all[start + 1:start + S + 1]
        logits = model.forward(ids)
        loss = V(0.0)
        for i, lg in enumerate(logits):
            loss = loss - softmax(lg)[tgt[i]].log()
        loss = loss * (1.0 / len(logits))
        for p in ps:
            p.g = 0.0
        loss.backward()
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / a.steps))
        for i, p in enumerate(ps):
            g = max(-5.0, min(5.0, p.g))
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            p.d -= lr * (m[i] / (1 - b1 ** step)) / (
                math.sqrt(v[i] / (1 - b2 ** step)) + eps)
        if step % 25 == 0 or step == 1:
            print("  step %4d  loss %.4f  lr %.4f" % (step, loss.d, lr))
            sys.stdout.flush()
    ck = dict(GEOM, arch="qwen", corpus=CORPUS, chars=chars,
              weights={name: [[c.d for c in row] for row in mt]
                       for name, mt in model.tensors()})
    with open(a.out, "w") as f:
        json.dump(ck, f)
    print("wrote %s (%.0f KB)" % (os.path.basename(a.out),
                                  os.path.getsize(a.out) / 1024.0))


if __name__ == "__main__":
    main()
