"""The real Qwen2.5-0.5B or Qwen3-0.6B, loaded and run in pure Python.

fetch_qwen.py downloads the weights; nothing here needs a numeric stack.
This file has three parts: a safetensors reader, Qwen's byte-level BPE
tokenizer, and the model's forward pass in float, decoding one token at a
time with a KV cache. The float pass is the reference the integer model
built from the generated blocks is measured against.

FPGAI_QWEN picks the checkpoint for this and every script built on it:
qwen2.5 (the default) or qwen3, the model Architect Labs hosted. Qwen3
has no q/k/v biases, a head dimension of its own (128, so q is wider than
the hidden state), and an RMSNorm over each head of q and k before RoPE.

Run: python3 qwen_real.py [--prompt "The capital of France is"] [--tokens 8]
     FPGAI_QWEN=qwen3 python3 qwen_real.py
"""
import argparse
import array
import json
import math
import operator
import os
import re
import struct
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("FPGAI_QWEN", "qwen2.5")
WDIR = {"qwen2.5": os.path.join(ROOT, "qwen_weights"),
        "qwen3": os.path.join(ROOT, "qwen_weights", "qwen3-0.6b")}[MODEL]


def head_dim(cfg):
    return cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]


# ---------------------------------------------------------------- weights

class SafeTensors:
    """Reads tensors lazily out of one .safetensors file."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def raw(self, name):
        h = self.header[name]
        a, b = h["data_offsets"]
        self.f.seek(self.base + a)
        return h, self.f.read(b - a)

    def f32(self, name):
        """A bf16 tensor as float32. bf16 is the top half of a float32, so
        the conversion is interleaving two zero bytes under each value,
        done with slice assignment rather than a Python loop."""
        h, data = self.raw(name)
        assert h["dtype"] == "BF16", h["dtype"]
        out = bytearray(len(data) * 2)
        out[2::4] = data[0::2]
        out[3::4] = data[1::2]
        a = array.array("f")
        a.frombytes(bytes(out))
        if sys.byteorder != "little":
            a.byteswap()
        return a, h["shape"]


def load(wdir=WDIR):
    cfg = json.load(open(os.path.join(wdir, "config.json")))
    st = SafeTensors(os.path.join(wdir, "model.safetensors"))
    W = {}
    for name in st.header:
        if name != "__metadata__":
            W[name] = st.f32(name)
    return cfg, W


# ---------------------------------------------------------------- tokenizer

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# Qwen2's pre-tokenizer, with \p{L} as [^\W\d_] and \p{N} as \d: Python's
# re has no Unicode property classes, and on English text these agree.
_PRE = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d"
                  r"| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


class Tokenizer:
    def __init__(self, wdir=WDIR):
        t = json.load(open(os.path.join(wdir, "tokenizer.json")))
        self.vocab = t["model"]["vocab"]
        for a in t.get("added_tokens", []):
            self.vocab[a["content"]] = a["id"]
        self.inv = {v: k for k, v in self.vocab.items()}
        merges = t["model"]["merges"]
        self.rank = {}
        for i, m in enumerate(merges):
            a, b = m.split(" ", 1) if isinstance(m, str) else m
            self.rank[(a, b)] = i
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.cache = {}

    def _bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, bi = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]
        self.cache[word] = parts
        return parts

    def encode(self, text):
        ids = []
        for piece in _PRE.findall(text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.vocab[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        s = "".join(self.inv[i] for i in ids)
        return bytes(self.u2b[c] for c in s).decode("utf-8", "replace")


# ---------------------------------------------------------------- float

def matvec(Wt, x, rows, cols, bias=None):
    """y = W x for W stored row-major [rows][cols], as the checkpoint
    stores projections: each output's weights are contiguous."""
    mul = operator.mul
    y = [sum(map(mul, x, Wt[r * cols:(r + 1) * cols])) for r in range(rows)]
    if bias is not None:
        y = [a + b for a, b in zip(y, bias)]
    return y


def rms(x, g, eps):
    inv = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * inv * gg for v, gg in zip(x, g)]


def rope(v, pos, hd, base):
    h = hd // 2
    out = list(v)
    for i in range(h):
        a = pos * base ** (-2.0 * i / hd)
        c, s = math.cos(a), math.sin(a)
        out[i] = v[i] * c - v[i + h] * s
        out[i + h] = v[i + h] * c + v[i] * s
    return out


class FloatQwen:
    """The checkpoint in float, one position at a time, with a KV cache.
    probe(name, vector, layer) sees every intermediate, for calibration."""

    def __init__(self, cfg, W, probe=None):
        self.c = cfg
        self.W = W
        self.D = cfg["hidden_size"]
        self.F = cfg["intermediate_size"]
        self.H = cfg["num_attention_heads"]
        self.KV = cfg["num_key_value_heads"]
        self.hd = head_dim(cfg)
        self.NL = cfg["num_hidden_layers"]
        self.V = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.base = cfg["rope_theta"]
        self.probe = probe or (lambda *a: None)
        self.reset()

    def reset(self):
        self.K = [[] for _ in range(self.NL)]
        self.Vc = [[] for _ in range(self.NL)]

    def w(self, name):
        return self.W[name][0]

    def b(self, name):
        """A bias, or None where the checkpoint has none (Qwen3)."""
        return self.W[name][0] if name in self.W else None

    def headnorm(self, v, heads, name):
        """Qwen3's RMSNorm over each head of q or k; Qwen2.5 has none."""
        if name not in self.W:
            return v
        hd, g = self.hd, self.w(name)
        return sum((rms(v[i * hd:(i + 1) * hd], g, self.eps)
                    for i in range(heads)), [])

    def step(self, tok, pos, logits=True):
        D, F, H, KV, hd = self.D, self.F, self.H, self.KV, self.hd
        grp = H // KV
        emb = self.w("model.embed_tokens.weight")
        x = list(emb[tok * D:(tok + 1) * D])
        self.probe("x0", x, None)
        for li in range(self.NL):
            P = "model.layers.%d." % li
            h = rms(x, self.w(P + "input_layernorm.weight"), self.eps)
            self.probe("xn", h, li)
            q = matvec(self.w(P + "self_attn.q_proj.weight"), h, H * hd, D,
                       self.b(P + "self_attn.q_proj.bias"))
            k = matvec(self.w(P + "self_attn.k_proj.weight"), h, KV * hd, D,
                       self.b(P + "self_attn.k_proj.bias"))
            v = matvec(self.w(P + "self_attn.v_proj.weight"), h, KV * hd, D,
                       self.b(P + "self_attn.v_proj.bias"))
            self.probe("qp", q, li)
            self.probe("kp", k, li)
            q = self.headnorm(q, H, P + "self_attn.q_norm.weight")
            k = self.headnorm(k, KV, P + "self_attn.k_norm.weight")
            q = sum((rope(q[i * hd:(i + 1) * hd], pos, hd, self.base)
                     for i in range(H)), [])
            k = sum((rope(k[i * hd:(i + 1) * hd], pos, hd, self.base)
                     for i in range(KV)), [])
            self.probe("q", q, li)
            self.probe("k", k, li)
            self.probe("v", v, li)
            self.K[li].append(k)
            self.Vc[li].append(v)
            ctx = []
            sc = 1.0 / math.sqrt(hd)
            for hh in range(H):
                kv = hh // grp
                qh = q[hh * hd:(hh + 1) * hd]
                s = [sum(map(operator.mul, qh, kk[kv * hd:(kv + 1) * hd]))
                     * sc for kk in self.K[li]]
                self.probe("score", s, li)
                m = max(s)
                e = [math.exp(z - m) for z in s]
                z = sum(e)
                for d in range(hd):
                    ctx.append(sum(e[j] * self.Vc[li][j][kv * hd + d]
                                   for j in range(len(e))) / z)
            self.probe("ctx", ctx, li)
            a = matvec(self.w(P + "self_attn.o_proj.weight"), ctx, D, H * hd)
            self.probe("a", a, li)
            x = [p + r for p, r in zip(x, a)]
            self.probe("x1", x, li)
            h2 = rms(x, self.w(P + "post_attention_layernorm.weight"), self.eps)
            self.probe("xn2", h2, li)
            g = matvec(self.w(P + "mlp.gate_proj.weight"), h2, F, D)
            u = matvec(self.w(P + "mlp.up_proj.weight"), h2, F, D)
            self.probe("g", g, li)
            self.probe("u", u, li)
            mm = [gg / (1.0 + math.exp(-gg)) * uu for gg, uu in zip(g, u)]
            self.probe("m", mm, li)
            dn = matvec(self.w(P + "mlp.down_proj.weight"), mm, D, F)
            self.probe("dn", dn, li)
            x = [p + r for p, r in zip(x, dn)]
            self.probe("x2", x, li)
        xf = rms(x, self.w("model.norm.weight"), self.eps)
        self.probe("xf", xf, None)
        if not logits:
            return None
        lg = matvec(emb, xf, self.V, D)
        self.probe("lg", lg, None)
        return lg


def greedy(model, ids, n):
    model.reset()
    out = list(ids)
    lg = None
    for i, t in enumerate(ids):
        lg = model.step(t, i, logits=(i == len(ids) - 1))
    for _ in range(n):
        nxt = max(range(len(lg)), key=lg.__getitem__)
        out.append(nxt)
        lg = model.step(nxt, len(out) - 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    cfg, W = load()
    tok = Tokenizer()
    print("loaded in %.0f s" % (time.time() - t0))
    ids = tok.encode(a.prompt)
    print("prompt ids", ids)
    m = FloatQwen(cfg, W)
    t0 = time.time()
    out = greedy(m, ids, a.tokens)
    print("float  : %r  (%.1f s/token)"
          % (tok.decode(out), (time.time() - t0) / (len(ids) + a.tokens)))


if __name__ == "__main__":
    main()
