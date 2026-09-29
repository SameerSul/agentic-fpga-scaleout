"""A small random checkpoint with a real Qwen's structure, for tests.

The full-size sequencer takes hours to simulate on the real weights. This
builds a checkpoint in the same format qwen_real.py loads, a few layers
and a hidden size of 64, in either Qwen's style: "qwen3" (no biases, q
wider than the hidden state, RMSNorm on every head of q and k) or
"qwen2.5" (q, k and v biases, head_dim = hidden / heads). The integer
model is calibrated on its own float run, so qwen_full.build_model can
generate and simulate the same sequencer on it in seconds. The
vocabulary is more than twice the MLP width, so the head runs in several
chunks with a short last one, as the real vocabularies do.
"""
import array
import math
import random
import sys

import qwen_int as qi
import qwen_real as qr


def config(style="qwen3", nl=2, vocab=600, hidden=64, heads=4, kv=2):
    c = {"hidden_size": hidden, "intermediate_size": 256, "num_attention_heads": heads,
         "num_key_value_heads": kv, "num_hidden_layers": nl, "vocab_size": vocab,
         "rms_norm_eps": 1e-6, "rope_theta": 1000000.0}
    if style == "qwen3":
        c["head_dim"] = 32
    return c


def _fast_floats(rnd, n, std):
    """n float32s of random sign, magnitudes spread over the four octaves
    around std, built as bit patterns from random bytes: a checkpoint at a
    real model's size in seconds where gauss() per element takes minutes
    and a list of every value gigabytes."""
    raw = bytearray()
    left = 4 * n
    while left:                     # randbytes takes at most 2**31 bits
        raw += rnd.randbytes(min(left, 1 << 24))
        left -= min(left, 1 << 24)
    k = (127 + int(math.floor(math.log2(std)))) // 2
    # The top byte of a little-endian float32: the sign and the exponent's
    # high seven bits, k - 1 or k; the exponent's low bit is byte 2's
    # random top bit, so the exponent is one of 2k - 2 to 2k + 1.
    raw[3::4] = raw[3::4].translate(bytes((b & 0x80) | (k - 1 + (b & 1))
                                          for b in range(256)))
    a = array.array("f")
    a.frombytes(bytes(raw))
    if sys.byteorder == "big":
        a.byteswap()
    return a


def weights(cfg, style="qwen3", seed=5):
    rnd = random.Random(seed)
    D, F, H, KV = (cfg["hidden_size"], cfg["intermediate_size"],
                   cfg["num_attention_heads"], cfg["num_key_value_heads"])
    hd = qr.head_dim(cfg)
    W = {}

    def put(name, shape, std, mean=0.0):
        n = 1
        for d in shape:
            n *= d
        if n >= 1 << 20 and mean == 0.0:
            W[name] = (_fast_floats(rnd, n, std), list(shape))
            return
        W[name] = (array.array("f", [mean + rnd.gauss(0.0, std) for _ in range(n)]),
                   list(shape))
    put("model.embed_tokens.weight", (cfg["vocab_size"], D), 1.0)
    put("model.norm.weight", (D,), 0.1, 1.0)
    for li in range(cfg["num_hidden_layers"]):
        P = "model.layers.%d." % li
        put(P + "input_layernorm.weight", (D,), 0.1, 1.0)
        put(P + "post_attention_layernorm.weight", (D,), 0.1, 1.0)
        for m, (r, c) in (("q_proj", (H * hd, D)), ("k_proj", (KV * hd, D)),
                          ("v_proj", (KV * hd, D)), ("o_proj", (D, H * hd))):
            put(P + "self_attn.%s.weight" % m, (r, c), c ** -0.5)
            if style == "qwen2.5" and m != "o_proj":
                put(P + "self_attn.%s.bias" % m, (r,), 0.5)
        if style == "qwen3":
            put(P + "self_attn.q_norm.weight", (hd,), 0.3, 1.5)
            put(P + "self_attn.k_norm.weight", (hd,), 0.3, 1.5)
        for m, (r, c) in (("gate_proj", (F, D)), ("up_proj", (F, D)),
                          ("down_proj", (D, F))):
            put(P + "mlp.%s.weight" % m, (r, c), c ** -0.5)
    return W


class _Ids:
    """calibrate() takes a tokenizer; this one hands back fixed ids."""

    def __init__(self, ids):
        self.ids = ids

    def encode(self, text):
        return self.ids


def model(style="qwen3", seed=5, nl=2, lanes=None, vocab=600, hidden=64, heads=4, kv=2):
    """(integer model, float model) for a synthetic checkpoint. lanes
    sets the projection's width instead of the board's rule."""
    cfg = config(style, nl, vocab, hidden, heads, kv)
    W = weights(cfg, style, seed)
    rnd = random.Random(seed + 1)
    calib = [rnd.randrange(cfg["vocab_size"]) for _ in range(12)]
    cal = qi.calibrate(cfg, W, _Ids(calib), log=lambda *a: None)
    im = qi.IntQwen(cfg, W, 16, True, cal, log=lambda *a: None, exact_io=True)
    if lanes:
        im.ms["lanes"] = lanes
    return im, qr.FloatQwen(cfg, W)
