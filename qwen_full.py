"""A real Qwen2.5-0.5B or Qwen3-0.6B token, decoded by generated RTL.

Every earlier RTL result for the real model ran one block at a time. This
generates one sequencer for the whole decode step at Qwen's own size, over
the same generated blocks (the per-column multi-lane projection, RMSNorm,
the multi-lane attention head, RoPE, SiLU, the requantizer and the
residual add, all derived for 8-bit weights and 16-bit activations), and
simulates it in iverilog on the checkpoint's own weights.

What would be DDR on a board is outside the design, behind ports: a
weight word per cycle, 16 lanes of 16 bits (32 with --lanes 32, the
ZC706's width), served by the testbench from a 1 GB binary image with
$fread; a constant memory with each projection column's bias,
scale and shift; the norm gains; and the KV cache, read and written a
lane at a time. The embedding lookup reads the tied head's weight words.
The head runs as 32 projection chunks with a streaming argmax, so the
151936 logits are never stored.

The token it chooses has to be the integer model's.

FPGAI_QWEN=qwen3 builds it for Qwen3-0.6B, the model Architect Labs
hosted: no biases, q wider than the hidden state (16 heads of 128), and
an RMSNorm over every head of q and k before RoPE, run by a second
instance of the norm block sized for a head.

Needs fetch_qwen.py. Run: python3 qwen_full.py [--prompt "..."]
"""
import argparse
import math
import os
import subprocess
import sys
import time

import agent
import chiplet_flow as cf
import qwen_cosim as qc
import qwen_int as qi
import qwen_real as qr
import specgen

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "build_qfull")
N = 16                       # projection lanes at 16-bit data, unless the build asks


def lanes_of(im):
    """The build's lane count: the projection's, which the model spec can
    set (im.ms["lanes"]) and otherwise derives from the board."""
    return specgen.derive_projn_spec(im.ms, per_column=True)["parameters"]["lanes"]


def head_chunk(im):
    """Head columns per projection call: the widest matrix the model's
    projection block is sized for. A fixed 4864, Qwen2.5's MLP width,
    overflowed Qwen3's 12-bit column port (it sizes for 3072): the block
    ran 768 columns and the sequencer waited for 4864, for good."""
    c = max(im.D, im.F)
    assert im.H * im.hd <= c and c % lanes_of(im) == 0
    return c


def _clog2(n):
    return max(1, (n - 1).bit_length())


def tp_share(im, tp):
    """Rank r's share of a split of the weights, tp = (rank, ranks) for
    even shares or (rank, ranks, part) with part's per-rank lists: "kv",
    KV heads (each with its query heads); "f", d_ff columns; "d", d_model
    columns of o's and down's outputs; "hk", [first, last] head chunks
    (last < first for none). Returns this rank's counts, the offsets of
    its slices in the gathered vectors, and every rank's (for the
    network): {H, KV, F, D, coff, kvoff, aoff, moff, hk0, hk1, all}."""
    r, T = tp[0], tp[1]
    part = tp[2] if len(tp) > 2 and tp[2] else None
    H, KV, F, D, hd = im.H, im.KV, im.F, im.D, im.hd
    grp = H // KV
    nck = -(-im.V // head_chunk(im))
    if part is None:
        per = -(-nck // T)
        part = dict(kv=[KV // T] * T, f=[F // T] * T, d=[D // T] * T,
                    hk=[[k * per, min(nck - 1, k * per + per - 1)] for k in range(T)])
    kv, f, d = list(part["kv"]), list(part["f"]), list(part["d"])
    assert len(kv) == len(f) == len(d) == T and sum(kv) == KV and sum(f) == F \
        and sum(d) == D and min(kv + f + d) > 0, "a split of the weights must cover them"
    pre = lambda xs, i: sum(xs[:i])
    ranks = [dict(H=grp * kv[i], KV=kv[i], F=f[i], D=d[i],
                  coff=grp * pre(kv, i) * hd, kvoff=pre(kv, i) * hd,
                  aoff=pre(d, i), moff=pre(f, i),
                  hk0=part["hk"][i][0], hk1=part["hk"][i][1]) for i in range(T)]
    return dict(ranks[r], all=ranks)


class Layout:
    """Where every matrix, column constant and gain lives."""

    MATS = (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
            ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
            ("g", "mlp.gate_proj"), ("u", "mlp.up_proj"),
            ("d", "mlp.down_proj"))

    def __init__(self, im, nl=None, table=True, tp=None):
        D, F, H, KV, hd = im.D, im.F, im.H, im.KV, im.hd
        nl = im.NL if nl is None else nl
        self.table = table
        self.N = N = lanes_of(im)
        # tp = (rank, ranks): the weights split by output column, every
        # matrix a slice of its rows: this rank's heads of q, k and v, its
        # share of the MLP's rows, and its share of o's and down's output
        # columns over their full depth. Each value it computes is the one
        # a single board computes; the ranks gather each other's slices.
        self.tp = tp
        sh = tp_share(im, tp) if tp else dict(H=H, KV=KV, F=F, D=D, coff=0,
                                             kvoff=0, aoff=0, moff=0)
        Hr, KVr, Fr, Dr = sh["H"], sh["KV"], sh["F"], sh["D"]
        self.shape = {"q": (Hr * hd, D), "k": (KVr * hd, D), "v": (KVr * hd, D),
                      "o": (Dr, H * hd), "g": (Fr, D), "u": (Fr, D), "d": (Dr, F)}
        self.row0 = {"q": sh["coff"], "k": sh["kvoff"], "v": sh["kvoff"],
                     "o": sh["aoff"], "g": sh["moff"], "u": sh["moff"], "d": sh["aoff"]}
        self.woff, self.coff = {}, {}
        w = c = 0
        for m, _ in self.MATS:
            rows, depth = self.shape[m]
            self.woff[m], self.coff[m] = w, c
            w += (rows // N) * depth
            c += rows
        self.lw, self.lc = w, c
        self.headw = nl * self.lw
        self.headc = nl * self.lc
        self.embc = self.headc + im.V
        # A middle pipeline stage neither embeds nor runs the head, so
        # its images stop before the tied table.
        self.words = self.headw + ((im.V // N) * D if table else 0)
        self.cwords = self.embc + im.V if table else self.headc


def write_images(im, lay, work, log=print, layers=None):
    """weights.bin, cparams.hex, gains.hex. Weight words are what $fread
    reads, most significant byte first: lane 15's high byte leads."""
    s = im.s
    layers = list(range(im.NL)) if layers is None else list(layers)
    N = lay.N
    t0 = time.time()
    sign = bytes((0xff if b >= 128 else 0) for b in range(256))
    with open(os.path.join(work, "weights.bin"), "wb") as f:
        def put(key, rows, depth, row0=0):
            Q = im.Q[key]
            for g in range(rows // N):
                buf = bytearray(depth * 2 * N)
                for j in range(N):
                    c = row0 + g * N + j
                    lo = Q[c * depth:(c + 1) * depth].tobytes()
                    buf[(N - 1 - j) * 2::2 * N] = lo.translate(sign)
                    buf[(N - 1 - j) * 2 + 1::2 * N] = lo
                f.write(buf)
        for li in layers:
            for m, name in lay.MATS:
                rows, depth = lay.shape[m]
                put("model.layers.%d.%s.weight" % (li, name), rows, depth,
                    lay.row0[m])
        if lay.table:
            put("model.embed_tokens.weight", im.V, im.D)
    log("weights.bin: %.0f MB in %.0f s"
        % (os.path.getsize(os.path.join(work, "weights.bin")) / 1e6,
           time.time() - t0))
    aw = im.sp["requant"]["parameters"]["acc_width"]
    mw, sw = im.mw, im.sw
    W = im.Wf

    def word(b, sc, sh):
        return ((b & ((1 << aw) - 1)) << (sw + mw)) | (sh << mw) | sc
    with open(os.path.join(work, "cparams.hex"), "w") as f:
        for li in layers:
            P = "model.layers.%d." % li
            src = {"q": ("xn", "q"), "k": ("xn", "k"), "v": ("xn", "v"),
                   "o": ("ctx", "a"), "g": ("xn2", "g"), "u": ("xn2", "u"),
                   "d": ("m", "dn")}
            if im.qkn:
                # Qwen3's q and k leave the projection at their own
                # scales; the head norm brings them to the scores'.
                src.update(q=("xn", "qp"), k=("xn", "kp"))
            for m, name in lay.MATS:
                sx, dst = src[m]
                bias = (W[P + name + ".bias"][0]
                        if P + name + ".bias" in W else None)
                cs = list(im.proj_consts(P + name + ".weight", s[(sx, li)],
                                         s[(dst, li)], bias))
                r0, rows = lay.row0[m], lay.shape[m][0]
                for b, sc, sh in cs[r0:r0 + rows]:
                    f.write("%x\n" % word(b, sc, sh))
        if lay.table:
            for b, sc, sh in im.proj_consts("model.embed_tokens.weight",
                                            s["xf"], s["lg"]):
                f.write("%x\n" % word(b, sc, sh))
            for t in range(im.V):
                sc, sh = im.embed_consts(t)
                f.write("%x\n" % word(0, sc, sh))
    consts = []
    with open(os.path.join(work, "gains.hex"), "w") as f:
        for li in layers:
            P = "model.layers.%d." % li
            for gname, dst in (("input_layernorm", "xn"),
                               ("post_attention_layernorm", "xn2")):
                gq, sc, sh = im.norm_consts(P + gname + ".weight",
                                            s[(dst, li)])
                f.write("\n".join("%x" % (v & 0xffff) for v in gq) + "\n")
                consts.append((sc, sh))
        gq, sc, sh = im.norm_consts("model.norm.weight", s["xf"])
        f.write("\n".join("%x" % (v & 0xffff) for v in gq) + "\n")
        consts.append((sc, sh))
        if im.qkn:
            # Then each layer's q and k head-norm gains, head_dim each.
            for li in layers:
                P = "model.layers.%d." % li
                for gname, dst in (("q_norm", "q"), ("k_norm", "k")):
                    gq = im.norm_consts(P + "self_attn.%s.weight" % gname,
                                        s[(dst, li)], "headnorm")[0]
                    f.write("\n".join("%x" % (v & 0xffff) for v in gq) + "\n")
    return consts


def gain_words(im, nl=None):
    nl = im.NL if nl is None else nl
    return (2 * nl + 1) * im.D + (2 * nl * im.hd if im.qkn else 0)


def layer_consts(im, norms, layers=None):
    """Per layer: the norms' and adds' constants, the attention's shift and
    output requantizer, the gate's shift into SiLU, the product's
    requantizer."""
    s = im.s
    at = im.sp["attn"]["parameters"]
    out = []
    layers = list(range(im.NL)) if layers is None else list(layers)
    for k, li in enumerate(layers):
        sx = s["x0"] if li == 0 else s[("x2", li - 1)]
        ctx = qi.pick(2.0 ** -at["weight_frac"] * s[("v", li)] / s[("ctx", li)],
                      at["scale_width"], at["shift_width"])
        out.append(dict(
            n1=norms[2 * k], n2=norms[2 * k + 1],
            r1=im.add_consts(sx, s[("a", li)], s[("x1", li)]),
            r2=im.add_consts(s[("x1", li)], s[("dn", li)], s[("x2", li)]),
            shs=im.shs[li], ctx=ctx, gsh=im.gsh[li],
            m=qi.pick(2.0 ** -8 * s[("u", li)] / s[("m", li)], im.mw, im.sw)))
        if im.qkn:
            P = "model.layers.%d.self_attn." % li
            out[-1].update(
                qn=im.norm_consts(P + "q_norm.weight", s[("q", li)], "headnorm")[1:],
                kn=im.norm_consts(P + "k_norm.weight", s[("k", li)], "headnorm")[1:])
    return out, norms[-1]


def render(im, lay, lc, nf, stage=False):
    """(see _render) With lay.tp = (rank, ranks), one rank of a
    tensor-parallel group: its own heads, its share of every projection's
    output columns, and a gather at each point the next step needs the
    whole vector: g_req with g_vec (1 the attention context, 2 o's or
    down's output, 3 the gated product, 4 the head's best), until g_done,
    while the network reads this rank's slice and writes the others' into
    that vector through gx_addr, gx_wdata and gx_rdata, a word a clock.
    The ranks then hold identical vectors, a single board's."""
    return _render(im, lay, lc, nf, stage)


def _render(im, lay, lc, nf, stage=False):
    """The sequencer. stage: a pipeline stage of a multi-board run, with
    a port to load the hidden state it starts from and read back the one
    it ends with while idle, and emb_en to choose between that and the
    embedding lookup."""
    D, F, H, KV, hd, V = im.D, im.F, im.H, im.KV, im.hd, im.V
    NL = len(lc)
    N = lay.N
    LB = _clog2(N)                  # lane index bits
    L = im.ms["seq_len"]
    grp, h2 = H // KV, hd // 2
    tp = lay.tp
    rank, T = tp[:2] if tp else (0, 1)
    # This rank's heads and slices; the offsets of its slice in the
    # gathered vectors.
    sh = tp_share(im, tp) if tp else dict(H=H, KV=KV, F=F, D=D, coff=0, aoff=0, moff=0)
    Hl, KVl, Fl, Dl = sh["H"], sh["KV"], sh["F"], sh["D"]
    COFF, AOFF, MOFF = sh["coff"], sh["aoff"], sh["moff"]
    ps = specgen.derive_projn_spec(im.ms, per_column=True)["parameters"]
    rn = im.sp["rmsnorm"]["parameters"]
    ap = specgen.derive_attnn_spec(im.ms)["parameters"]
    ro = im.sp["rope"]["parameters"]
    sl = im.sp["silu"]["parameters"]
    rq = im.sp["requant"]["parameters"]
    ra = im.sp["resadd"]["parameters"]
    assert ps["lanes"] == N and ap["lanes"] == N
    tw, pw = _clog2(V), ro["pos_width"]
    WA, CA = _clog2(lay.words), _clog2(lay.cwords)
    qkn = im.qkn
    GA = _clog2(gain_words(im, NL))
    GQ = (2 * NL + 1) * D
    KWN = NL * KVl * (L // N) * hd
    VWN = NL * KVl * L * (hd // N)
    KA, VA = _clog2(KWN), _clog2(VWN)
    CW = ps["col_word_width"]
    mw, sw, aw = rq["scale_width"], rq["shift_width"], rq["acc_width"]
    siw = sl["width"]
    cw_ = 13
    states = ["S_IDLE", "S_EMB", "S_N1", "S_Q", "S_K", "S_V", "S_RQ", "S_RK",
              "S_LQ", "S_ATT", "S_O", "S_R1", "S_N2", "S_G", "S_U", "S_GLU",
              "S_DN", "S_R2", "S_NF", "S_HD"] + (["S_QN", "S_KN"] if qkn else []) \
        + (["S_GC", "S_GA", "S_GM", "S_GH"] if tp else [])
    stw = _clog2(len(states))
    A = []
    a = A.append
    a("// GENERATED by qwen_full.py: do not edit by hand.")
    a("// One %s decode step, embedding to argmax, over the"
      % ("Qwen3-0.6B" if qkn else "Qwen2.5-0.5B"))
    a("// generated blocks; weights, constants and the KV cache are external.")
    a("module qwen_full (")
    a("  input clk, input rst_n, input start, input head_en,")
    a("  input [%d:0] tok, input [%d:0] pos," % (tw - 1, pw - 1))
    a("  output [%d:0] w_addr, input [%d:0] w_data," % (WA - 1, N * 16 - 1))
    a("  output [%d:0] c_addr, input [%d:0] c_data," % (CA - 1, CW - 1))
    a("  output [%d:0] g_addr, input signed [15:0] g_data," % (GA - 1))
    a("  output [%d:0] k_raddr, input [%d:0] k_rdata," % (KA - 1, N * 16 - 1))
    a("  output [%d:0] v_raddr, input [%d:0] v_rdata," % (VA - 1, N * 16 - 1))
    a("  output reg kw0_en, output reg [%d:0] kw0_addr, output reg [%d:0] kw0_lane," % (KA - 1, LB - 1))
    a("  output reg signed [15:0] kw0_data,")
    a("  output reg kw1_en, output reg [%d:0] kw1_addr, output reg [%d:0] kw1_lane," % (KA - 1, LB - 1))
    a("  output reg signed [15:0] kw1_data,")
    a("  output reg vw_en, output reg [%d:0] vw_addr, output reg [%d:0] vw_lane," % (VA - 1, LB - 1))
    a("  output reg signed [15:0] vw_data,")
    a("  output reg [%d:0] next_tok, output reg signed [15:0] best," % (tw - 1))
    if stage:
        a("  // The hidden state in and out, while idle: a pipeline stage.")
        a("  input emb_en, input x_we, input [%d:0] x_addr, input signed [15:0] x_wdata," % (_clog2(D) - 1))
        a("  output reg signed [15:0] x_rdata,")
    if tp:
        a("  // Rank %d of %d: the gathers, and the port the network fills them" % (rank, T))
        a("  // through: element gx_addr of the vector g_vec names, while g_req.")
        a("  output reg g_req, output reg [2:0] g_vec, input g_done,")
        a("  input gx_we, input [%d:0] gx_addr, input signed [15:0] gx_wdata,"
          % (_clog2(max(H * hd, D, F)) - 1))
        a("  output reg signed [15:0] gx_rdata,")
    a("  output reg done, output reg busy")
    a(");")
    a("  localparam " + ", ".join("%s = %d'd%d" % (x, stw, i)
                                  for i, x in enumerate(states)) + ";")
    a("  reg [%d:0] st;" % (stw - 1))
    a("  reg ph, hen;")
    if tp:
        a("  reg gph;                     // which of o's and down's outputs is gathered")
    a("  reg [%d:0] lyr;" % (max(5, _clog2(NL)) - 1))
    a("  reg [%d:0] hh;" % (max(4, _clog2(H)) - 1))
    chunk = head_chunk(im)
    nlast = -(-V // chunk) - 1          # the last chunk, never empty
    # A rank runs the head over its own run of chunks.
    hk0, hk1 = (sh["hk0"], sh["hk1"]) if tp else (0, nlast)
    a("  reg [%d:0] hk;" % (max(6, _clog2(nlast + 1)) - 1))
    a("  reg [%d:0] tok_r;" % (tw - 1))
    a("  reg [%d:0] pos_r;" % (pw - 1))
    for name, size in (("xm", D), ("nm", D), ("qlo", H * h2), ("qhi", H * h2),
                       ("klo", KV * h2), ("khi", KV * h2), ("cm", H * hd),
                       ("am", D), ("gm", F), ("um", F), ("mm", F)):
        a("  reg signed [15:0] %s [0:%d];" % (name, size - 1))
    a("  reg [%d:0] fj, ocnt, gcnt;" % (cw_ - 1))
    a("")
    # ---- projection
    a("  reg pj_start;")
    a("  reg [12:0] pj_depth, pj_cols;")
    a("  wire [12:0] pj_a_addr, pj_c_addr, pj_index;")
    a("  wire [%d:0] pj_w_addr;" % (ps["word_addr_width"] - 1))
    a("  reg signed [15:0] pj_a_data;")
    a("  wire pj_valid, pj_busy;")
    a("  wire signed [15:0] pj_data;")
    a("  projn u_proj (.clk(clk), .rst_n(rst_n), .start(pj_start),")
    a("    .depth(pj_depth), .cols(pj_cols), .scale(%d'd0), .shift(%d'd0)," % (mw, sw))
    a("    .a_addr(pj_a_addr), .a_data(pj_a_data), .w_addr(pj_w_addr),")
    a("    .w_data(w_data), .c_addr(pj_c_addr), .c_data(c_data),")
    a("    .o_valid(pj_valid), .o_index(pj_index), .o_data(pj_data), .busy(pj_busy));")
    # ---- norm
    a("  reg rn_start;")
    a("  reg [%d:0] rn_scale;" % (rn["scale_width"] - 1))
    a("  reg [%d:0] rn_shift;" % (rn["shift_width"] - 1))
    a("  wire [%d:0] rn_x_addr, rn_g_addr, rn_index;" % (rn["addr_width"] - 1))
    a("  reg signed [15:0] rn_x_data;")
    a("  wire rn_valid, rn_busy;")
    a("  wire signed [15:0] rn_data;")
    a("  rmsnorm u_norm (.clk(clk), .rst_n(rst_n), .start(rn_start),")
    a("    .eps(%d'd1), .scale_o(rn_scale), .shift_o(rn_shift)," % rn["rsqrt_in_width"])
    a("    .x_addr(rn_x_addr), .x_data(rn_x_data), .g_addr(rn_g_addr),")
    a("    .g_data(g_data), .o_valid(rn_valid), .o_index(rn_index),")
    a("    .o_data(rn_data), .busy(rn_busy));")
    if qkn:
        hn = im.sp["headnorm"]["parameters"]
        a("  // Qwen3: the same norm over one head of q or k, in place.")
        a("  reg hn_start;")
        a("  reg [%d:0] hn_scale;" % (hn["scale_width"] - 1))
        a("  reg [%d:0] hn_shift;" % (hn["shift_width"] - 1))
        a("  wire [%d:0] hn_x_addr, hn_g_addr, hn_index;" % (hn["addr_width"] - 1))
        a("  reg signed [15:0] hn_x_data;")
        a("  wire hn_valid, hn_busy;")
        a("  wire signed [15:0] hn_data;")
        a("  rmsnorm_hd u_hnorm (.clk(clk), .rst_n(rst_n), .start(hn_start),")
        a("    .eps(%d'd1), .scale_o(hn_scale), .shift_o(hn_shift)," % hn["rsqrt_in_width"])
        a("    .x_addr(hn_x_addr), .x_data(hn_x_data), .g_addr(hn_g_addr),")
        a("    .g_data(g_data), .o_valid(hn_valid), .o_index(hn_index),")
        a("    .o_data(hn_data), .busy(hn_busy));")
    # ---- attention
    a("  reg at_start, at_load_valid;")
    a("  reg signed [15:0] at_load_data;")
    a("  reg [4:0] at_shs;")
    a("  reg [%d:0] at_scale;" % (ap["scale_width"] - 1))
    a("  reg [%d:0] at_shift;" % (ap["shift_width"] - 1))
    a("  wire [%d:0] at_k_addr;" % (ap["k_addr_width"] - 1))
    a("  wire [%d:0] at_v_addr;" % (ap["v_addr_width"] - 1))
    a("  wire at_valid, at_busy;")
    a("  wire [%d:0] at_index;" % (ap["head_dim_width"] - 1))
    a("  wire signed [15:0] at_data;")
    a("  attnn u_attn (.clk(clk), .rst_n(rst_n), .load_valid(at_load_valid),")
    a("    .load_data(at_load_data), .start(at_start),")
    a("    .n({%d'd0, pos_r} + %d'd1), .shift_s(at_shs)," % (ap["n_width"] - pw, ap["n_width"]))
    a("    .scale_o(at_scale), .shift_o(at_shift), .k_addr(at_k_addr),")
    a("    .k_data(k_rdata), .v_addr(at_v_addr), .v_data(v_rdata),")
    a("    .o_valid(at_valid), .o_index(at_index), .o_data(at_data), .busy(at_busy));")
    # ---- residual add, rope, silu, requant
    a("  reg signed [15:0] ra_a, ra_b;")
    a("  reg ra_v;")
    a("  reg [%d:0] ra_sa, ra_sb;" % (ra["scale_width"] - 1))
    a("  reg [%d:0] ra_sh;" % (ra["shift_width"] - 1))
    a("  wire signed [15:0] ra_y;")
    a("  wire ra_vout;")
    a("  resadd u_add (.clk(clk), .rst_n(rst_n), .a(ra_a), .b(ra_b),")
    a("    .scale_a(ra_sa), .scale_b(ra_sb), .shift(ra_sh), .valid_in(ra_v),")
    a("    .y(ra_y), .valid_out(ra_vout));")
    a("  reg signed [15:0] ro_x1, ro_x2;")
    a("  reg [%d:0] ro_idx;" % (ro["index_width"] - 1))
    a("  reg ro_v;")
    a("  wire signed [15:0] ro_y1, ro_y2;")
    a("  wire ro_vout;")
    a("  rope u_rope (.clk(clk), .rst_n(rst_n), .x1(ro_x1), .x2(ro_x2),")
    a("    .idx(ro_idx), .pos(pos_r), .valid_in(ro_v), .y1(ro_y1), .y2(ro_y2),")
    a("    .valid_out(ro_vout));")
    a("  reg signed [%d:0] si_x;" % (siw - 1))
    a("  reg si_v;")
    a("  wire signed [%d:0] si_y;" % (siw - 1))
    a("  wire si_vout;")
    a("  silu u_silu (.clk(clk), .rst_n(rst_n), .x(si_x), .valid_in(si_v),")
    a("    .y(si_y), .valid_out(si_vout));")
    a("  reg signed [%d:0] rq_acc;" % (aw - 1))
    a("  reg rq_v;")
    a("  reg [%d:0] rq_scale;" % (mw - 1))
    a("  reg [%d:0] rq_shift;" % (sw - 1))
    a("  wire signed [15:0] rq_q;")
    a("  wire rq_sat, rq_vout;")
    a("  requant u_rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc), .scale(rq_scale),")
    a("    .shift(rq_shift), .valid_in(rq_v), .q_out(rq_q), .sat(rq_sat),")
    a("    .valid_out(rq_vout));")
    a("")
    # ---- per-layer constants
    a("  reg [%d:0] c_n1s, c_n2s, c_ctxs, c_ms, c_r1a, c_r1b, c_r2a, c_r2b;" % (mw - 1))
    a("  reg [%d:0] c_n1h, c_n2h, c_ctxh, c_mh, c_r1h, c_r2h;" % (sw - 1))
    a("  reg [4:0] c_shs;")
    a("  reg signed [4:0] c_gsh;")
    if qkn:
        a("  reg [%d:0] c_qns, c_kns;" % (hn["scale_width"] - 1))
        a("  reg [%d:0] c_qnh, c_knh;" % (hn["shift_width"] - 1))
    a("  always @(*) begin")
    a("    case (lyr)")
    for li, c in enumerate(lc):
        a("      %d: begin c_n1s = %d; c_n1h = %d; c_n2s = %d; c_n2h = %d;"
          % ((li,) + tuple(c["n1"]) + tuple(c["n2"])))
        a("        c_r1a = %d; c_r1b = %d; c_r1h = %d; c_r2a = %d; c_r2b = %d; c_r2h = %d;"
          % (tuple(c["r1"]) + tuple(c["r2"])))
        a("        c_shs = %d; c_ctxs = %d; c_ctxh = %d; c_gsh = %d; c_ms = %d; c_mh = %d;%s end"
          % ((c["shs"],) + tuple(c["ctx"]) + (c["gsh"],) + tuple(c["m"])
             + ((" c_qns = %d; c_qnh = %d; c_kns = %d; c_knh = %d;"
                 % (tuple(c["qn"]) + tuple(c["kn"]))) if qkn else "",)))
    a("      default: begin c_n1s = 0; c_n1h = 0; c_n2s = 0; c_n2h = 0; c_r1a = 0;")
    a("        c_r1b = 0; c_r1h = 0; c_r2a = 0; c_r2b = 0; c_r2h = 0; c_shs = 0;")
    a("        c_ctxs = 0; c_ctxh = 0; c_gsh = 0; c_ms = 0; c_mh = 0;%s end"
      % (" c_qns = 0; c_qnh = 0; c_kns = 0; c_knh = 0;" if qkn else ""))
    a("    endcase")
    a("  end")
    a("")
    # ---- bases
    a("  wire [%d:0] lw = lyr * %d;" % (WA - 1, lay.lw))
    a("  wire [%d:0] lcb = lyr * %d;" % (CA - 1, lay.lc))
    a("  reg [%d:0] wbase;" % (WA - 1))
    a("  reg [%d:0] cbase;" % (CA - 1))
    a("  reg [%d:0] gbase;" % (GA - 1))
    a("  reg [12:0] expect_n;")
    a("  wire [%d:0] hcols = (hk == %d) ? %d : %d;"
      % (12, nlast, V - nlast * chunk, chunk))
    a("  always @(*) begin")
    a("    wbase = 0; cbase = 0; gbase = 0; pj_depth = %d; pj_cols = %d;" % (D, D))
    a("    expect_n = %d; rn_scale = 0; rn_shift = 0; ra_sa = 0; ra_sb = 0; ra_sh = 0;" % D)
    a("    at_shs = c_shs; at_scale = c_ctxs; at_shift = c_ctxh;")
    if qkn:
        a("    hn_scale = 0; hn_shift = 0;")
    a("    case (st)")
    a("      S_EMB: begin wbase = %d + (tok_r >> %d) * %d; cbase = %d + tok_r; end"
      % (lay.headw, LB, D, lay.embc))
    a("      S_N1: begin gbase = lyr * %d; rn_scale = c_n1s; rn_shift = c_n1h; end" % (2 * D))
    a("      S_N2: begin gbase = lyr * %d + %d; rn_scale = c_n2s; rn_shift = c_n2h; end" % (2 * D, D))
    a("      S_NF: begin gbase = %d; rn_scale = %d; rn_shift = %d; end"
      % ((2 * NL * D,) + tuple(nf)))
    for s_, m in (("S_Q", "q"), ("S_K", "k"), ("S_V", "v"), ("S_O", "o"),
                  ("S_G", "g"), ("S_U", "u"), ("S_DN", "d")):
        rows, depth = lay.shape[m]
        a("      %s: begin wbase = lw + %d; cbase = lcb + %d; pj_depth = %d;"
          " pj_cols = %d; expect_n = %d; end"
          % (s_, lay.woff[m], lay.coff[m], depth, rows, rows))
    a("      S_HD: begin wbase = %d + hk * %d; cbase = %d + hk * %d;"
      " pj_cols = hcols; expect_n = hcols; end"
      % (lay.headw, (chunk // N) * D, lay.headc, chunk))
    a("      S_R1: begin ra_sa = c_r1a; ra_sb = c_r1b; ra_sh = c_r1h; end")
    a("      S_R2: begin ra_sa = c_r2a; ra_sb = c_r2b; ra_sh = c_r2h; end")
    a("      S_ATT: expect_n = %d;" % hd)
    if qkn:
        a("      S_QN: begin gbase = %d + lyr * %d; hn_scale = c_qns; hn_shift = c_qnh;"
          " expect_n = %d; end" % (GQ, 2 * hd, hd))
        a("      S_KN: begin gbase = %d + lyr * %d + %d; hn_scale = c_kns; hn_shift = c_knh;"
          " expect_n = %d; end" % (GQ, 2 * hd, hd, hd))
    a("      S_RQ: expect_n = %d;" % (Hl * h2))
    a("      S_RK: expect_n = %d;" % (KVl * h2))
    a("      S_GLU: expect_n = %d;" % Fl)
    a("      default: ;")
    a("    endcase")
    a("  end")
    a("  wire in_norm = (st == S_N1) || (st == S_N2) || (st == S_NF);")
    if qkn:
        a("  wire in_hnorm = (st == S_QN) || (st == S_KN);")
    a("  reg [12:0] ed;")
    a("  assign w_addr = wbase + ((st == S_EMB) ? ed : pj_w_addr);")
    a("  assign c_addr = cbase + ((st == S_EMB) ? 13'd0 : pj_c_addr);")
    a("  assign g_addr = gbase + %s;"
      % ("(in_hnorm ? hn_g_addr : rn_g_addr)" if qkn else "rn_g_addr"))
    a("  // The KV head this query head reads.")
    a("  wire [%d:0] kvsel = lyr * %d + hh / %d;" % (KA - 1, KVl, grp))
    a("  assign k_raddr = kvsel * %d + at_k_addr;" % ((L // N) * hd))
    a("  assign v_raddr = kvsel * %d + at_v_addr;" % (L * (hd // N)))
    a("  always @(posedge clk) begin")
    a("    rn_x_data <= xm[rn_x_addr];")
    if stage:
        a("    x_rdata <= xm[x_addr];")
    if tp:
        a("    gx_rdata <= (g_vec == 3'd1) ? cm[gx_addr] : (g_vec == 3'd2) ? am[gx_addr] : mm[gx_addr];")
    a("    pj_a_data <= (st == S_O) ? cm[pj_a_addr] : (st == S_DN) ? mm[pj_a_addr] : nm[pj_a_addr];")
    if qkn:
        # Element i of head hh: the first half of a head is in the lo
        # array, the second in the hi one, h2 to a head in each.
        hb = _clog2(h2)
        a("    if (st == S_QN) hn_x_data <= hn_x_addr[%d] ? qhi[hh * %d + hn_x_addr[%d:0]]"
          " : qlo[hh * %d + hn_x_addr[%d:0]];" % (hb, h2, hb - 1, h2, hb - 1))
        a("    else hn_x_data <= hn_x_addr[%d] ? khi[hh * %d + hn_x_addr[%d:0]]"
          " : klo[hh * %d + hn_x_addr[%d:0]];" % (hb, h2, hb - 1, h2, hb - 1))
    a("  end")
    a("  wire run_proj = (st == S_Q) || (st == S_K) || (st == S_V) || (st == S_O) ||")
    a("                  (st == S_G) || (st == S_U) || (st == S_DN) || (st == S_HD);")
    a("  wire blk_busy = run_proj ? pj_busy : in_norm ? rn_busy : %sat_busy;"
      % ("in_hnorm ? hn_busy : " if qkn else ""))
    a("  wire pj_half = pj_index[%d];" % _clog2(h2))
    a("  wire [12:0] pj_hadr = ((pj_index >> %d) << %d) | (pj_index & %d);"
      % (_clog2(hd), _clog2(h2), h2 - 1))
    a("  wire [12:0] fh = fj >> %d, fp = fj & %d;" % (_clog2(h2), h2 - 1))
    a("  wire [12:0] oh = ocnt >> %d, op = ocnt & %d;" % (_clog2(h2), h2 - 1))
    a("  // KV cache words: keys interleaved by position, values by dimension.")
    a("  wire [%d:0] kslot = (lyr * %d + oh) * %d + (pos_r >> %d) * %d;"
      % (KA - 1, KVl, (L // N) * hd, LB, hd))
    a("  wire [%d:0] vslot = ((lyr * %d + (pj_index >> %d)) * %d + pos_r) * %d"
      " + ((pj_index & %d) >> %d);" % (VA - 1, KVl, _clog2(hd), L, hd // N, hd - 1, LB))
    a("  reg ev1;")
    a("  reg [%d:0] lane_r;" % (LB - 1))
    a("  wire signed [15:0] ew = w_data[lane_r * 16 +: 16];")
    a("  reg [%d:0] hbase;" % (tw - 1))
    a("")
    a("  always @(posedge clk) begin")
    a("    if (!rst_n) begin")
    a("      st <= S_IDLE; ph <= 1'b0; busy <= 1'b0; done <= 1'b0; hen <= 1'b0;")
    a("      lyr <= 0; hh <= 0; hk <= 0; tok_r <= 0; pos_r <= 0; next_tok <= 0;")
    a("      best <= 0; fj <= 0; ocnt <= 0; gcnt <= 0; ed <= 0; ev1 <= 1'b0; lane_r <= 0;%s"
      % (" g_req <= 1'b0; g_vec <= 0; gph <= 1'b0;" if tp else ""))
    a("      pj_start <= 1'b0; rn_start <= 1'b0; at_start <= 1'b0; at_load_valid <= 1'b0;%s"
      % (" hn_start <= 1'b0;" if qkn else ""))
    a("      at_load_data <= 0; ra_a <= 0; ra_b <= 0; ra_v <= 1'b0; ro_x1 <= 0;")
    a("      ro_x2 <= 0; ro_idx <= 0; ro_v <= 1'b0; si_x <= 0; si_v <= 1'b0;")
    a("      rq_acc <= 0; rq_v <= 1'b0; rq_scale <= 0; rq_shift <= 0;")
    a("      kw0_en <= 1'b0; kw1_en <= 1'b0; vw_en <= 1'b0; kw0_addr <= 0; kw1_addr <= 0;")
    a("      vw_addr <= 0; kw0_lane <= 0; kw1_lane <= 0; vw_lane <= 0; kw0_data <= 0;")
    a("      kw1_data <= 0; vw_data <= 0; hbase <= 0;")
    a("    end else begin")
    a("      done <= 1'b0; pj_start <= 1'b0; rn_start <= 1'b0; at_start <= 1'b0;%s"
      % (" hn_start <= 1'b0;" if qkn else ""))
    a("      at_load_valid <= 1'b0; ra_v <= 1'b0; ro_v <= 1'b0; si_v <= 1'b0;")
    a("      rq_v <= 1'b0; kw0_en <= 1'b0; kw1_en <= 1'b0; vw_en <= 1'b0;")
    a("      if (run_proj && pj_valid) begin")
    a("        ocnt <= ocnt + 1;")
    a("        case (st)")
    a("          S_Q: if (pj_half) qhi[pj_hadr] <= pj_data; else qlo[pj_hadr] <= pj_data;")
    a("          S_K: if (pj_half) khi[pj_hadr] <= pj_data; else klo[pj_hadr] <= pj_data;")
    a("          S_V: begin vw_en <= 1'b1; vw_addr <= vslot; vw_lane <= pj_index[%d:0];" % (LB - 1))
    a("                 vw_data <= pj_data; end")
    a("          S_O, S_DN: am[%spj_index] <= pj_data;" % ("%d + " % AOFF if AOFF else ""))
    a("          S_G: gm[pj_index] <= pj_data;")
    a("          S_U: um[pj_index] <= pj_data;")
    a("          S_HD: if ((hk == %d && ocnt == 0) || pj_data > best) begin" % hk0)
    a("                  best <= pj_data; next_tok <= hbase + pj_index; end")
    a("          default: ;")
    a("        endcase")
    a("      end")
    a("      if (in_norm && rn_valid) begin nm[rn_index] <= rn_data; ocnt <= ocnt + 1; end")
    if stage:
        a("      if (st == S_IDLE && x_we) xm[x_addr] <= x_wdata;")
    if tp:
        a("      if (g_req && gx_we)")
        a("        case (g_vec)")
        a("          3'd1: cm[gx_addr] <= gx_wdata;")
        a("          3'd2: am[gx_addr] <= gx_wdata;")
        a("          3'd3: mm[gx_addr] <= gx_wdata;")
        a("          default: ;")
        a("        endcase")
    if qkn:
        a("      if (in_hnorm && hn_valid) begin")
        a("        if (st == S_QN) begin")
        a("          if (hn_index[%d]) qhi[hh * %d + hn_index[%d:0]] <= hn_data;"
          " else qlo[hh * %d + hn_index[%d:0]] <= hn_data;" % (hb, h2, hb - 1, h2, hb - 1))
        a("        end else begin")
        a("          if (hn_index[%d]) khi[hh * %d + hn_index[%d:0]] <= hn_data;"
          " else klo[hh * %d + hn_index[%d:0]] <= hn_data;" % (hb, h2, hb - 1, h2, hb - 1))
        a("        end")
        a("        ocnt <= ocnt + 1;")
        a("      end")
    a("      if (st == S_ATT && at_valid) begin")
    a("        cm[%shh * %d + at_index] <= at_data; ocnt <= ocnt + 1; end"
      % ("%d + " % COFF if COFF else "", hd))
    a("      if ((st == S_R1 || st == S_R2) && ra_vout) begin xm[ocnt] <= ra_y; ocnt <= ocnt + 1; end")
    a("      if (st == S_RQ && ro_vout) begin qlo[ocnt] <= ro_y1; qhi[ocnt] <= ro_y2; ocnt <= ocnt + 1; end")
    a("      if (st == S_RK && ro_vout) begin")
    a("        kw0_en <= 1'b1; kw0_addr <= kslot + op; kw0_lane <= pos_r[%d:0]; kw0_data <= ro_y1;" % (LB - 1))
    a("        kw1_en <= 1'b1; kw1_addr <= kslot + op + %d; kw1_lane <= pos_r[%d:0]; kw1_data <= ro_y2;" % (h2, LB - 1))
    a("        ocnt <= ocnt + 1;")
    a("      end")
    a("      if (st == S_GLU && si_vout) begin")
    a("        rq_acc <= si_y * um[gcnt]; rq_v <= 1'b1; gcnt <= gcnt + 1; end")
    a("      if ((st == S_GLU || st == S_EMB) && rq_vout) begin")
    a("        if (st == S_GLU) mm[%socnt] <= rq_q; else xm[ocnt] <= rq_q;"
      % ("%d + " % MOFF if MOFF else ""))
    a("        ocnt <= ocnt + 1;")
    a("      end")
    a("")
    a("      case (st)")
    a("        S_IDLE: if (start) begin")
    a("          tok_r <= tok; pos_r <= pos; hen <= head_en; busy <= 1'b1; lyr <= 0;")
    a("          hh <= 0; hk <= %d; st <= %s; ed <= 0; ev1 <= 1'b0; ocnt <= 0; ph <= 1'b0;"
      % (hk0, "emb_en ? S_EMB : S_N1" if stage else "S_EMB"))
    a("          lane_r <= tok[%d:0];" % (LB - 1))
    a("        end")
    a("        // The embedding row out of the head's weight words, requantized")
    a("        // with this token's own constants.")
    a("        S_EMB: begin")
    a("          rq_scale <= c_data[%d:0]; rq_shift <= c_data[%d:%d];" % (mw - 1, sw + mw - 1, mw))
    a("          if (ed < %d) ed <= ed + 1;" % D)
    a("          ev1 <= (ed < %d);" % D)
    a("          if (ev1) begin rq_acc <= ew; rq_v <= 1'b1; end")
    a("          if (ocnt == %d) begin ocnt <= 0; fj <= 0; ph <= 1'b0; st <= S_N1; end" % D)
    a("        end")
    a("        S_R1, S_R2: begin")
    a("          if (fj < %d) begin ra_a <= xm[fj]; ra_b <= am[fj]; ra_v <= 1'b1; fj <= fj + 1; end" % D)
    a("          if (ocnt == %d) begin" % D)
    a("            fj <= 0; ocnt <= 0; ph <= 1'b0;")
    a("            if (st == S_R1) st <= S_N2;")
    a("            else if (lyr + 1 < %d) begin lyr <= lyr + 1; st <= S_N1; end" % NL)
    a("            else if (hen) begin st <= S_NF; end")
    a("            else begin busy <= 1'b0; done <= 1'b1; st <= S_IDLE; end")
    a("          end")
    a("        end")
    a("        S_RQ, S_RK: begin")
    a("          if (fj < expect_n) begin")
    a("            ro_x1 <= (st == S_RQ) ? qlo[fj] : klo[fj];")
    a("            ro_x2 <= (st == S_RQ) ? qhi[fj] : khi[fj];")
    a("            ro_idx <= fp[%d:0]; ro_v <= 1'b1; fj <= fj + 1;" % (ro["index_width"] - 1))
    a("          end")
    a("          if (ocnt == expect_n) begin fj <= 0; ocnt <= 0; ph <= 1'b0;")
    a("            st <= (st == S_RQ) ? S_RK : S_LQ; end")
    a("        end")
    a("        S_LQ: begin")
    a("          if (fj < %d) begin" % hd)
    a("            at_load_valid <= 1'b1;")
    a("            at_load_data <= fj[%d] ? qhi[hh * %d + fj[%d:0]] : qlo[hh * %d + fj[%d:0]];"
      % (_clog2(h2), h2, _clog2(h2) - 1, h2, _clog2(h2) - 1))
    a("            fj <= fj + 1;")
    a("          end else begin fj <= 0; ocnt <= 0; ph <= 1'b0; st <= S_ATT; end")
    a("        end")
    a("        S_GLU: begin")
    a("          rq_scale <= c_ms; rq_shift <= c_mh;")
    a("          if (fj < %d) begin" % Fl)
    a("            si_x <= (c_gsh >= 0) ? (gm[fj] <<< c_gsh) : (gm[fj] >>> (-c_gsh));")
    a("            si_v <= 1'b1; fj <= fj + 1;")
    a("          end")
    a("          if (ocnt == %d) begin fj <= 0; ocnt <= 0; gcnt <= 0; ph <= 1'b0; st <= %s; end"
      % (Fl, "S_GM" if tp else "S_DN"))
    a("        end")
    if tp:
        # Each gather: ask once, then wait for the network; the arrays
        # are written from outside meanwhile, and nothing here reads them.
        for gs, vec, nxt in (("S_GC", 1, "st <= S_O;"),
                             ("S_GA", 2, "st <= gph ? S_R2 : S_R1;"),
                             ("S_GM", 3, "st <= S_DN;"),
                             ("S_GH", 4, "begin busy <= 1'b0; done <= 1'b1; st <= S_IDLE; end")):
            a("        %s: if (!ph) begin ph <= 1'b1; g_req <= 1'b1; g_vec <= 3'd%d; end" % (gs, vec))
            a("            else if (g_done) begin g_req <= 1'b0; ph <= 1'b0; %s end" % nxt)
    a("        default: begin")
    a("          if (!ph) begin")
    a("            ph <= 1'b1; ocnt <= 0;")
    a("            if (run_proj) pj_start <= 1'b1;")
    a("            else if (in_norm) rn_start <= 1'b1;")
    if qkn:
        a("            else if (in_hnorm) hn_start <= 1'b1;")
    a("            else at_start <= 1'b1;")
    a("            if (st == S_HD) hbase <= hk * %d;" % chunk)
    a("          end else if (!blk_busy && ocnt == expect_n && !pj_start && !rn_start && !at_start%s) begin"
      % (" && !hn_start" if qkn else ""))
    a("            ph <= 1'b0; ocnt <= 0; fj <= 0;")
    a("            case (st)")
    a("              S_N1: st <= S_Q;")
    a("              S_Q: st <= S_K;")
    a("              S_K: st <= S_V;")
    a("              S_V: st <= %s;" % ("S_QN" if qkn else "S_RQ"))
    if qkn:
        a("              S_QN: if (hh + 1 < %d) hh <= hh + 1;" % Hl)
        a("                    else begin hh <= 0; st <= S_KN; end")
        a("              S_KN: if (hh + 1 < %d) hh <= hh + 1;" % KVl)
        a("                    else begin hh <= 0; st <= S_RQ; end")
    a("              S_ATT: if (hh + 1 < %d) begin hh <= hh + 1; st <= S_LQ; end" % Hl)
    a("                     else begin hh <= 0; st <= %s; end" % ("S_GC" if tp else "S_O"))
    a("              S_O: %s" % ("begin gph <= 1'b0; st <= S_GA; end" if tp else "st <= S_R1;"))
    a("              S_N2: st <= S_G;")
    a("              S_G: st <= S_U;")
    a("              S_U: st <= S_GLU;")
    a("              S_DN: %s" % ("begin gph <= 1'b1; st <= S_GA; end" if tp else "st <= S_R2;"))
    if tp and hk0 > hk1:
        # A rank with no chunk of the vocabulary left: no head, and a
        # logit no other rank's can lose to.
        a("              S_NF: begin best <= -16'sd32768; st <= S_GH; end")
    else:
        a("              S_NF: st <= S_HD;")
    a("              S_HD: if (hk < %d) hk <= hk + 1;" % hk1)
    a("                    else %s" % ("st <= S_GH;" if tp else
                                   "begin busy <= 1'b0; done <= 1'b1; st <= S_IDLE; end"))
    a("              default: st <= S_IDLE;")
    a("            endcase")
    a("          end")
    a("        end")
    a("      endcase")
    a("    end")
    a("  end")
    a("endmodule")
    return "\n".join(A) + "\n", dict(WA=WA, CA=CA, GA=GA, KA=KA, VA=VA,
                                     KWN=KWN, VWN=VWN, CW=CW, tw=tw, pw=pw,
                                     N=N, LB=LB, GX=_clog2(max(H * hd, D, F)))


TB = """`timescale 1ns/1ps
module tb_qfull;
  reg clk = 0, rst_n = 0, start = 0, head_en = 0;
  reg [%(tw)d:0] tok = 0;
  reg [%(pw)d:0] pos = 0;
  wire [%(WA)d:0] w_addr;
  reg [%(LW)d:0] w_data, wtmp;
  wire [%(CA)d:0] c_addr;
  reg [%(CW)d:0] c_data;
  wire [%(GA)d:0] g_addr;
  reg signed [15:0] g_data;
  wire [%(KA)d:0] k_raddr, kw0_addr, kw1_addr;
  wire [%(VA)d:0] v_raddr, vw_addr;
  reg [%(LW)d:0] k_rdata, v_rdata, kt;
  wire kw0_en, kw1_en, vw_en;
  wire [%(LB)d:0] kw0_lane, kw1_lane, vw_lane;
  wire signed [15:0] kw0_data, kw1_data, vw_data;
  wire [%(tw)d:0] next_tok;
  wire signed [15:0] best;
  wire done, busy;
  reg [%(CW)d:0] cmem [0:%(cn)d];
  reg signed [15:0] gmem [0:%(gn)d];
  reg [%(LW)d:0] km [0:%(kn)d];
  reg [%(LW)d:0] vm [0:%(vn)d];
  integer fd, r, cyc = 0, t0 = 0, i;
  reg [%(WA)d:0] wlast;
  always #5 clk = ~clk;
  always @(posedge clk) cyc = cyc + 1;
  // A heartbeat with the sequencer's state, so a hang reads as one.
  always @(posedge clk) if (cyc %% 10000000 == 0 && cyc > 0) begin
    $display("PROGRESS cyc=%%0d st=%%0d lyr=%%0d hk=%%0d ocnt=%%0d", cyc, dut.st, dut.lyr, dut.hk, dut.ocnt);
    $fflush;
  end
  // What would be DDR: one weight word a cycle, from the image on disk.
  always @(posedge clk) begin
    if (w_addr !== wlast) begin
      if (w_addr !== wlast + 1) r = $fseek(fd, w_addr * %(WB)d, 0);
      r = $fread(wtmp, fd);
      wlast = w_addr;
    end
    w_data <= wtmp;
    c_data <= cmem[c_addr];
    g_data <= gmem[g_addr];
    k_rdata <= km[k_raddr];
    v_rdata <= vm[v_raddr];
    if (kw0_en) begin kt = km[kw0_addr]; kt[kw0_lane * 16 +: 16] = kw0_data; km[kw0_addr] = kt; end
    if (kw1_en) begin kt = km[kw1_addr]; kt[kw1_lane * 16 +: 16] = kw1_data; km[kw1_addr] = kt; end
    if (vw_en) begin kt = vm[vw_addr]; kt[vw_lane * 16 +: 16] = vw_data; vm[vw_addr] = kt; end
  end
  qwen_full dut (.clk(clk), .rst_n(rst_n), .start(start), .head_en(head_en),
    .tok(tok), .pos(pos), .w_addr(w_addr), .w_data(w_data), .c_addr(c_addr),
    .c_data(c_data), .g_addr(g_addr), .g_data(g_data), .k_raddr(k_raddr),
    .k_rdata(k_rdata), .v_raddr(v_raddr), .v_rdata(v_rdata),
    .kw0_en(kw0_en), .kw0_addr(kw0_addr), .kw0_lane(kw0_lane), .kw0_data(kw0_data),
    .kw1_en(kw1_en), .kw1_addr(kw1_addr), .kw1_lane(kw1_lane), .kw1_data(kw1_data),
    .vw_en(vw_en), .vw_addr(vw_addr), .vw_lane(vw_lane), .vw_data(vw_data),
    .next_tok(next_tok), .best(best), .done(done), .busy(busy));
  task step(input integer t, input integer p, input integer he);
    begin
      tok = t; pos = p; head_en = he;
      @(negedge clk); start = 1; t0 = cyc; @(negedge clk); start = 0;
      while (!done) @(negedge clk);
      $display("STEP pos=%%0d tok=%%0d cycles=%%0d next=%%0d best=%%0d", p, t, cyc - t0, next_tok, best);
      $fflush;
    end
  endtask
  initial begin
    fd = $fopen("weights.bin", "rb");
    wlast = {%(WA1)d{1'b1}};
    $readmemh("cparams.hex", cmem);
    $readmemh("gains.hex", gmem);
    for (i = 0; i <= %(kn)d; i = i + 1) km[i] = 0;
    for (i = 0; i <= %(vn)d; i = i + 1) vm[i] = 0;
    repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
%(steps)s
    $finish;
  end
endmodule
"""


def build(prompt, n_gen, work=WORK, log=print, layers=None, lanes=None):
    cfg, W = qr.load()
    tok = qr.Tokenizer()
    cal = qc.calibration(cfg, W, tok)
    if layers:
        # A shortened stack, for a quick check of the same sequencer: the
        # token it picks is meaningless, but it has to match the integer
        # model run with the same layers.
        cfg = dict(cfg, num_hidden_layers=layers)
    im = qi.IntQwen(cfg, W, 16, True, cal, exact_io=True)
    if lanes:
        im.ms["lanes"] = lanes      # the board's width, not the default rule
    ids = tok.encode(prompt)
    want, srcs = build_model(im, ids, n_gen, work, log, tok.decode)
    return tok, ids, want, srcs


def build_model(im, ids, n_gen, work, log=print, decode=str, layers=None,
                stage=False, want=None, table=True, tp=None):
    """The sequencer, its blocks, images and testbench for an integer
    model already built, and the tokens it has to choose. layers and
    stage build one pipeline stage of a multi-board run instead: that
    range of layers, with its own images, and the hidden-state port."""
    # The integer model's own run: what the RTL has to choose.
    t0 = time.time()
    if want is None:
        want = qr.greedy(im, ids, n_gen)
        log("integer model: %r (%.0f s)" % (decode(want), time.time() - t0))
    os.makedirs(work, exist_ok=True)
    layers = list(range(im.NL)) if layers is None else list(layers)
    lay = Layout(im, len(layers), table, tp)
    norms = write_images(im, lay, work, log, layers)
    lc, nf = layer_consts(im, norms, layers)
    rtl, w = render(im, lay, lc, nf, stage)
    with open(os.path.join(work, "qwen_full.v"), "w") as f:
        f.write(rtl)
    # The blocks, at Qwen's size.
    ms = im.ms
    rr = agent.RuleBasedAgent()
    cf.write_attn_deps(ms, work)
    cf.write_rmsnorm_deps(ms, work)
    cf.write_silu_deps(ms, work)
    cf.write_rope_deps(im.sp["rope"], work)
    blocks = {
        "b_projn.v": rr.render_projn(specgen.derive_projn_spec(ms, per_column=True),
                                     {agent.FIX_LANE}),
        "b_attnn.v": rr.render_attnn(specgen.derive_attnn_spec(ms), {agent.FIX_KLANE}),
        "b_rmsnorm.v": rr.render_rmsnorm(im.sp["rmsnorm"], {agent.FIX_EPS}),
        "b_rope.v": rr.render_rope(im.sp["rope"], {agent.FIX_ROTDIR}),
        "b_silu.v": rr.render_silu(im.sp["silu"], {agent.FIX_SIGN}),
        "b_resadd.v": rr.render_resadd(im.sp["resadd"], {agent.FIX_RRND}),
    }
    if im.qkn:
        blocks["b_rmsnorm_hd.v"] = rr.render_rmsnorm(
            im.sp["headnorm"], {agent.FIX_EPS}).replace(
                "module rmsnorm (", "module rmsnorm_hd (", 1)
    for fn, src in blocks.items():
        with open(os.path.join(work, fn), "w") as f:
            f.write(src)
    deps = sorted(set(cf.ATTN_DEPS) | set(cf.RMSNORM_DEPS) | set(cf.SILU_DEPS)
                  | {"rope_rom.v"})
    steps = []
    feed = list(ids)
    for p in range(len(want) - 1):
        # The prompt, then each generated token fed back; the head only
        # where the next token is wanted.
        he = 1 if p >= len(ids) - 1 else 0
        steps.append("    step(%d, %d, %d);" % (want[p], p, he))
    tb = TB % dict(tw=w["tw"] - 1, pw=w["pw"] - 1, WA=w["WA"] - 1,
                   WA1=w["WA"], CA=w["CA"] - 1, CW=w["CW"] - 1,
                   GA=w["GA"] - 1, KA=w["KA"] - 1, VA=w["VA"] - 1,
                   cn=lay.cwords - 1, gn=gain_words(im, len(layers)) - 1,
                   kn=w["KWN"] - 1, vn=w["VWN"] - 1, LW=16 * w["N"] - 1,
                   LB=w["LB"] - 1, WB=2 * w["N"], steps="\n".join(steps))
    with open(os.path.join(work, "tb_qfull.v"), "w") as f:
        f.write(tb)
    srcs = ["tb_qfull.v", "qwen_full.v"] + sorted(blocks) + deps
    if stage or tp:
        return want, srcs, dict(w, lay=lay, gn=gain_words(im, len(layers)))
    return want, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--no-sim", action="store_true",
                    help="write the images and RTL only, e.g. for board_zybo.py")
    ap.add_argument("--lanes", type=int, default=None,
                    help="projection lanes (16 by default; 32 for the ZC706)")
    a = ap.parse_args()
    tok, ids, want, srcs = build(a.prompt, a.tokens, a.work, layers=a.layers,
                                 lanes=a.lanes)
    if a.no_sim:
        print("wrote", a.work)
        return
    r = subprocess.run(["iverilog", "-g2005", "-o", "q.out"] + srcs, cwd=a.work,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(1)
    t0 = time.time()
    p = subprocess.Popen(["vvp", "q.out"], cwd=a.work, stdout=subprocess.PIPE,
                         text=True)
    got = []
    for line in p.stdout:
        print(line.rstrip(), "(%.0f s)" % (time.time() - t0))
        sys.stdout.flush()
        if line.startswith("STEP") and "next=" in line:
            pos_ = int(line.split("pos=")[1].split()[0])
            if pos_ >= len(ids) - 1:
                got.append(int(line.split("next=")[1].split()[0]))
    p.wait()
    gen = want[len(ids):]
    print("RTL chose:     %r" % tok.decode(got))
    print("integer model: %r" % tok.decode(gen))
    print("MATCH" if got == gen else "MISMATCH")
    sys.exit(0 if got == gen else 1)


if __name__ == "__main__":
    main()
