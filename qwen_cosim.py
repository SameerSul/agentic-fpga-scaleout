"""The real Qwen2.5-0.5B's layer-0 blocks, through the generated RTL.

qwen_int.py runs the checkpoint on the blocks' golden models. This takes
real layer-0 matrices at their full size, with the activation the
integer model actually feeds them for a real token, and runs them through
the multi-lane projection RTL in iverilog: int8 per-channel weights,
int16 activations, a bias, scale and shift per output column. Every
output has to match the integer model bit for bit.

Needs fetch_qwen.py first. Run: python3 qwen_cosim.py
"""
import json
import math
import os
import subprocess
import sys
import time

import chiplet_flow as cf
import qwen_int as qi
import qwen_real as qr
import specgen
from agent import RuleBasedAgent, FIX_LANE

ROOT = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(qr.WDIR, "calibration.json")


def calibration(cfg, W, tok):
    """Calibration is five minutes of float decode, so it is kept next to
    the weights it belongs to."""
    if os.path.exists(CAL):
        raw = json.load(open(CAL))
        return {((k.split("|")[0], int(k.split("|")[1])) if "|" in k else k): v
                for k, v in raw.items()}
    cal = qi.calibrate(cfg, W, tok)
    json.dump({("%s|%s" % k if isinstance(k, tuple) else k): v
               for k, v in cal.items()}, open(CAL, "w"))
    return cal


TB = """`timescale 1ns/1ps
module tb_qcosim;
  reg clk = 0, rst_n = 0, start = 0;
  reg [{depwm}:0] depth = {depth};
  reg [{colwm}:0] cols = {cols};
  wire [{depwm}:0] a_addr;
  wire [{wawm}:0] w_addr;
  wire [{colwm}:0] c_addr, o_index;
  wire o_valid, busy;
  wire signed [{dwm}:0] o_data;
  reg signed [{dwm}:0] a_data;
  reg [{wdm}:0] w_data;
  reg [{cwm}:0] c_data;
  reg signed [{dwm}:0] amem [0:{depth}-1];
  reg [{wdm}:0] wmem [0:{words}-1];
  reg [{cwm}:0] cmem [0:{cols}-1];
  reg signed [{dwm}:0] want [0:{cols}-1];
  integer seen = 0, bad = 0, cyc = 0, t0 = 0;
  always #5 clk = ~clk;
  always @(posedge clk) cyc = cyc + 1;
  always @(posedge clk) begin
    a_data <= amem[a_addr]; w_data <= wmem[w_addr]; c_data <= cmem[c_addr];
  end
  projn dut (.clk(clk), .rst_n(rst_n), .start(start), .depth(depth),
             .cols(cols), .scale({mw}'d0), .shift({sw}'d0), .a_addr(a_addr),
             .a_data(a_data), .w_addr(w_addr), .w_data(w_data),
             .c_addr(c_addr), .c_data(c_data), .o_valid(o_valid),
             .o_index(o_index), .o_data(o_data), .busy(busy));
  always @(posedge clk) if (rst_n && o_valid) begin
    seen = seen + 1;
    if (o_data !== want[o_index]) begin
      if (bad < 4) $display("TB_FAIL col=%0d expected=%0d got=%0d", o_index, want[o_index], o_data);
      bad = bad + 1;
    end
  end
  initial begin
    $readmemh("a.hex", amem); $readmemh("w.hex", wmem);
    $readmemh("c.hex", cmem); $readmemh("y.hex", want);
    repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
    start = 1; t0 = cyc; @(negedge clk); start = 0;
    while (busy) @(negedge clk);
    repeat (4) @(negedge clk);
    $display("TB_PROFILE span_cycles=%0d outputs=%0d", cyc - t0, seen);
    if (bad || seen != {cols}) $display("TB_RESULT: FAIL bad=%0d seen=%0d", bad, seen);
    else $display("TB_RESULT: PASS checks=%0d", seen);
    $finish;
  end
endmodule
"""


def cosim(im, spec, key, x, sx, dst, bias, work):
    """Run one real matrix through the RTL; return (ok, cycles, golden)."""
    p = spec["parameters"]
    N, dw, aw = p["lanes"], p["data_width"], p["acc_width"]
    mw, shw = p["scale_width"], p["shift_width"]
    Qw, ws = im.Q[key], im.WS[key]
    depth = len(x)
    cols = len(Qw) // depth
    golden = im.proj(x, key, sx, dst, bias)
    groups = -(-cols // N)
    mask = (1 << dw) - 1
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "a.hex"), "w") as f:
        f.write("\n".join("%x" % (v & mask) for v in x) + "\n")
    with open(os.path.join(work, "w.hex"), "w") as f:
        for g in range(groups):
            for r in range(depth):
                word = 0
                for j in range(N):
                    c = g * N + j
                    v = Qw[c * depth + r] if c < cols else 0
                    word |= (v & mask) << (dw * j)
                f.write("%x\n" % word)
    with open(os.path.join(work, "c.hex"), "w") as f:
        for r in range(cols):
            b = int(round(bias[r] / (sx * ws[r]))) if bias is not None else 0
            sc, sh = qi.pick(sx * ws[r] / dst, im.mw, im.sw)
            f.write("%x\n" % (((b & ((1 << aw) - 1)) << (shw + mw))
                               | (sh << mw) | sc))
    with open(os.path.join(work, "y.hex"), "w") as f:
        f.write("\n".join("%x" % (v & mask) for v in golden) + "\n")
    with open(os.path.join(work, "tb.v"), "w") as f:
        f.write(TB.format(depwm=p["depth_width"] - 1, colwm=p["col_width"] - 1,
                          wawm=p["word_addr_width"] - 1, dwm=dw - 1,
                          wdm=N * dw - 1, cwm=p["col_word_width"] - 1,
                          depth=depth, cols=cols, words=groups * depth,
                          mw=mw, sw=shw))
    with open(os.path.join(work, "p.v"), "w") as f:
        f.write(RuleBasedAgent().render_projn(spec, {FIX_LANE}))
    cf.write_projn_deps(im.ms, work)
    r = subprocess.run(["iverilog", "-g2005", "-o", "s.out", "tb.v", "p.v"]
                       + list(cf.PROJN_DEPS), cwd=work, capture_output=True,
                       text=True)
    if r.returncode:
        return False, 0, r.stdout + r.stderr
    out = subprocess.run(["vvp", "s.out"], cwd=work, capture_output=True,
                         text=True, timeout=3600).stdout
    cyc = int(out.split("span_cycles=")[1].split()[0]) if "span_cycles=" in out else 0
    return "TB_RESULT: PASS" in out, cyc, out


def _run(work, tb, srcs):
    with open(os.path.join(work, "tb.v"), "w") as f:
        f.write(tb)
    r = subprocess.run(["iverilog", "-g2005", "-o", "s.out", "tb.v"] + srcs,
                       cwd=work, capture_output=True, text=True)
    if r.returncode:
        return r.stdout + r.stderr
    return subprocess.run(["vvp", "s.out"], cwd=work, capture_output=True,
                          text=True, timeout=3600).stdout


def _hex(path, vals, w):
    with open(path, "w") as f:
        f.write("\n".join("%x" % (v & ((1 << w) - 1)) for v in vals) + "\n")


STREAM_TB = """`timescale 1ns/1ps
module tb_s;
  reg clk = 0, rst_n = 0, valid_in = 0;
{decl}
  integer i, got = 0, bad = 0;
  always #5 clk = ~clk;
{dut}
  always @(posedge clk) if (rst_n && valid_out) begin
{check}
    got = got + 1;
  end
  initial begin
{load}
    repeat (3) @(negedge clk); rst_n = 1;
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
{drive}
      valid_in = 1;
    end
    @(negedge clk); valid_in = 0;
    repeat (40) @(negedge clk);
    if (bad || got != {n}) $display("TB_RESULT: FAIL bad=%0d got=%0d", bad, got);
    else $display("TB_RESULT: PASS checks=%0d", got);
    $finish;
  end
endmodule
"""


def cosim_blocks(im, W, tok, work):
    """RMSNorm, RoPE, attention and SiLU on real layer-0 data."""
    import agent
    ms, s = im.ms, im.s
    rr = agent.RuleBasedAgent()
    os.makedirs(work, exist_ok=True)
    ids = tok.encode("The capital of France is")
    seen = {}
    im.probe = lambda n, v, sc, li: seen.__setitem__((n, li), (list(v), sc))
    im.reset()
    for i, t_ in enumerate(ids):
        # The probe keeps the last position's intermediates, the one whose
        # attention has a real cache behind it.
        im.step(t_, i, logits=False)
    pos = len(ids) - 1
    results = []
    A = 16
    P = "model.layers.0."

    # RMSNorm over 896: x0 -> xn, with the real input_layernorm gains.
    rn = im.sp["rmsnorm"]
    pp = rn["parameters"]
    x0, sx0 = seen[("x0", None)]
    gf = W[P + "input_layernorm.weight"][0]
    gs = max(abs(v) for v in gf) / 127.0
    gq = [int(round(v / gs)) for v in gf]
    sc, sh = qi.pick(2.0 ** (pp["norm_shift"] - pp["rsqrt_out_width"])
                     * math.sqrt(im.D) * gs / s[("xn", 0)],
                     pp["scale_width"], pp["shift_width"])
    want = specgen.rmsnorm_golden(x0, gq, 1, sc, sh, pp,
                                  rn["derivation"]["rsqrt"])[4]
    assert want == seen[("xn", 0)][0]
    cf.write_rmsnorm_deps(ms, work)
    with open(os.path.join(work, "dut.v"), "w") as f:
        f.write(rr.render_rmsnorm(rn, {agent.FIX_EPS}))
    _hex(os.path.join(work, "x.hex"), x0, A)
    _hex(os.path.join(work, "g.hex"), gq, A)
    _hex(os.path.join(work, "y.hex"), want, A)
    aw_ = pp["addr_width"]
    tb = """`timescale 1ns/1ps
module tb_n;
  reg clk = 0, rst_n = 0, start = 0;
  wire [%(am)d:0] x_addr, g_addr, o_index;
  reg signed [15:0] x_data, g_data;
  wire o_valid, busy;
  wire signed [15:0] o_data;
  reg signed [15:0] xm [0:%(d)d], gm [0:%(d)d], ym [0:%(d)d];
  integer got = 0, bad = 0;
  always #5 clk = ~clk;
  always @(posedge clk) begin x_data <= xm[x_addr]; g_data <= gm[g_addr]; end
  rmsnorm dut (.clk(clk), .rst_n(rst_n), .start(start), .eps(%(ew)d'd1),
    .scale_o(18'd%(sc)d), .shift_o(7'd%(sh)d), .x_addr(x_addr), .x_data(x_data),
    .g_addr(g_addr), .g_data(g_data), .o_valid(o_valid), .o_index(o_index),
    .o_data(o_data), .busy(busy));
  always @(posedge clk) if (rst_n && o_valid) begin
    if (o_data !== ym[o_index]) bad = bad + 1; got = got + 1; end
  initial begin
    $readmemh("x.hex", xm); $readmemh("g.hex", gm); $readmemh("y.hex", ym);
    repeat (3) @(negedge clk); rst_n = 1; @(negedge clk);
    start = 1; @(negedge clk); start = 0;
    while (busy) @(negedge clk); repeat (8) @(negedge clk);
    if (bad || got != %(n)d) $display("TB_RESULT: FAIL bad=%%0d got=%%0d", bad, got);
    else $display("TB_RESULT: PASS checks=%%0d", got);
    $finish;
  end
endmodule
""" % dict(am=aw_ - 1, d=im.D - 1, ew=pp["rsqrt_in_width"], sc=sc, sh=sh,
           n=im.D)
    out = _run(work, tb, ["dut.v"] + list(cf.RMSNORM_DEPS))
    results.append(("rmsnorm, 896 wide, real gains", "TB_RESULT: PASS" in out, out))

    # RoPE: q before rotation, all 14 heads at the last position.
    ro = im.sp["rope"]
    rp, fr = ro["parameters"], ro["derivation"]["freqs"]
    xn, sn = seen[("xn", 0)]
    qpre = im.proj(xn, P + "self_attn.q_proj.weight", sn, s[("q", 0)],
                   W[P + "self_attn.q_proj.bias"][0])
    qrot = im.rope(qpre, im.H, pos)
    assert qrot == seen[("q", 0)][0]
    hd, h2 = im.hd, im.hd // 2
    pairs = [(qpre[h * hd + i], qpre[h * hd + i + h2], i)
             for h in range(im.H) for i in range(h2)]
    exp_ = [specgen.rope_golden(a, b, i, pos, rp, fr) for a, b, i in pairs]
    cf.write_rope_deps(ro, work)
    with open(os.path.join(work, "dut.v"), "w") as f:
        f.write(rr.render_rope(ro, {agent.FIX_ROTDIR}))
    _hex(os.path.join(work, "a.hex"), [p_[0] for p_ in pairs], A)
    _hex(os.path.join(work, "b.hex"), [p_[1] for p_ in pairs], A)
    _hex(os.path.join(work, "i.hex"), [p_[2] for p_ in pairs], 8)
    _hex(os.path.join(work, "y1.hex"), [e[0] for e in exp_], A)
    _hex(os.path.join(work, "y2.hex"), [e[1] for e in exp_], A)
    np_ = len(pairs)
    tb = STREAM_TB.format(
        n=np_,
        decl=("  reg signed [15:0] x1 = 0, x2 = 0;\n  reg [%d:0] idx = 0;\n"
              "  wire signed [15:0] y1, y2;\n  wire valid_out;\n"
              "  reg signed [15:0] am [0:%d], bm [0:%d], e1 [0:%d], e2 [0:%d];\n"
              "  reg [7:0] im_ [0:%d];" % (rp["index_width"] - 1, np_ - 1,
                                           np_ - 1, np_ - 1, np_ - 1, np_ - 1)),
        dut=("  rope dut (.clk(clk), .rst_n(rst_n), .x1(x1), .x2(x2), .idx(idx),"
             " .pos(%d'd%d), .valid_in(valid_in), .y1(y1), .y2(y2),"
             " .valid_out(valid_out));" % (rp["pos_width"], pos)),
        check="    if (y1 !== e1[got] || y2 !== e2[got]) bad = bad + 1;",
        load=('    $readmemh("a.hex", am); $readmemh("b.hex", bm);\n'
              '    $readmemh("i.hex", im_); $readmemh("y1.hex", e1);\n'
              '    $readmemh("y2.hex", e2);'),
        drive="      x1 = am[i]; x2 = bm[i]; idx = im_[i];")
    out = _run(work, tb, ["dut.v", "rope_rom.v"])
    results.append(("rope, 14 heads at position %d" % pos,
                    "TB_RESULT: PASS" in out, out))

    # Attention: head 0 over the real cache of the prompt.
    at = specgen.derive_attnn_spec(ms)
    ap = at["parameters"]
    La, n = ap["lanes"], pos + 1
    qh = qrot[0:hd]
    K = [k_[0:hd] for k_ in im.K[0]]
    V = [v_[0:hd] for v_ in im.Vc[0]]
    sca, sha = qi.pick(2.0 ** -ap["weight_frac"] * s[("v", 0)] / s[("ctx", 0)],
                       ap["scale_width"], ap["shift_width"])
    want = specgen.attn_golden(qh, K, V, n, im.shs[0], sca, sha, ap,
                               im.sp["softmax"]["parameters"])[4]
    assert want == seen[("ctx", 0)][0][0:hd]
    cf.write_attn_deps(ms, work)
    with open(os.path.join(work, "dut.v"), "w") as f:
        f.write(rr.render_attnn(at, {agent.FIX_KLANE}))
    m16 = (1 << A) - 1
    kwords, vwords = [], []
    for g in range(-(-ap["capacity"] // La)):
        for d in range(hd):
            w_ = 0
            for l in range(La):
                j = g * La + l
                w_ |= ((K[j][d] if j < n else 0) & m16) << (A * l)
            kwords.append(w_)
    for j in range(ap["capacity"]):
        for dg in range(hd // La):
            w_ = 0
            for l in range(La):
                w_ |= ((V[j][dg * La + l] if j < n else 0) & m16) << (A * l)
            vwords.append(w_)
    with open(os.path.join(work, "k.hex"), "w") as f:
        f.write("\n".join("%x" % w_ for w_ in kwords) + "\n")
    with open(os.path.join(work, "v.hex"), "w") as f:
        f.write("\n".join("%x" % w_ for w_ in vwords) + "\n")
    _hex(os.path.join(work, "q.hex"), qh, A)
    _hex(os.path.join(work, "y.hex"), want, A)
    tb = """`timescale 1ns/1ps
module tb_a;
  reg clk = 0, rst_n = 0, start = 0, load_valid = 0;
  reg signed [15:0] load_data = 0;
  wire [%(kaw)d:0] k_addr;
  wire [%(vaw)d:0] v_addr;
  reg [%(wd)d:0] k_data, v_data;
  wire o_valid, busy;
  wire [%(hw)d:0] o_index;
  wire signed [15:0] o_data;
  reg [%(wd)d:0] km [0:%(kn)d], vm [0:%(vn)d];
  reg signed [15:0] qm [0:%(hd)d], ym [0:%(hd)d];
  integer i, got = 0, bad = 0;
  always #5 clk = ~clk;
  always @(posedge clk) begin k_data <= km[k_addr]; v_data <= vm[v_addr]; end
  attnn dut (.clk(clk), .rst_n(rst_n), .load_valid(load_valid),
    .load_data(load_data), .start(start), .n(%(nw)d'd%(n)d), .shift_s(5'd%(shs)d),
    .scale_o(18'd%(sc)d), .shift_o(7'd%(sh)d), .k_addr(k_addr), .k_data(k_data),
    .v_addr(v_addr), .v_data(v_data), .o_valid(o_valid), .o_index(o_index),
    .o_data(o_data), .busy(busy));
  always @(posedge clk) if (rst_n && o_valid) begin
    if (o_data !== ym[o_index]) bad = bad + 1; got = got + 1; end
  initial begin
    $readmemh("k.hex", km); $readmemh("v.hex", vm);
    $readmemh("q.hex", qm); $readmemh("y.hex", ym);
    repeat (3) @(negedge clk); rst_n = 1;
    for (i = 0; i <= %(hd)d; i = i + 1) begin
      @(negedge clk); load_data = qm[i]; load_valid = 1; end
    @(negedge clk); load_valid = 0;
    @(negedge clk); start = 1; @(negedge clk); start = 0;
    while (busy) @(negedge clk); repeat (8) @(negedge clk);
    if (bad || got != %(hd1)d) $display("TB_RESULT: FAIL bad=%%0d got=%%0d", bad, got);
    else $display("TB_RESULT: PASS checks=%%0d", got);
    $finish;
  end
endmodule
""" % dict(kaw=ap["k_addr_width"] - 1, vaw=ap["v_addr_width"] - 1,
           wd=La * A - 1, hw=ap["head_dim_width"] - 1, kn=len(kwords) - 1,
           vn=len(vwords) - 1, hd=hd - 1, hd1=hd, nw=ap["n_width"], n=n,
           shs=im.shs[0], sc=sca, sh=sha)
    out = _run(work, tb, ["dut.v"] + list(cf.ATTN_DEPS))
    results.append(("attention head 0 over %d cached positions" % n,
                    "TB_RESULT: PASS" in out, out))

    # SiLU on the real 4864-wide gate.
    sl = im.sp["silu"]
    g, _ = seen[("g", 0)]
    e = im.gsh[0]
    xs = [c << e if e >= 0 else c >> -e for c in g]
    ys = [specgen.silu_golden(x_, sl["derivation"]) for x_ in xs]
    cf.write_silu_deps(ms, work)
    with open(os.path.join(work, "dut.v"), "w") as f:
        f.write(rr.render_silu(sl, {agent.FIX_SIGN}))
    iw = sl["parameters"]["width"]
    _hex(os.path.join(work, "x.hex"), xs, iw)
    _hex(os.path.join(work, "y.hex"), ys, iw)
    nx = len(xs)
    tb = STREAM_TB.format(
        n=nx,
        decl=("  reg signed [%d:0] x = 0;\n  wire signed [%d:0] y;\n"
              "  wire valid_out;\n  reg signed [%d:0] xm [0:%d], ym [0:%d];"
              % (iw - 1, iw - 1, iw - 1, nx - 1, nx - 1)),
        dut=("  silu dut (.clk(clk), .rst_n(rst_n), .x(x), .valid_in(valid_in),"
             " .y(y), .valid_out(valid_out));"),
        check="    if (y !== ym[got]) bad = bad + 1;",
        load='    $readmemh("x.hex", xm); $readmemh("y.hex", ym);',
        drive="      x = xm[i];")
    out = _run(work, tb, ["dut.v"] + list(cf.SILU_DEPS))
    results.append(("silu on the 4864-wide gate", "TB_RESULT: PASS" in out, out))

    # The residual add after attention: x0 + a -> x1, at their own scales.
    rp = im.sp["resadd"]["parameters"]
    x0, sx0 = seen[("x0", None)]
    a_, sa = seen[("a", 0)]
    x1, sx1 = seen[("x1", 0)]
    best = None
    for sh in range(1, 60):
        ka, kb = int(round(sx0 / sx1 * (1 << sh))), int(round(sa / sx1 * (1 << sh)))
        if max(ka, kb) >= 1 << rp["scale_width"]:
            break
        best = (ka, kb, sh)
    assert [specgen.resadd_golden(u, v, best[0], best[1], best[2], A)
            for u, v in zip(x0, a_)] == x1
    with open(os.path.join(work, "dut.v"), "w") as f:
        f.write(rr.render_resadd(im.sp["resadd"], {agent.FIX_RRND}))
    _hex(os.path.join(work, "a.hex"), x0, A)
    _hex(os.path.join(work, "b.hex"), a_, A)
    _hex(os.path.join(work, "y.hex"), x1, A)
    nx = len(x0)
    tb = STREAM_TB.format(
        n=nx,
        decl=("  reg signed [15:0] a = 0, b = 0;\n  wire signed [15:0] y;\n"
              "  wire valid_out;\n  reg signed [15:0] am [0:%d], bm [0:%d], ym [0:%d];"
              % (nx - 1, nx - 1, nx - 1)),
        dut=("  resadd dut (.clk(clk), .rst_n(rst_n), .a(a), .b(b),"
             " .scale_a(%d'd%d), .scale_b(%d'd%d), .shift(%d'd%d),"
             " .valid_in(valid_in), .y(y), .valid_out(valid_out));"
             % (rp["scale_width"], best[0], rp["scale_width"], best[1],
                rp["shift_width"], best[2])),
        check="    if (y !== ym[got]) bad = bad + 1;",
        load=('    $readmemh("a.hex", am); $readmemh("b.hex", bm);\n'
              '    $readmemh("y.hex", ym);'),
        drive="      a = am[i]; b = bm[i];")
    out = _run(work, tb, ["dut.v"])
    results.append(("residual add after attention", "TB_RESULT: PASS" in out, out))
    return results


def main():
    cfg, W = qr.load()
    tok = qr.Tokenizer()
    cal = calibration(cfg, W, tok)
    cfg1 = dict(cfg, num_hidden_layers=1)
    im = qi.IntQwen(cfg1, W, 16, True, cal)
    spec = specgen.derive_projn_spec(im.ms, per_column=True)
    print("projection: %d lanes of %d-bit, %d-bit accumulator"
          % (spec["parameters"]["lanes"], spec["parameters"]["data_width"],
             spec["parameters"]["acc_width"]))
    # The activations the integer model feeds layer 0 for a real token.
    seen = {}
    im.probe = lambda n, v, s, li: seen.setdefault((n, li), (list(v), s))
    im.step(tok.encode("Paris")[0], 0, logits=False)
    s = im.s
    P = "model.layers.0."
    xn, sn = seen[("xn", 0)]
    m, sm = seen[("m", 0)]
    cases = [
        ("q_proj, 896 to 896, with bias", P + "self_attn.q_proj.weight", xn,
         sn, s[("q", 0)], W[P + "self_attn.q_proj.bias"][0]),
        ("down_proj, 4864 to 896", P + "mlp.down_proj.weight", m, sm,
         s[("dn", 0)], None),
    ]
    ok_all = True
    for label, key, x, sx, dst, bias in cases:
        t0 = time.time()
        ok, cyc, out = cosim(im, spec, key, x, sx, dst, bias,
                             os.path.join(ROOT, "build_qcosim"))
        ok_all &= ok
        print("%-32s %s  %d cycles  (%.0f s)" % (label, "bit-exact" if ok
                                                 else "MISMATCH", cyc,
                                                 time.time() - t0))
        if not ok:
            print(out[-800:])
    for label, ok, out in cosim_blocks(im, W, tok,
                                       os.path.join(ROOT, "build_qcosim_b")):
        ok_all &= ok
        print("%-32s %s" % (label, "bit-exact" if ok else "MISMATCH"))
        if not ok:
            print(out[-800:])
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
