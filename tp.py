"""Boards splitting the weights: tensor parallelism, every board on every layer.

The layer split (gals.py) fits a model larger than any one board, but a
single token still passes through every board's layers in turn, so it goes
no faster. Decoding reads every weight once a token, and the time that
takes is set by how fast those bytes arrive; this split divides the bytes.
Each of T ranks holds a slice of every matrix, split by output column:
its own attention heads (q, k and v rows, and its share of the KV cache),
its share of the MLP's gate and up rows, and its share of o's and down's
output columns over their full depth. Every value a rank computes is one
a single board computes, whole and requantized, so the ranks only have to
gather each other's slices, four times a layer:

  the attention context   each rank's heads, before o
  o's output              each rank's columns, before the residual add
  the gated product       each rank's share of the MLP width, before down
  down's output           each rank's columns, before the residual add

and once for the head, whose vocabulary the ranks also split, to agree on
the token with the largest logit (the lowest index on a tie, as the
integer model's argmax). The norms, RoPE, the residual adds and the
embedding run on every rank, on identical vectors. A rank waits in a
gather state while the network writes the others' slices into its arrays;
a rank can be at most one gather ahead of another, and the slice it sends
early only ever lands in a region the receiver is not using.

Here the network is the testbench's: when every rank has asked for the
same gather it reads each rank's slice through that rank's gather port
and writes it into the others', as a board's ARM does through its
registers (board_zybo.py). Each rank runs
on its own clock (10, 7.9 and 12.3 ns by default) and at its own width, so
the ranks can be different boards. The tokens and logits have to be the
one-board integer model's.

Run: python3 tp.py [--ranks 2] [--style qwen3]
"""
import argparse
import os
import shutil
import subprocess
import time

import qwen_full
import qwen_real as qr

ROOT = os.path.dirname(os.path.abspath(__file__))


def _clog2(n):
    return max(1, (n - 1).bit_length())


def _rank_block(p, w, lay, gn, half):
    """One rank's clock, core and memories, as gals.py's stages have."""
    return """  // ---- rank {p}: its own clock and memories
  reg {p}_clk = 0;
  always #{half} {p}_clk = ~{p}_clk;
  reg {p}_rst_n = 0;
  wire [{WA}:0] {p}_w_addr; reg [{LW}:0] {p}_w_data, {p}_wtmp;
  wire [{CA}:0] {p}_c_addr; reg [{CW}:0] {p}_c_data;
  wire [{GA}:0] {p}_g_addr; reg signed [15:0] {p}_g_data;
  wire [{KA}:0] {p}_k_raddr, {p}_kw0_addr, {p}_kw1_addr;
  wire [{VA}:0] {p}_v_raddr, {p}_vw_addr;
  reg [{LW}:0] {p}_k_rdata, {p}_v_rdata, {p}_kt;
  wire {p}_kw0_en, {p}_kw1_en, {p}_vw_en;
  wire [{LB}:0] {p}_kw0_lane, {p}_kw1_lane, {p}_vw_lane;
  wire signed [15:0] {p}_kw0_data, {p}_kw1_data, {p}_vw_data;
  reg [{CW}:0] {p}_cmem [0:{cn}];
  reg signed [15:0] {p}_gmem [0:{gn}];
  reg [{LW}:0] {p}_km [0:{kn}];
  reg [{LW}:0] {p}_vm [0:{vn}];
  integer {p}_fd, {p}_r, {p}_i;
  reg [{WA}:0] {p}_wlast;
  always @(posedge {p}_clk) begin
    if ({p}_w_addr !== {p}_wlast) begin
      if ({p}_w_addr !== {p}_wlast + 1) {p}_r = $fseek({p}_fd, {p}_w_addr * {WB}, 0);
      {p}_r = $fread({p}_wtmp, {p}_fd);
      {p}_wlast = {p}_w_addr;
    end
    {p}_w_data <= {p}_wtmp;
    {p}_c_data <= {p}_cmem[{p}_c_addr];
    {p}_g_data <= {p}_gmem[{p}_g_addr];
    {p}_k_rdata <= {p}_km[{p}_k_raddr];
    {p}_v_rdata <= {p}_vm[{p}_v_raddr];
    if ({p}_kw0_en) begin {p}_kt = {p}_km[{p}_kw0_addr]; {p}_kt[{p}_kw0_lane * 16 +: 16] = {p}_kw0_data; {p}_km[{p}_kw0_addr] = {p}_kt; end
    if ({p}_kw1_en) begin {p}_kt = {p}_km[{p}_kw1_addr]; {p}_kt[{p}_kw1_lane * 16 +: 16] = {p}_kw1_data; {p}_km[{p}_kw1_addr] = {p}_kt; end
    if ({p}_vw_en) begin {p}_kt = {p}_vm[{p}_vw_addr]; {p}_kt[{p}_vw_lane * 16 +: 16] = {p}_vw_data; {p}_vm[{p}_vw_addr] = {p}_kt; end
  end
  reg {p}_start = 0, {p}_head_en = 0, {p}_gd = 0;
  reg [{twm}:0] {p}_tok = 0;
  reg [{pwm}:0] {p}_pos = 0;
  wire [{twm}:0] {p}_next_tok;
  wire signed [15:0] {p}_best;
  wire {p}_done, {p}_busy, {p}_g_req;
  wire [2:0] {p}_g_vec;
  reg {p}_gx_we = 0;
  reg [{GX}:0] {p}_gx_addr = 0;
  reg signed [15:0] {p}_gx_wdata = 0;
  wire signed [15:0] {p}_gx_rdata;
  integer {p}_busyc = 0, {p}_compc = 0;
  always @(posedge {p}_clk) if ({p}_busy) {p}_busyc = {p}_busyc + 1;
  // Cycles computed, not spent waiting on a gather.
  always @(posedge {p}_clk) if ({p}_busy && !{p}_g_req) {p}_compc = {p}_compc + 1;
  // The network's answer to this rank's gather stays up until it drops
  // the request; a rank already past it waits for the next one.
  always @(negedge {p}_g_req) {p}_gd = 0;
  qwen_full_{p} {p}_core (.clk({p}_clk), .rst_n({p}_rst_n), .start({p}_start),
    .head_en({p}_head_en), .tok({p}_tok), .pos({p}_pos), .w_addr({p}_w_addr),
    .w_data({p}_w_data), .c_addr({p}_c_addr), .c_data({p}_c_data), .g_addr({p}_g_addr),
    .g_data({p}_g_data), .k_raddr({p}_k_raddr), .k_rdata({p}_k_rdata),
    .v_raddr({p}_v_raddr), .v_rdata({p}_v_rdata),
    .kw0_en({p}_kw0_en), .kw0_addr({p}_kw0_addr), .kw0_lane({p}_kw0_lane), .kw0_data({p}_kw0_data),
    .kw1_en({p}_kw1_en), .kw1_addr({p}_kw1_addr), .kw1_lane({p}_kw1_lane), .kw1_data({p}_kw1_data),
    .vw_en({p}_vw_en), .vw_addr({p}_vw_addr), .vw_lane({p}_vw_lane), .vw_data({p}_vw_data),
    .g_req({p}_g_req), .g_vec({p}_g_vec), .g_done({p}_gd),
    .gx_we({p}_gx_we), .gx_addr({p}_gx_addr), .gx_wdata({p}_gx_wdata), .gx_rdata({p}_gx_rdata),
    .next_tok({p}_next_tok), .best({p}_best), .done({p}_done), .busy({p}_busy));
  initial begin
    {p}_fd = $fopen("{p}/weights.bin", "rb");
    {p}_wlast = {{{WA1}{{1'b1}}}};
    $readmemh("{p}/cparams.hex", {p}_cmem);
    $readmemh("{p}/gains.hex", {p}_gmem);
    for ({p}_i = 0; {p}_i <= {kn}; {p}_i = {p}_i + 1) {p}_km[{p}_i] = 0;
    for ({p}_i = 0; {p}_i <= {vn}; {p}_i = {p}_i + 1) {p}_vm[{p}_i] = 0;
    repeat (4) @(negedge {p}_clk); {p}_rst_n = 1;
  end
""".format(p=p, half=half, WA=w["WA"] - 1, WA1=w["WA"], CA=w["CA"] - 1,
           CW=w["CW"] - 1, GA=w["GA"] - 1, KA=w["KA"] - 1, VA=w["VA"] - 1,
           cn=lay.cwords - 1, gn=gn - 1, kn=w["KWN"] - 1, vn=w["VWN"] - 1,
           twm=w["tw"] - 1, pwm=w["pw"] - 1, LW=16 * w["N"] - 1, LB=w["LB"] - 1,
           WB=2 * w["N"], GX=w["GX"] - 1)


def render_tb(ranks, im, seq, n_prompt):
    """ranks: (name, w, lay, gn, half period). The host starts every rank
    on each position and prints the token the ranks agreed on; the network
    serves each gather once every rank has asked for it, through the
    ranks' gather ports only."""
    T = len(ranks)
    H, KV, hd, D, F = im.H, im.KV, im.hd, im.D, im.F
    Hl, Dl, Fl = H // T, D // T, F // T
    P = [r[0] for r in ranks]
    blocks = "".join(_rank_block(*r) for r in ranks)
    allreq = " && ".join("%s_g_req" % p for p in P)
    nogd = " && ".join("!%s_gd" % p for p in P)
    # Each slice is read from its rank's gather port a word a clock, on
    # that rank's clock, and written into every other rank's on theirs.
    copies = []
    for vec, n, off in ((1, Hl * hd, lambda s: s * Hl * hd), (2, Dl, lambda s: s * Dl),
                        (3, Fl, lambda s: s * Fl)):
        body = []
        for s, ps in enumerate(P):
            body.append("        for (i = %d; i < %d; i = i + 1) begin" % (off(s), off(s) + n))
            body.append("          %s_gx_addr = i; @(posedge %s_clk); #1 gv = %s_gx_rdata;" % (ps, ps, ps))
            for d, pd in enumerate(P):
                if s != d:
                    body.append("          %s_gx_addr = i; %s_gx_wdata = gv; %s_gx_we = 1;"
                                " @(posedge %s_clk); #1 %s_gx_we = 0;" % (pd, pd, pd, pd, pd))
            body.append("        end")
        copies.append("      %d: begin\n%s\n      end" % (vec, "\n".join(body)))
    # The head: the largest logit over the ranks' runs of the vocabulary,
    # which ascend, so a tie keeps the lower index.
    pick = ["        gbest = %s_best; gtok = %s_next_tok;" % (P[0], P[0])]
    for p in P[1:]:
        pick.append("        if (%s_best > gbest) begin gbest = %s_best; gtok = %s_next_tok; end"
                    % (p, p, p))
    copies.append("      4: begin\n%s\n      end" % "\n".join(pick))
    steps = "\n".join("    host_step(%d, %d, %d);" % (seq[k], k, int(k >= n_prompt - 1))
                      for k in range(len(seq) - 1))
    starts = "\n".join("""      begin
        {p}_tok = t; {p}_pos = p; {p}_head_en = he;
        @(negedge {p}_clk); {p}_start = 1;
        @(negedge {p}_clk); while (!{p}_busy) @(negedge {p}_clk); {p}_start = 0;
        @(posedge {p}_done);
      end""".format(p=p) for p in P)
    tw = ranks[0][1]["tw"]
    return """`timescale 1ns/1ps
module tb_tp;
{blocks}
  // ---- the network: every gather served once every rank asks for it
  integer i, gathers = 0, bad = 0;
  reg signed [15:0] gbest, gv;
  reg [{twm}:0] gtok;
  initial forever begin
    wait (({allreq}) && ({nogd}));
    #1;
    if ({same}) ; else begin bad = bad + 1; $display("TP_FAIL ranks asked for different gathers"); end
    case (a_g_vec)
{copies}
      default: ;
    endcase
    gathers = gathers + 1;
{setgd}
  end
  // ---- the host: every rank starts each position, rank a answers
  task host_step(input integer t, input integer p, input integer he);
    begin
      fork
{starts}
      join
      if (he) $display("TOKEN pos=%0d tok=%0d best=%0d", p + 1, gtok, gbest);
      $fflush;
    end
  endtask
  initial begin
    #2000;
{steps}
    $display("TP gathers=%0d bad=%0d", gathers, bad);
{busy}
    $finish;
  end
endmodule
""".format(blocks=blocks, allreq=allreq, nogd=nogd, twm=tw - 1,
           same=" && ".join("%s_g_vec == a_g_vec" % p for p in P[1:]) or "1",
           copies="\n".join(copies), setgd="\n".join("    %s_gd = 1;" % p for p in P),
           starts=starts, steps=steps,
           busy="\n".join('    $display("RANK %s busy_cycles=%%0d compute_cycles=%%0d", %s_busyc, %s_compc);'
                           % (p, p, p) for p in P))


def build(im, ids, n_gen, work, ranks=2, clocks=(10.0, 7.9, 12.3), lanes=None,
          log=print):
    """ranks tensor-parallel ranks over the whole model; lanes[i], if
    given, is rank i's width. Returns the integer model's tokens and the
    testbench sources."""
    T = ranks
    assert im.H % T == 0 and im.KV % T == 0 and im.F % T == 0 and im.D % T == 0, \
        "the heads, KV heads, d_ff and d_model must split %d ways" % T
    want = qr.greedy(im, ids, n_gen)
    os.makedirs(work, exist_ok=True)
    own = im.ms.get("lanes")
    rks, srcs = [], set()
    for r in range(T):
        p = chr(ord("a") + r)
        d = os.path.join(work, p)
        if lanes:
            im.ms["lanes"] = lanes[r]
        _, s, w = qwen_full.build_model(im, ids, n_gen, d, log=lambda *a: None,
                                        want=want, tp=(r, T))
        rtl = open(os.path.join(d, "qwen_full.v")).read().replace(
            "module qwen_full (", "module qwen_full_%s (" % p, 1).replace(
            "  projn u_proj (", "  projn_%s u_proj (" % p, 1).replace(
            "  attnn u_attn (", "  attnn_%s u_attn (" % p, 1)
        with open(os.path.join(work, "qwen_full_%s.v" % p), "w") as f:
            f.write(rtl)
        for f in s:
            if f in ("b_projn.v", "b_attnn.v"):
                m = f[2:-2]
                with open(os.path.join(work, "b_%s_%s.v" % (m, p)), "w") as g:
                    g.write(open(os.path.join(d, f)).read().replace(
                        "module %s (" % m, "module %s_%s (" % (m, p), 1))
                srcs.add("b_%s_%s.v" % (m, p))
            elif f not in ("tb_qfull.v", "qwen_full.v"):
                shutil.copyfile(os.path.join(d, f), os.path.join(work, f))
                srcs.add(f)
        rks.append((p, w, w["lay"], w["gn"], clocks[r % len(clocks)] / 2))
    im.ms["lanes"] = own
    with open(os.path.join(work, "tb_tp.v"), "w") as f:
        f.write(render_tb(rks, im, want, len(ids)))
    return want, (["tb_tp.v"] + ["qwen_full_%s.v" % r[0] for r in rks] + sorted(srcs))


ARM = """  // ---- the ARM's side of each gather, as the rank's program does it:
  // read this rank's slice through GADDR and GDATA, post it, and once the
  // network has brought the others', write them in and let the core go.
  localparam RANK = {r}, RANKS = {T};
  reg signed [15:0] vbuf [0:{vm}];
  reg post = 0, ack = 0, inpos = 0;
  reg [2:0] pvec = 0;
  integer pn, k, s_, gathers = 0;
  reg [{twm}:0] ltok = 0, gtok = 0;
  reg signed [15:0] lbest = 0, gbest = 0;
  task run_pos(input integer tk, input integer p, input integer he);
    reg fin;
    begin
      inpos = 1;
      wr(7'h04, tk); wr(7'h08, p);
      wr(7'h00, 1 | (he << 1));
      fin = 0;
      while (!fin) begin
        rd(7'h44);
        if (rv[0]) begin
          pvec = rv[3:1];
          pn = pvec == 1 ? {n1} : pvec == 2 ? {n2} : {n3};
          if (pvec == 4) begin
            rd(7'h10); ltok = rv; rd(7'h14); lbest = rv[15:0];
          end else begin
            wr(7'h38, RANK * pn);
            for (k = 0; k < pn; k = k + 1) begin rd(7'h3c); vbuf[RANK * pn + k] = rv[15:0]; end
          end
          post = 1; wait (ack); post = 0; wait (!ack);
          if (pvec != 4)
            for (s_ = 0; s_ < RANKS; s_ = s_ + 1)
              if (s_ != RANK) begin
                wr(7'h38, s_ * pn);
                for (k = 0; k < pn; k = k + 1) wr(7'h3c, vbuf[s_ * pn + k]);
              end
          wr(7'h44, 1);
          gathers = gathers + 1;
        end else begin
          rd(7'h0c); fin = rv[1];
        end
      end
      inpos = 0;
    end
  endtask
"""


def _rename(src, p, mods):
    for m in mods:
        src = src.replace("module %s (" % m, "module %s_%s (" % (m, p), 1)
        src = src.replace("  %s " % m, "  %s_%s " % (m, p))
    return src


def build_boards(im, ids, n_gen, work, ranks=2, clocks=(10.0, 7.9, 12.3), lanes=None,
                 lat=30, jit=1, log=print):
    """The same ranks as build, each as its board runs it: the package's
    register block and DDR bridge around the core, a DDR model of its own
    (from the rank's weights8.bin and cparams.hex) that stalls at random
    with jit, its own clock, and the ARM's side of every gather done
    through the registers. Only the network between the ARMs is the
    testbench's (the ARM program's own is board_zybo.TP_C)."""
    import board_zybo
    import zybo
    T = ranks
    want, srcs = build(im, ids, n_gen, work, ranks, clocks, lanes, log)
    H, hd, D, F = im.H, im.hd, im.D, im.F
    n1, n2, n3 = H * hd // T, D // T, F // T
    P = [chr(ord("a") + r) for r in range(T)]
    tops, mods = [], []
    for r, p in enumerate(P):
        d = os.path.join(work, p)
        zybo.write_w8(d)
        L = zybo.layout(d, board_zybo.BASE)
        L["steps"] = []
        L["tp"] = dict(rank=r, ranks=T)
        top = zybo.render_top(L["W"], L["gn"], L["cb"], L["kb"], L["vb"], 4, L["wb"])
        top = _rename(top, p, ["qwen_zybo"]).replace(
            "  qwen_full core (", "  qwen_full_%s core (" % p, 1).replace(
            '$readmemh("gains.hex"', '$readmemh("%s/gains.hex"' % p, 1)
        wrap = _rename(board_zybo.render_wrapper(L), p, ["fpgai_zybo"]).replace(
            "  qwen_zybo br (", "  qwen_zybo_%s br (" % p, 1)
        tb = board_zybo.tb_text(L, 4, lat, jit)
        for x, y in (("module tb_fpgai_zybo;", "module rank_%s;\n  reg up = 0;" % p),
                     ("  always #5 clk = ~clk;", "  always #%g clk = ~clk;" % (clocks[r % len(clocks)] / 2)),
                     ("  fpgai_zybo dut (", "  fpgai_zybo_%s dut (" % p),
                     ('$fopen("weights8.bin"', '$fopen("%s/weights8.bin"' % p),
                     ('$readmemh("cparams.hex"', '$readmemh("%s/cparams.hex"' % p),
                     ("reg [5:0] s_awaddr = 0, s_araddr = 0;", "reg [6:0] s_awaddr = 0, s_araddr = 0;")):
            assert tb.count(x) == 1, x
            tb = tb.replace(x, y)
        tb = tb.replace("input [5:0] a", "input [6:0] a")
        tail = "\n    $finish;\n  end\nendmodule\n"
        assert tb.endswith(tail)
        tb = tb[:-len(tail)] + "\n    up = 1;\n  end\n" + ARM.format(
            r=r, T=T, vm=max(H * hd, D, F) - 1, twm=L["W"]["tok"] - 1,
            n1=n1, n2=n2, n3=n3) + "endmodule\n"
        for name, text in (("qwen_zybo_%s.v" % p, top), ("fpgai_zybo_%s.v" % p, wrap),
                           ("rank_%s.v" % p, tb)):
            with open(os.path.join(work, name), "w") as f:
                f.write(text)
            tops.append(name)
    # The network between the ARMs, and the host.
    allp = " && ".join("%s.post" % p for p in P)
    nop = " && ".join("!%s.post" % p for p in P)
    copies = []
    for sr, ps in enumerate(P):
        for dr, pd in enumerate(P):
            if sr != dr:
                copies.append("        for (i = 0; i < a.pn; i = i + 1) %s.vbuf[%d * a.pn + i] = %s.vbuf[%d * a.pn + i];"
                              % (pd, sr, ps, sr))
    pick = ["        gbest = a.lbest; gtok = a.ltok;"]
    pick += ["        if (%s.lbest > gbest) begin gbest = %s.lbest; gtok = %s.ltok; end" % (p, p, p)
             for p in P[1:]]
    steps = "\n".join("    host_step(%d, %d, %d);" % (want[k], k, int(k >= len(ids) - 1))
                      for k in range(len(want) - 1))
    tw = im.V.bit_length()
    tb = """`timescale 1ns/1ps
module tb_tpboards;
{inst}
  // ---- the network: each gather once every rank's ARM has posted it
  integer i, gathers = 0;
  reg signed [15:0] gbest;
  reg [{twm}:0] gtok;
  initial forever begin
    wait ({allp});
    #1;
    // A rank that serves one gather twice posts out of step: stop there.
    if ({same}) ; else begin
      $display("TP_FAIL ranks posted different gathers: {vfmt}", {vecs});
      $display("TP gathers=%0d bad=1", gathers);
      $finish;
    end
    if (a.pvec == 4) begin
{pick}
    end else begin
{copies}
    end
    gathers = gathers + 1;
{acks}
    wait ({nop});
{unacks}
  end
  // A gather posted by a rank whose peers are done with the position, or
  // a rank done while a peer waits on it: the same gather served twice.
{guards}
  task host_step(input integer t, input integer p, input integer he);
    begin
      fork
{starts}
      join
      if (he) $display("TOKEN pos=%0d tok=%0d best=%0d", p + 1, gtok, gbest);
      $fflush;
    end
  endtask
  initial begin
    wait ({up});
{steps}
    $display("TP gathers=%0d bad=0", gathers);
{cycles}
    $finish;
  end
endmodule
""".format(inst="\n".join("  rank_%s %s ();" % (p, p) for p in P), twm=tw - 1,
           allp=allp, nop=nop, same=" && ".join("%s.pvec == a.pvec" % p for p in P[1:]) or "1",
           vecs=", ".join("%s.pvec" % p for p in P), vfmt=" ".join(["%0d"] * T),
           guards="\n".join(
               "  always @(posedge %s.post) if (!(%s)) begin\n"
               "    $display(\"TP_FAIL rank %s posted a gather its peers were done with\");\n"
               "    $display(\"TP gathers=%%0d bad=1\", gathers); $finish;\n  end\n"
               "  always @(negedge %s.inpos) if (%s) begin\n"
               "    $display(\"TP_FAIL rank %s finished while a peer waited on it\");\n"
               "    $display(\"TP gathers=%%0d bad=1\", gathers); $finish;\n  end"
               % (p, " && ".join("%s.inpos" % q for q in P), p,
                  p, " || ".join("%s.post" % q for q in P if q != p) or "0", p)
               for p in P),
           pick="\n".join(pick), copies="\n".join(copies),
           acks="\n".join("    %s.ack = 1;" % p for p in P),
           unacks="\n".join("    %s.ack = 0;" % p for p in P),
           starts="\n".join("        %s.run_pos(t, p, he);" % p for p in P),
           up=" && ".join("%s.up" % p for p in P), steps=steps,
           cycles="\n".join('    $display("RANK %s core_cycles=%%0d clk_cycles=%%0d", %s.core_cycles, %s.cyc);'
                            % (p, p, p) for p in P))
    with open(os.path.join(work, "tb_tpboards.v"), "w") as f:
        f.write(tb)
    return want, ["tb_tpboards.v"] + tops + srcs[1:]


def run(work, srcs, timeout=3600):
    r = subprocess.run(["iverilog", "-g2005", "-DSIM", "-o", "t.out"] + srcs, cwd=work,
                       capture_output=True, text=True)
    if r.returncode:
        return r.stdout + r.stderr
    return subprocess.run(["vvp", "t.out"], cwd=work, capture_output=True,
                          text=True, timeout=timeout).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="qwen3", choices=("qwen3", "qwen2.5"))
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--work", default=os.path.join(ROOT, "build_tp"))
    a = ap.parse_args()
    import qwen_synth
    im, _ = qwen_synth.model(a.style)
    ids = [3, 77, 12, 140]
    t0 = time.time()
    shutil.rmtree(a.work, ignore_errors=True)
    want, srcs = build(im, ids, 3, a.work, a.ranks)
    out = run(a.work, srcs)
    print(out[-2500:])
    print("integer model:", want[len(ids):], "(%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
