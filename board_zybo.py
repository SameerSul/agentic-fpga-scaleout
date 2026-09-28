"""The Qwen sequencer as a board package, for any Zynq board in boards.py.

--board picks the board (zybo_z7_20, the default, or zc706); its part,
Vivado preset, LED and text come from boards.PACKAGES, and the package
goes to that entry's folder (board_zybo/, board_zc706/). Everything
else, the RTL, the registers, the DDR layout and the ARM program, is the
same on every Zynq: the PS7's HP ports and 1 GB of DDR look alike.

zybo.py put the generated core on a DDR bus shaped like the Zynq's. This
writes what the board needs around it, in board_zybo/:

  rtl/        the core and its blocks, the DDR bridge with the regions at
              their real DDR addresses, and fpgai_zybo, a top level with
              an AXI-Lite register block for the ARM and five AXI4
              masters: four for the weights, one for the constants and
              the KV cache
  build.tcl   the Vivado block design: the PS7 with the board preset, the
              weight masters on HP0-3, the fifth on ACP, the registers on
              GP0, FCLK0 at 50 MHz; writes the bitstream and the .xsa
  sw/         bare-metal C for the ARM: loads the SD files into DDR, runs
              the prompt a position at a time through the registers, and
              prints each generated token's text
  sd/         the SD card: weights8.bin, cparams.bin, vocab.bin and
              prompt.bin (not committed; 500 MB)

The register block and a whole one-layer run through it are simulated:
the testbench is zybo.py's DDR model, and it drives the design only
through AXI-Lite writes and reads, as the ARM would.

Run: python3 board_zybo.py                       (package, from build_qfull)
     python3 board_zybo.py --work build_qfull1 --out build_bz1 --sim
"""
import argparse
import os
import shutil
import struct
import sys

import boards
import zybo

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 0x08000000               # weights; below it the ARM's program
FPGAI_ID = 0xF96A0001
MHZ = 50

REGS = [  # offset, name, what
    (0x00, "CTRL", "write bit 0 to start a step; bit 1 is head_en"),
    (0x04, "TOK", "token id for the step"),
    (0x08, "POS", "position for the step"),
    (0x0C, "STATUS", "bit 0 busy, bit 1 done (cleared by a start)"),
    (0x10, "NEXT_TOK", "the head's greedy token, valid when done"),
    (0x14, "BEST", "its logit, sign-extended"),
    (0x18, "CORE_CYCLES", "core clock edges since reset"),
    (0x1C, "BUS_CYCLES", "bus clocks the last step took"),
    (0x20, "ID", "0x%08X" % FPGAI_ID),
    (0x24, "WBASE", "DDR address of weights8.bin"),
    (0x28, "CBASE", "DDR address of cparams.bin"),
    (0x2C, "KBASE", "DDR address of the K cache"),
    (0x30, "VBASE", "DDR address of the V cache"),
    (0x34, "KVEND", "end of the V cache"),
]
# A pipeline stage's package adds these (and CTRL bit 2 runs the
# embedding; clear it to start from a loaded hidden state).
STAGE_REGS = [
    (0x38, "XADDR", "hidden-state element to access next (auto-increments)"),
    (0x3C, "XDATA", "that element, 16 bits signed: write to load, read to fetch"),
    (0x40, "STAGE", "bits 7:0 first layer, 15:8 end layer, 16 embeds, 17 has the head"),
]


def axi4_master(p, write=False):
    """An AXI4 master's ports, named so Vivado infers the interface."""
    r = ["  output [31:0] {p}_araddr, output [7:0] {p}_arlen,",
         "  output [2:0] {p}_arsize, output [1:0] {p}_arburst,",
         "  output [3:0] {p}_arcache, output [2:0] {p}_arprot,",
         "  output {p}_arlock, output [3:0] {p}_arqos,",
         "  output {p}_arvalid, input {p}_arready,",
         "  input [63:0] {p}_rdata, input [1:0] {p}_rresp, input {p}_rlast,",
         "  input {p}_rvalid, output {p}_rready,"]
    if write:
        r += ["  output [31:0] {p}_awaddr, output [7:0] {p}_awlen,",
              "  output [2:0] {p}_awsize, output [1:0] {p}_awburst,",
              "  output [3:0] {p}_awcache, output [2:0] {p}_awprot,",
              "  output {p}_awlock, output [3:0] {p}_awqos,",
              "  output {p}_awvalid, input {p}_awready,",
              "  output [63:0] {p}_wdata, output [7:0] {p}_wstrb,",
              "  output {p}_wlast, output {p}_wvalid, input {p}_wready,",
              "  input [1:0] {p}_bresp, input {p}_bvalid, output {p}_bready,"]
    return "\n".join(r).format(p=p)


def ties(p, write=False):
    s = ("  assign {p}_arsize = 3'd3; assign {p}_arburst = 2'b01;\n"
         "  assign {p}_arcache = 4'b0011; assign {p}_arprot = 3'd0;\n"
         "  assign {p}_arlock = 1'b0; assign {p}_arqos = 4'd0;\n")
    if write:
        s += ("  assign {p}_awsize = 3'd3; assign {p}_awburst = 2'b01;\n"
              "  assign {p}_awcache = 4'b0011; assign {p}_awprot = 3'd0;\n"
              "  assign {p}_awlock = 1'b0; assign {p}_awqos = 4'd0;\n")
    return s.format(p=p)


def render_wrapper(L, np_=4):
    W = L["W"]
    wm = ["m_axi_w%d" % p for p in range(np_)]
    busif = ":".join(["s_axi"] + wm + ["m_axi_kv"])
    ports = "\n".join(axi4_master(m) for m in wm) + "\n" + axi4_master("m_axi_kv", True)
    conn = "".join(
        "    .w{p}_araddr({m}_araddr), .w{p}_arlen({m}_arlen[3:0]),"
        " .w{p}_arvalid({m}_arvalid),\n"
        "    .w{p}_arready({m}_arready), .w{p}_rdata({m}_rdata),"
        " .w{p}_rvalid({m}_rvalid),\n"
        "    .w{p}_rlast({m}_rlast), .w{p}_rready({m}_rready),\n"
        .format(p=p, m=m) for p, m in enumerate(wm))
    lens = "".join("  assign %s_arlen[7:4] = 4'd0;\n" % m for m in wm)
    rd = "\n".join("          4'h%X: s_axi_rdata <= %s;" % (o >> 2, e) for o, e in [
        (0x00, "{30'd0, head_en, 1'b0}"),
        (0x04, "{%d'd0, tok}" % (32 - W["tok"])),
        (0x08, "{%d'd0, pos}" % (32 - W["pos"])),
        (0x0C, "{30'd0, done_s, busy}"),
        (0x10, "{%d'd0, next_tok}" % (32 - W["tok"])),
        (0x14, "{{16{best[15]}}, best}"),
        (0x18, "core_cycles"),
        (0x1C, "bus_cycles"),
        (0x20, "32'h%08X" % FPGAI_ID),
        (0x24, "32'h%08X" % L["wb"]),
        (0x28, "32'h%08X" % L["cb"]),
        (0x2C, "32'h%08X" % L["kb"]),
        (0x30, "32'h%08X" % L["vb"]),
        (0x34, "32'h%08X" % L["end"])])
    v = """// GENERATED by board_zybo.py: do not edit by hand.
// The Qwen sequencer for a Zynq's PL. The ARM drives it through
// an AXI-Lite register block on GP0 (see board_zybo/README.md for the
// map); the weights come in on four AXI4 masters, one per HP port, and
// the constants and the KV cache on a fifth, on ACP. Every region is at
// its absolute DDR address, fixed at generation and readable here so the
// software can check the SD files were made for this bitstream.
module fpgai_zybo (
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF {busif}, ASSOCIATED_RESET aresetn" *)
  input aclk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input aresetn,
  input [5:0] s_axi_awaddr, input s_axi_awvalid, output reg s_axi_awready,
  input [31:0] s_axi_wdata, input [3:0] s_axi_wstrb, input s_axi_wvalid,
  output reg s_axi_wready, output [1:0] s_axi_bresp, output reg s_axi_bvalid,
  input s_axi_bready,
  input [5:0] s_axi_araddr, input s_axi_arvalid, output reg s_axi_arready,
  output reg [31:0] s_axi_rdata, output [1:0] s_axi_rresp,
  output reg s_axi_rvalid, input s_axi_rready,
{ports}
  output busy_led
);
{ties}{kvties}{lens}  assign m_axi_kv_arlen[7:4] = 4'd0; assign m_axi_kv_awlen[7:4] = 4'd0;
  assign s_axi_bresp = 2'b00; assign s_axi_rresp = 2'b00;

  reg start, head_en, done_s;
  reg [{twm}:0] tok; reg [{pwm}:0] pos;
  reg [31:0] bus_cycles;
  wire [{twm}:0] next_tok; wire signed [15:0] best;
  wire done, busy;
  wire [31:0] core_cycles;
  assign busy_led = busy;

  qwen_zybo br (.clk(aclk), .rst_n(aresetn), .start(start), .head_en(head_en),
    .tok(tok), .pos(pos), .next_tok(next_tok), .best(best), .done(done),
    .busy(busy),
{conn}    .araddr(m_axi_kv_araddr), .arlen(m_axi_kv_arlen[3:0]),
    .arvalid(m_axi_kv_arvalid), .arready(m_axi_kv_arready),
    .rdata(m_axi_kv_rdata), .rvalid(m_axi_kv_rvalid), .rlast(m_axi_kv_rlast),
    .rready(m_axi_kv_rready), .awaddr(m_axi_kv_awaddr),
    .awlen(m_axi_kv_awlen[3:0]), .awvalid(m_axi_kv_awvalid),
    .awready(m_axi_kv_awready), .wdata(m_axi_kv_wdata),
    .wstrb(m_axi_kv_wstrb), .wlast(m_axi_kv_wlast), .wvalid(m_axi_kv_wvalid),
    .wready(m_axi_kv_wready), .bvalid(m_axi_kv_bvalid),
    .bready(m_axi_kv_bready), .core_cycles(core_cycles));

  // ---- AXI-Lite: one write and one read at a time, both always OKAY.
  // A start is held until the core shows busy: the core's clock may be
  // stopped for a line fill, so a one-cycle pulse could fall between its
  // edges. done is its one-edge pulse, kept until the next start.
  reg [5:0] wa; reg hav_a, hav_w; reg [31:0] wd;
  always @(posedge aclk) begin
    if (!aresetn) begin
      s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_bvalid <= 1'b0;
      s_axi_arready <= 1'b0; s_axi_rvalid <= 1'b0; hav_a <= 1'b0;
      hav_w <= 1'b0; start <= 1'b0; head_en <= 1'b0; done_s <= 1'b0;
      tok <= 0; pos <= 0; bus_cycles <= 0;
    end else begin
      // Ready is one cycle, and the transfer is the edge it meets valid.
      s_axi_awready <= 1'b0; s_axi_wready <= 1'b0; s_axi_arready <= 1'b0;
      if (s_axi_awvalid && !s_axi_awready && !hav_a && !s_axi_bvalid)
        s_axi_awready <= 1'b1;
      if (s_axi_awvalid && s_axi_awready) begin wa <= s_axi_awaddr; hav_a <= 1'b1; end
      if (s_axi_wvalid && !s_axi_wready && !hav_w && !s_axi_bvalid)
        s_axi_wready <= 1'b1;
      if (s_axi_wvalid && s_axi_wready) begin wd <= s_axi_wdata; hav_w <= 1'b1; end
      if (hav_a && hav_w) begin
        hav_a <= 1'b0; hav_w <= 1'b0; s_axi_bvalid <= 1'b1;
        case (wa[5:2])
          4'h0: begin head_en <= wd[1];
                      if (wd[0]) begin start <= 1'b1; done_s <= 1'b0; bus_cycles <= 0; end end
          4'h1: tok <= wd[{twm}:0];
          4'h2: pos <= wd[{pwm}:0];
          default: ;
        endcase
      end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;
      if (start && busy) start <= 1'b0;
      if (done && !start) done_s <= 1'b1;
      if ((start || busy) && !done_s) bus_cycles <= bus_cycles + 1;
      if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid)
        s_axi_arready <= 1'b1;
      if (s_axi_arvalid && s_axi_arready) begin
        s_axi_rvalid <= 1'b1;
        case (s_axi_araddr[5:2])
{rd}
          default: s_axi_rdata <= 32'd0;
        endcase
      end
      if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 1'b0;
    end
  end
endmodule
""".format(busif=busif, ports=ports, ties="".join(ties(m) for m in wm),
           kvties=ties("m_axi_kv", True), lens=lens, conn=conn, rd=rd,
           twm=W["tok"] - 1, pwm=W["pos"] - 1)
    return stage_wrapper(v, L) if "x_addr" in W else v


def stage_wrapper(v, L):
    """The register block for one pipeline stage of a multi-board run:
    XADDR and XDATA load the hidden state the stage starts from and read
    back the one it ends with, CTRL bit 2 runs the embedding instead, and
    STAGE says which layers this bitstream holds. The core's clock is
    held while a line fills, so an XDATA access holds its AXI response
    until the core has taken the edges it needs: two to take a write,
    three to have the word read."""
    xa = L["W"]["x_addr"]
    st = L.get("stage", dict(l0=0, l1=0, emb=1, head=1))
    def rep(a, b, n=1):
        nonlocal v
        assert v.count(a) == n, a
        v = v.replace(a, b)
    rep("input [5:0] s_axi_awaddr", "input [6:0] s_axi_awaddr")
    rep("input [5:0] s_axi_araddr", "input [6:0] s_axi_araddr")
    rep("reg [5:0] wa;", "reg [6:0] wa;")
    rep("case (wa[5:2])", "case (wa[6:2])")
    rep("case (s_axi_araddr[5:2])", "case (s_axi_araddr[6:2])")
    for i in range(14):
        rep("          4'h%X: s_axi_rdata <=" % i, "          5'h%02X: s_axi_rdata <=" % i)
    rep("5'h00: s_axi_rdata <= {30'd0, head_en, 1'b0};",
        "5'h00: s_axi_rdata <= {29'd0, emb_en, head_en, 1'b0};")
    rep("""          default: s_axi_rdata <= 32'd0;""",
        """          5'h0E: s_axi_rdata <= {%d'd0, x_addr};
          5'h10: s_axi_rdata <= 32'h%08X;
          default: s_axi_rdata <= 32'd0;""" % (32 - xa, st["l0"] | st["l1"] << 8
                                                 | st["emb"] << 16 | st["head"] << 17))
    rep("""          4'h0: begin head_en <= wd[1];""", """          5'h00: begin head_en <= wd[1]; emb_en <= wd[2];""")
    rep("""          4'h1: tok <= wd[""", """          5'h01: tok <= wd[""")
    rep("""          4'h2: pos <= wd[""", """          5'h02: pos <= wd[""")
    rep("""          default: ;
        endcase
      end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;""", """          5'h0E: x_addr <= wd[%d:0];
          5'h0F: begin x_wdata <= wd[15:0]; x_we <= 1'b1; xcc <= core_cycles; xop <= 2'd1; end
          default: ;
        endcase
      end
      // The hidden-state accesses finish on the core's edges.
      if (xop == 2'd1 && core_cycles - xcc >= 2) begin
        x_we <= 1'b0; xop <= 2'd0; s_axi_bvalid <= 1'b1; x_addr <= x_addr + 1;
      end
      if (xop == 2'd2 && core_cycles - xcc >= 3) begin
        s_axi_rdata <= {{16{x_rdata[15]}}, x_rdata}; s_axi_rvalid <= 1'b1;
        xop <= 2'd0; x_addr <= x_addr + 1;
      end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;""" % (xa - 1))
    rep("""        hav_a <= 1'b0; hav_w <= 1'b0; s_axi_bvalid <= 1'b1;""",
        """        hav_a <= 1'b0; hav_w <= 1'b0; s_axi_bvalid <= (wa[6:2] != 5'h0F);""")
    rep("""        s_axi_rvalid <= 1'b1;
        case (s_axi_araddr[6:2])""", """        s_axi_rvalid <= (s_axi_araddr[6:2] != 5'h0F);
        if (s_axi_araddr[6:2] == 5'h0F) begin xop <= 2'd2; xcc <= core_cycles; end
        case (s_axi_araddr[6:2])""")
    rep("if (s_axi_awvalid && !s_axi_awready && !hav_a && !s_axi_bvalid)",
        "if (s_axi_awvalid && !s_axi_awready && !hav_a && !s_axi_bvalid && xop == 2'd0)")
    rep("if (s_axi_wvalid && !s_axi_wready && !hav_w && !s_axi_bvalid)",
        "if (s_axi_wvalid && !s_axi_wready && !hav_w && !s_axi_bvalid && xop == 2'd0)")
    rep("if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid)",
        "if (s_axi_arvalid && !s_axi_arready && !s_axi_rvalid && xop == 2'd0)")
    rep("""      tok <= 0; pos <= 0; bus_cycles <= 0;""",
        """      tok <= 0; pos <= 0; bus_cycles <= 0;
      emb_en <= 1'b1; x_we <= 1'b0; x_addr <= 0; x_wdata <= 0; xop <= 2'd0; xcc <= 0;""")
    rep("""  reg start, head_en, done_s;""", """  reg start, head_en, done_s;
  reg emb_en, x_we; reg [%d:0] x_addr; reg signed [15:0] x_wdata;
  wire signed [15:0] x_rdata; reg [1:0] xop; reg [31:0] xcc;""" % (xa - 1))
    rep("""    .tok(tok), .pos(pos), .next_tok(next_tok), .best(best), .done(done),
    .busy(busy),
""", """    .tok(tok), .pos(pos), .next_tok(next_tok), .best(best), .done(done),
    .busy(busy),
    .emb_en(emb_en), .x_we(x_we), .x_addr(x_addr), .x_wdata(x_wdata), .x_rdata(x_rdata),
""")
    return v


# The testbench's DUT is the wrapper; the step task is an AXI-Lite master.
DUT = """  wire [7:0] kv_arlen, kv_awlen;
  wire s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
  wire [1:0] s_bresp, s_rresp;
  wire [31:0] s_rdata;
  reg [5:0] s_awaddr = 0, s_araddr = 0;
  reg [31:0] s_wdata = 0;
  reg s_awvalid = 0, s_wvalid = 0, s_arvalid = 0;
  reg s_bready = 0, s_rready = 0;
  assign arlen = kv_arlen[3:0]; assign awlen = kv_awlen[3:0];
  fpgai_zybo dut (.aclk(clk), .aresetn(rst_n),
    .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
    .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hf), .s_axi_wvalid(s_wvalid),
    .s_axi_wready(s_wready), .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid),
    .s_axi_bready(s_bready), .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid),
    .s_axi_arready(s_arready), .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp),
    .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
%(wconn)s    .m_axi_kv_araddr(araddr), .m_axi_kv_arlen(kv_arlen),
    .m_axi_kv_arvalid(arvalid), .m_axi_kv_arready(arready),
    .m_axi_kv_rdata(rdata), .m_axi_kv_rresp(2'b00), .m_axi_kv_rvalid(rvalid),
    .m_axi_kv_rlast(rlast), .m_axi_kv_rready(rready),
    .m_axi_kv_awaddr(awaddr), .m_axi_kv_awlen(kv_awlen),
    .m_axi_kv_awvalid(awvalid), .m_axi_kv_awready(awready),
    .m_axi_kv_wdata(wdata), .m_axi_kv_wstrb(wstrb), .m_axi_kv_wlast(wlast),
    .m_axi_kv_wvalid(wvalid), .m_axi_kv_wready(wready),
    .m_axi_kv_bresp(2'b00), .m_axi_kv_bvalid(bvalid), .m_axi_kv_bready(bready));
  assign core_cycles = dut.core_cycles;
"""

STEP = """  // An AXI-Lite master: each valid is held until the edge where it
  // meets ready, as the PS7's GP0 would.
  reg aw_hs, w_hs, ar_hs;
  task wr(input [5:0] a, input [31:0] d);
    begin
      @(negedge clk); s_awaddr = a; s_wdata = d; s_awvalid = 1; s_wvalid = 1;
      s_bready = 1;
      while (s_awvalid || s_wvalid) begin
        @(negedge clk); aw_hs = s_awvalid && s_awready; w_hs = s_wvalid && s_wready;
        @(posedge clk); #1;
        if (aw_hs) s_awvalid = 0;
        if (w_hs) s_wvalid = 0;
      end
      while (!s_bvalid) @(negedge clk);
      @(posedge clk); #1; s_bready = 0;
      if (s_bresp != 0) begin $display("FAIL bresp"); $finish; end
    end
  endtask
  reg [31:0] rv;
  task rd(input [5:0] a);
    begin
      @(negedge clk); s_araddr = a; s_arvalid = 1; s_rready = 1;
      ar_hs = 0;
      while (!ar_hs) begin
        @(negedge clk); ar_hs = s_arready;
        @(posedge clk); #1;
      end
      s_arvalid = 0;
      while (!s_rvalid) @(negedge clk);
      rv = s_rdata; @(posedge clk); #1; s_rready = 0;
    end
  endtask
  task expect_reg(input [5:0] a, input [31:0] v);
    begin
      rd(a);
      if (rv !== v) begin
        $display("FAIL reg %%h = %%h, want %%h", a, rv, v); $finish;
      end
    end
  endtask
  reg checked = 0;
  task step(input integer tk, input integer p, input integer he);
    begin
      if (!checked) begin
        // The register block before any step: identity, layout, idle.
%(checks)s        wr(6'h04, 32'h5a5a5); expect_reg(6'h04, 32'h5a5a5 & %(tmask)d);
        checked = 1;
      end
      wr(6'h04, tk); wr(6'h08, p);
      t0 = cyc; c0 = core_cycles;
      wr(6'h00, 1 | (he << 1));
      rv = 0;
      while (!rv[1]) begin repeat (64) @(negedge clk); rd(6'h0c); end
      rd(6'h1c); t0 = rv;
      rd(6'h10);
      $display("STEP pos=%%0d tok=%%0d clk_cycles=%%0d core_cycles=%%0d next=%%0d",
               p, tk, t0, core_cycles - c0, rv);
      $fflush;
    end
  endtask
"""


def tb_text(L, np_, lat, jit=0):
    checks = "".join("        expect_reg(6'h%02x, 32'h%08x);\n" % (o, v) for o, v in [
        (0x20, FPGAI_ID), (0x24, L["wb"]), (0x28, L["cb"]), (0x2C, L["kb"]),
        (0x30, L["vb"]), (0x34, L["end"]), (0x0C, 0)])
    step = STEP % dict(checks=checks, tmask=(1 << L["W"]["tok"]) - 1)
    wconn = "".join(
        "    .m_axi_w{p}_araddr(w{p}_araddr), .m_axi_w{p}_arlen(w{p}_arlen8),\n"
        "    .m_axi_w{p}_arvalid(w{p}_arvalid), .m_axi_w{p}_arready(w{p}_arready),\n"
        "    .m_axi_w{p}_rdata(w{p}_rdata), .m_axi_w{p}_rresp(2'b00),\n"
        "    .m_axi_w{p}_rvalid(w{p}_rvalid), .m_axi_w{p}_rlast(w{p}_rlast),\n"
        "    .m_axi_w{p}_rready(w{p}_rready),\n".format(p=p) for p in range(np_))
    decl = "".join("  wire [7:0] w{p}_arlen8; assign w{p}_arlen = w{p}_arlen8[3:0];\n"
                   .format(p=p) for p in range(np_))
    dut = (decl + DUT).replace("%(wconn)s", wconn.replace("%", "%%"))
    tb = zybo.tb_text(L, np_, lat, dut=dut, step=step, fs="dut.br.fs", jit=jit)
    return tb.replace("module tb_zybo;", "module tb_fpgai_zybo;")


BUILD = """# GENERATED by board_zybo.py. Build the {title} bitstream and platform:
#   vivado -mode batch -source build.tcl
# Needs an x86 Vivado (2020.2 or later) with
#   {board_files},
# for the PS7's DDR and MIO preset. License: {license}.
# Writes fpgai.xsa for Vitis (see sw/).
create_project -force fpgai ./vivado -part {part}
set bp [lindex [get_board_parts -quiet -latest_file_version "{vivado_board}"] 0]
if {{$bp eq ""}} {{
  puts "ERROR: no {title} board preset: install {board_files}"
  exit 1
}}
set_property board_part $bp [current_project]
add_files [glob ./rtl/*.v]
add_files ./rtl/gains.hex
set_property file_type {{Memory Initialization Files}} [get_files gains.hex]
update_compile_order -fileset sources_1

create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \\
  -config {{make_external "FIXED_IO, DDR" apply_board_preset "1"}} [get_bd_cells ps7]
set_property -dict [list \\
  CONFIG.PCW_USE_M_AXI_GP0 {{1}} \\
  CONFIG.PCW_USE_S_AXI_HP0 {{1}} CONFIG.PCW_USE_S_AXI_HP1 {{1}} \\
  CONFIG.PCW_USE_S_AXI_HP2 {{1}} CONFIG.PCW_USE_S_AXI_HP3 {{1}} \\
  CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {{64}} CONFIG.PCW_S_AXI_HP1_DATA_WIDTH {{64}} \\
  CONFIG.PCW_S_AXI_HP2_DATA_WIDTH {{64}} CONFIG.PCW_S_AXI_HP3_DATA_WIDTH {{64}} \\
  CONFIG.PCW_USE_S_AXI_ACP {{1}} \\
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {{{mhz}}}] [get_bd_cells ps7]
create_bd_cell -type module -reference fpgai_zybo fpgai

# Registers on GP0; the weights on HP0-3; constants and KV on ACP.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {{Clk_master {{/ps7/FCLK_CLK0}} \\
  Clk_slave {{/ps7/FCLK_CLK0}} Clk_xbar {{/ps7/FCLK_CLK0}} Master {{/ps7/M_AXI_GP0}} \\
  Slave {{/fpgai/s_axi}} ddr_seg {{Auto}} intc_ip {{New AXI Interconnect}} master_apm {{0}}}} \\
  [get_bd_intf_pins fpgai/s_axi]
foreach {{m s}} {{m_axi_w0 S_AXI_HP0 m_axi_w1 S_AXI_HP1 m_axi_w2 S_AXI_HP2 m_axi_w3 S_AXI_HP3 m_axi_kv S_AXI_ACP}} {{
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master /ps7/FCLK_CLK0 \\
    Clk_slave /ps7/FCLK_CLK0 Clk_xbar /ps7/FCLK_CLK0 Master /fpgai/$m Slave /ps7/$s \\
    ddr_seg Auto intc_ip {{New AXI SmartConnect}} master_apm 0] [get_bd_intf_pins ps7/$s]
}}
{led_bd}assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data] -filter {{NAME =~ "*fpgai*"}}]
set_property offset 0x{regs:08X} $seg
set_property range 64K $seg
validate_bd_design
save_bd_design

set w [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $w
set_property top system_wrapper [current_fileset]
{led_xdc}launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_timing_summary -file timing.rpt
report_utilization -file utilization.rpt
write_hw_platform -fixed -include_bit -force ./fpgai.xsa
puts "platform: ./fpgai.xsa"
"""

def render_xdc(pk, open_flow=False):
    """The one pin the PL drives: an LED that shows the core busy, where
    the board has one whose bank voltage is known."""
    if not pk.get("led"):
        return "# GENERATED by board_zybo.py. %s: no pins; STATUS shows busy.\n" % pk["title"]
    pin, std, name = pk["led"]
    if open_flow:
        return ("# GENERATED by board_zybo.py, for nextpnr-xilinx: %s shows the core busy.\n"
                "set_property PACKAGE_PIN %s [get_ports busy_led]\n"
                "set_property IOSTANDARD %s [get_ports busy_led]\n" % (name, pin, std))
    return ("# GENERATED by board_zybo.py. The %s's %s shows the core busy.\n"
            "set_property -dict {PACKAGE_PIN %s IOSTANDARD %s} [get_ports busy_led]\n"
            % (pk["title"], name, pin, std))


def render_build(pk):
    led = bool(pk.get("led"))
    return BUILD.format(
        mhz=MHZ, regs=REG_BASE, title=pk["title"], part=pk["part"],
        vivado_board=pk["vivado_board"], board_files=pk["board_files"],
        license=pk["license"],
        led_bd=("create_bd_port -dir O busy_led\n"
                "connect_bd_net [get_bd_pins fpgai/busy_led] [get_bd_ports busy_led]\n"
                if led else ""),
        led_xdc="add_files -fileset constrs_1 ./board.xdc\n" if led else "")


def render_ps7_top(np_=4, led=True):
    """The block design as Verilog, for the open flow (Yosys, nextpnr-xilinx,
    Project X-Ray), which has no IP integrator: the PS7 instanced directly,
    GP0 turned into the register block's AXI-Lite (the CPU's register
    accesses are single beats; their IDs are kept and returned), the
    weight masters on HP0-3, constants and KV on ACP, all on FCLK0."""
    hp = []
    for p in range(np_):
        hp.append("""    .SAXIHP{p}ACLK(clk), .SAXIHP{p}ARADDR(w{p}_araddr), .SAXIHP{p}ARLEN(w{p}_arlen[3:0]),
    .SAXIHP{p}ARSIZE(w{p}_arsize[1:0]), .SAXIHP{p}ARBURST(w{p}_arburst),
    .SAXIHP{p}ARCACHE(w{p}_arcache), .SAXIHP{p}ARPROT(w{p}_arprot),
    .SAXIHP{p}ARLOCK({{1'b0, w{p}_arlock}}), .SAXIHP{p}ARQOS(w{p}_arqos), .SAXIHP{p}ARID(6'd0),
    .SAXIHP{p}ARVALID(w{p}_arvalid), .SAXIHP{p}ARREADY(w{p}_arready),
    .SAXIHP{p}RDATA(w{p}_rdata), .SAXIHP{p}RRESP(w{p}_rresp), .SAXIHP{p}RLAST(w{p}_rlast),
    .SAXIHP{p}RVALID(w{p}_rvalid), .SAXIHP{p}RREADY(w{p}_rready),
    .SAXIHP{p}RDISSUECAP1EN(1'b0), .SAXIHP{p}WRISSUECAP1EN(1'b0),
    .SAXIHP{p}AWVALID(1'b0), .SAXIHP{p}WVALID(1'b0), .SAXIHP{p}BREADY(1'b1),
""".format(p=p))
    wdecl = "".join("""  wire [31:0] w{p}_araddr; wire [7:0] w{p}_arlen; wire [2:0] w{p}_arsize, w{p}_arprot;
  wire [1:0] w{p}_arburst, w{p}_rresp; wire [3:0] w{p}_arcache, w{p}_arqos;
  wire w{p}_arlock, w{p}_arvalid, w{p}_arready, w{p}_rlast, w{p}_rvalid, w{p}_rready;
  wire [63:0] w{p}_rdata;
""".format(p=p) for p in range(np_))
    wconn = "".join("""    .m_axi_w{p}_araddr(w{p}_araddr), .m_axi_w{p}_arlen(w{p}_arlen),
    .m_axi_w{p}_arsize(w{p}_arsize), .m_axi_w{p}_arburst(w{p}_arburst),
    .m_axi_w{p}_arcache(w{p}_arcache), .m_axi_w{p}_arprot(w{p}_arprot),
    .m_axi_w{p}_arlock(w{p}_arlock), .m_axi_w{p}_arqos(w{p}_arqos),
    .m_axi_w{p}_arvalid(w{p}_arvalid), .m_axi_w{p}_arready(w{p}_arready),
    .m_axi_w{p}_rdata(w{p}_rdata), .m_axi_w{p}_rresp(w{p}_rresp),
    .m_axi_w{p}_rlast(w{p}_rlast), .m_axi_w{p}_rvalid(w{p}_rvalid),
    .m_axi_w{p}_rready(w{p}_rready),
""".format(p=p) for p in range(np_))
    return """// GENERATED by board_zybo.py: do not edit by hand.
// fpgai_zybo on the Zynq's PS7 without Vivado's block design, for the
// open flow in open/. Vivado uses build.tcl instead.
module fpgai_ps7 ({ports});
  wire [3:0] fclk, frstn;
  wire clk;
  BUFG bg_fclk (.I(fclk[0]), .O(clk));
  reg [2:0] rs;
  always @(posedge clk) rs <= {{rs[1:0], frstn[0]}};
  wire rst_n = rs[2];

  // GP0, AXI3 from the CPU, to the register block's AXI-Lite.
  wire [31:0] gp_awaddr, gp_araddr, gp_wdata, gp_rdata;
  wire [11:0] gp_awid, gp_arid;
  wire gp_awvalid, gp_awready, gp_wvalid, gp_wready, gp_bvalid, gp_bready;
  wire gp_arvalid, gp_arready, gp_rvalid, gp_rready;
  wire [3:0] gp_wstrb;
  wire [1:0] gp_bresp, gp_rresp;
  reg [11:0] bid, rid;
  always @(posedge clk) begin
    if (gp_awvalid && gp_awready) bid <= gp_awid;
    if (gp_arvalid && gp_arready) rid <= gp_arid;
  end

{wdecl}  wire [31:0] kv_araddr, kv_awaddr; wire [7:0] kv_arlen, kv_awlen;
  wire [2:0] kv_arsize, kv_arprot, kv_awsize, kv_awprot;
  wire [1:0] kv_arburst, kv_awburst, kv_rresp, kv_bresp;
  wire [3:0] kv_arcache, kv_arqos, kv_awcache, kv_awqos;
  wire kv_arlock, kv_awlock, kv_arvalid, kv_arready, kv_rlast, kv_rvalid, kv_rready;
  wire kv_awvalid, kv_awready, kv_wlast, kv_wvalid, kv_wready, kv_bvalid, kv_bready;
  wire [63:0] kv_rdata, kv_wdata; wire [7:0] kv_wstrb;

  fpgai_zybo u (.aclk(clk), .aresetn(rst_n),
    .s_axi_awaddr(gp_awaddr[5:0]), .s_axi_awvalid(gp_awvalid), .s_axi_awready(gp_awready),
    .s_axi_wdata(gp_wdata), .s_axi_wstrb(gp_wstrb), .s_axi_wvalid(gp_wvalid),
    .s_axi_wready(gp_wready), .s_axi_bresp(gp_bresp), .s_axi_bvalid(gp_bvalid),
    .s_axi_bready(gp_bready), .s_axi_araddr(gp_araddr[5:0]), .s_axi_arvalid(gp_arvalid),
    .s_axi_arready(gp_arready), .s_axi_rdata(gp_rdata), .s_axi_rresp(gp_rresp),
    .s_axi_rvalid(gp_rvalid), .s_axi_rready(gp_rready),
{wconn}    .m_axi_kv_araddr(kv_araddr), .m_axi_kv_arlen(kv_arlen), .m_axi_kv_arsize(kv_arsize),
    .m_axi_kv_arburst(kv_arburst), .m_axi_kv_arcache(kv_arcache), .m_axi_kv_arprot(kv_arprot),
    .m_axi_kv_arlock(kv_arlock), .m_axi_kv_arqos(kv_arqos), .m_axi_kv_arvalid(kv_arvalid),
    .m_axi_kv_arready(kv_arready), .m_axi_kv_rdata(kv_rdata), .m_axi_kv_rresp(kv_rresp),
    .m_axi_kv_rlast(kv_rlast), .m_axi_kv_rvalid(kv_rvalid), .m_axi_kv_rready(kv_rready),
    .m_axi_kv_awaddr(kv_awaddr), .m_axi_kv_awlen(kv_awlen), .m_axi_kv_awsize(kv_awsize),
    .m_axi_kv_awburst(kv_awburst), .m_axi_kv_awcache(kv_awcache), .m_axi_kv_awprot(kv_awprot),
    .m_axi_kv_awlock(kv_awlock), .m_axi_kv_awqos(kv_awqos), .m_axi_kv_awvalid(kv_awvalid),
    .m_axi_kv_awready(kv_awready), .m_axi_kv_wdata(kv_wdata), .m_axi_kv_wstrb(kv_wstrb),
    .m_axi_kv_wlast(kv_wlast), .m_axi_kv_wvalid(kv_wvalid), .m_axi_kv_wready(kv_wready),
    .m_axi_kv_bresp(kv_bresp), .m_axi_kv_bvalid(kv_bvalid), .m_axi_kv_bready(kv_bready),
    .busy_led({led}));

  (* keep *) PS7 ps7 (
    .FCLKCLK(fclk), .FCLKRESETN(frstn),
    .MAXIGP0ACLK(clk), .MAXIGP0AWADDR(gp_awaddr), .MAXIGP0AWID(gp_awid),
    .MAXIGP0AWVALID(gp_awvalid), .MAXIGP0AWREADY(gp_awready),
    .MAXIGP0WDATA(gp_wdata), .MAXIGP0WSTRB(gp_wstrb), .MAXIGP0WVALID(gp_wvalid),
    .MAXIGP0WREADY(gp_wready), .MAXIGP0BID(bid), .MAXIGP0BRESP(gp_bresp),
    .MAXIGP0BVALID(gp_bvalid), .MAXIGP0BREADY(gp_bready),
    .MAXIGP0ARADDR(gp_araddr), .MAXIGP0ARID(gp_arid), .MAXIGP0ARVALID(gp_arvalid),
    .MAXIGP0ARREADY(gp_arready), .MAXIGP0RDATA(gp_rdata), .MAXIGP0RID(rid),
    .MAXIGP0RRESP(gp_rresp), .MAXIGP0RLAST(1'b1), .MAXIGP0RVALID(gp_rvalid),
    .MAXIGP0RREADY(gp_rready),
{hp}    .SAXIACPACLK(clk), .SAXIACPARADDR(kv_araddr), .SAXIACPARLEN(kv_arlen[3:0]),
    .SAXIACPARSIZE(kv_arsize[1:0]), .SAXIACPARBURST(kv_arburst), .SAXIACPARCACHE(kv_arcache),
    .SAXIACPARPROT(kv_arprot), .SAXIACPARLOCK({{1'b0, kv_arlock}}), .SAXIACPARQOS(kv_arqos),
    .SAXIACPARID(3'd0), .SAXIACPARUSER(5'd0), .SAXIACPARVALID(kv_arvalid),
    .SAXIACPARREADY(kv_arready), .SAXIACPRDATA(kv_rdata), .SAXIACPRRESP(kv_rresp),
    .SAXIACPRLAST(kv_rlast), .SAXIACPRVALID(kv_rvalid), .SAXIACPRREADY(kv_rready),
    .SAXIACPAWADDR(kv_awaddr), .SAXIACPAWLEN(kv_awlen[3:0]), .SAXIACPAWSIZE(kv_awsize[1:0]),
    .SAXIACPAWBURST(kv_awburst), .SAXIACPAWCACHE(kv_awcache), .SAXIACPAWPROT(kv_awprot),
    .SAXIACPAWLOCK({{1'b0, kv_awlock}}), .SAXIACPAWQOS(kv_awqos), .SAXIACPAWID(3'd0),
    .SAXIACPAWUSER(5'd0), .SAXIACPAWVALID(kv_awvalid), .SAXIACPAWREADY(kv_awready),
    .SAXIACPWDATA(kv_wdata), .SAXIACPWSTRB(kv_wstrb), .SAXIACPWLAST(kv_wlast),
    .SAXIACPWID(3'd0), .SAXIACPWVALID(kv_wvalid), .SAXIACPWREADY(kv_wready),
    .SAXIACPBRESP(kv_bresp), .SAXIACPBVALID(kv_bvalid), .SAXIACPBREADY(kv_bready));
endmodule
""".format(wdecl=wdecl, wconn=wconn, hp="".join(hp),
           ports="output busy_led" if led else "", led="busy_led" if led else "")


# nextpnr-xilinx pins every DSP48E1 that starts no cascade to the lower
# DSP of its tile, which leaves 110 sites on the XC7Z020 for this design's
# 132 lone multipliers: placement fails on the 111th. This lets a lone DSP
# take either site.
DSP_PATCH = "diff --git a/xilinx/pack_dsp_xc7.cc b/xilinx/pack_dsp_xc7.cc\nindex c78250a..4c563fb 100644\n--- a/xilinx/pack_dsp_xc7.cc\n+++ b/xilinx/pack_dsp_xc7.cc\n@@ -162,6 +162,14 @@ void XC7Packer::pack_dsps()\n         root->constr_abs_z = true;\n         root->constr_z = BEL_LOWER_DSP;\n         walk_dsp(root, root, BEL_UPPER_DSP);\n+        // A DSP with no cascade is not a chain root: either DSP48E1 of the\n+        // tile will do. Pinning every one to the lower bel leaves half the\n+        // die's DSPs unusable, and a design with more lone DSPs than DSP\n+        // tiles fails legalisation outright.\n+        if (root->constr_children.empty()) {\n+            root->constr_abs_z = false;\n+            root->constr_z = root->UNCONSTR;\n+        }\n     }\n }\n \n"

OPEN_SH = """#!/usr/bin/env bash
# GENERATED by board_zybo.py. The {title} bitstream without Vivado: Yosys,
# nextpnr-xilinx and Project X-Ray (openXC7). One-off setup, tested on an
# ARM Mac (Apple clang, no OpenMP):
#   git clone --recurse-submodules https://github.com/openXC7/nextpnr-xilinx
#     (tested at bc9b2346), git apply nextpnr-xilinx-dsp.patch, then
#   cmake -B build -DARCH=xilinx -DBUILD_GUI=OFF -DBUILD_PYTHON=OFF -DUSE_OPENMP=OFF
#   cmake --build build
#   python3 xilinx/python/bbaexport.py --device {part} --bba xilinx/{chip}.bba
#   build/bbasm --l xilinx/{chip}.bba xilinx/{chip}.bin
#   git clone --recurse-submodules https://github.com/openXC7/prjxray
#     (tested at 9553f1ad), cmake -B build, cmake --build build
# Set NEXTPNR_XILINX to the
# nextpnr-xilinx checkout (with xilinx/{chip}.bin built) and XRAY_DIR to
# a built prjxray checkout, and PYTHON to a python3 that has prjxray's
# packages (pip install -e "$XRAY_DIR/third_party/fasm" -e "$XRAY_DIR").
# The PS7's own setup (DDR, clocks, level
# shifters) is software's job, done by ps7_init from any {title} Vivado
# project with its board preset, as on any Zynq.
set -euo pipefail
cd "$(dirname "$0")"
: "${{NEXTPNR_XILINX:?set NEXTPNR_XILINX}}" "${{XRAY_DIR:?set XRAY_DIR}}"
cp ../rtl/gains.hex .
yosys -q -l yosys.log -p "read_verilog fpgai_ps7.v $(ls ../rtl/*.v | tr '\\n' ' '); \\
  synth_xilinx -flatten -abc9 -arch xc7 -top fpgai_ps7; write_json fpgai.json"
"$NEXTPNR_XILINX/build/nextpnr-xilinx" --chipdb "$NEXTPNR_XILINX/xilinx/{chip}.bin" \\
  --xdc open.xdc --json fpgai.json --fasm fpgai.fasm --freq {mhz} \\
  --report report.json --log nextpnr.log
"${{PYTHON:-python3}}" "$XRAY_DIR/utils/fasm2frames.py" --part {part} \\
  --db-root "$NEXTPNR_XILINX/xilinx/external/prjxray-db/zynq7" fpgai.fasm > fpgai.frames
"$XRAY_DIR/build/tools/xc7frames2bit" \\
  --part_file "$NEXTPNR_XILINX/xilinx/external/prjxray-db/zynq7/{part}/part.yaml" \\
  --part_name {part} --frm_file fpgai.frames --output_file fpgai.bit
echo "bitstream: $(pwd)/fpgai.bit"
"""


REG_BASE = 0x43C00000


def render_header(L, n_gen, title="Zybo Z7-20", stage=None):
    lines = ["// GENERATED by board_zybo.py: the DDR layout and registers of",
             "// the bitstream this was generated with.",
             "#ifndef FPGAI_LAYOUT_H", "#define FPGAI_LAYOUT_H",
             "#define FPGAI_REGS      0x%08XU" % REG_BASE]
    lines += ["#define FPGAI_%-10s 0x%02XU  /* %s */" % (n, o, w) for o, n, w in REGS]
    lines += ["#define FPGAI_ID_VALUE  0x%08XU" % FPGAI_ID,
              "#define WBASE           0x%08XU" % L["wb"],
              "#define WBYTES          %dU" % (L["words"] * 16),
              "#define CBASE           0x%08XU" % L["cb"],
              "#define CBYTES          %dU" % (L["cn"] * 8),
              "#define KBASE           0x%08XU" % L["kb"],
              "#define KVEND           0x%08XU" % L["end"],
              "#define VOCAB_BASE      0x%08XU" % L["end"],
              "#define MAX_POS         %d" % (1 << L["W"]["pos"]),
              "#define N_GEN           %d" % n_gen,
              "#define CORE_MHZ        %d" % MHZ,
              '#define FPGAI_BOARD     "%s"' % title]
    if stage:
        ip = lambda k: ", ".join(str(v) for v in stage[k])
        lines += ["#define FPGAI_%-10s 0x%02XU  /* %s */" % (n, o, w) for o, n, w in STAGE_REGS]
        lines += ["#define FPGAI_D         %d" % stage["D"],
                  "#define STAGE_INDEX     %d" % stage["index"],
                  "#define STAGE_COUNT     %d" % stage["count"],
                  "#define STAGE_L0        %d" % stage["l0"],
                  "#define STAGE_L1        %d" % stage["l1"],
                  "#define STAGE_FIRST     %d" % int(stage["emb"]),
                  "#define STAGE_LAST      %d" % int(stage["head"]),
                  "#define STAGE_VALUE     0x%08XU" % (stage["l0"] | stage["l1"] << 8
                                                       | int(stage["emb"]) << 16
                                                       | int(stage["head"]) << 17),
                  "#define IP_MY           %s" % ip("ip"),
                  "#define IP_NEXT         %s" % ip("next_ip"),
                  "#define IP_FIRST        %s" % ip("first_ip"),
                  "#define UDP_PORT        %d" % stage.get("port", 5000)]
    lines += ["#endif", ""]
    return "\n".join(lines)


MAIN_C = r"""// GENERATED by board_zybo.py. Bare-metal driver for the fpgai_zybo PL.
// Build it in Vitis as an application on fpgai.xsa with the xilffs
// library (FatFs) enabled in the BSP. It reads the SD card's files into
// DDR, runs the prompt a position at a time, then generates N_GEN tokens,
// printing each one's text on the USB-UART at 115200.
#include <stdio.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "ff.h"
#include "fpgai_layout.h"

#define REG(o) (FPGAI_REGS + (o))
static FATFS fs;

static int load(const char *name, UINTPTR addr, u32 want)
{
    FIL f;
    UINT n;
    u32 got = 0;
    if (f_open(&f, name, FA_READ) != FR_OK) {
        printf("missing %s\r\n", name);
        return -1;
    }
    if (want && f_size(&f) != want) {
        printf("%s is %lu bytes, this bitstream wants %lu\r\n", name,
               (unsigned long)f_size(&f), (unsigned long)want);
        return -1;
    }
    while (f_read(&f, (void *)(addr + got), 1 << 20, &n) == FR_OK && n) {
        got += n;
        if ((got & ((64 << 20) - 1)) == 0)
            printf("  %s: %lu MB\r\n", name, (unsigned long)(got >> 20));
    }
    f_close(&f);
    printf("%s: %lu bytes at 0x%08lx\r\n", name, (unsigned long)got,
           (unsigned long)addr);
    return (int)got;
}

static u32 step(u32 tok, u32 pos, int head)
{
    Xil_Out32(REG(FPGAI_TOK), tok);
    Xil_Out32(REG(FPGAI_POS), pos);
    Xil_Out32(REG(FPGAI_CTRL), 1 | (head ? 2 : 0));
    while (!(Xil_In32(REG(FPGAI_STATUS)) & 2))
        ;
    return Xil_In32(REG(FPGAI_NEXT_TOK));
}

static const u32 *vocab;

static void print_tok(u32 t)
{
    u32 n = vocab[0];
    const char *s = (const char *)(vocab + n + 2);
    if (t < n)
        printf("%.*s", (int)(vocab[t + 2] - vocab[t + 1]), s + vocab[t + 1]);
    fflush(stdout);
}

int main(void)
{
    static u32 prompt[MAX_POS + 1];
    XTime a, b;
    printf("\r\nfpgai: Qwen on the " FPGAI_BOARD " PL\r\n");
    if (Xil_In32(REG(FPGAI_ID)) != FPGAI_ID_VALUE ||
        Xil_In32(REG(FPGAI_WBASE)) != WBASE ||
        Xil_In32(REG(FPGAI_KVEND)) != KVEND) {
        printf("the bitstream's layout is not this program's\r\n");
        return 1;
    }
    if (f_mount(&fs, "0:/", 1) != FR_OK) {
        printf("no SD card\r\n");
        return 1;
    }
    if (load("weights8.bin", WBASE, WBYTES) < 0 ||
        load("cparams.bin", CBASE, CBYTES) < 0 ||
        load("vocab.bin", VOCAB_BASE, 0) < 0 ||
        load("prompt.bin", (UINTPTR)prompt, 0) < 0)
        return 1;
    vocab = (const u32 *)VOCAB_BASE;
    // A new sequence starts with an empty KV cache; the PL reads DDR, so
    // everything the ARM wrote must leave its caches first.
    memset((void *)KBASE, 0, KVEND - KBASE);
    Xil_DCacheFlush();

    u32 n = prompt[0], tok = 0, pos;
    for (pos = 0; pos < n; pos++) {
        print_tok(prompt[pos + 1]);
        XTime_GetTime(&a);
        tok = step(prompt[pos + 1], pos, pos == n - 1);
        XTime_GetTime(&b);
    }
    // N_GEN tokens: each one printed, then fed back for the next.
    for (int g = 0; g < N_GEN && pos < MAX_POS; g++, pos++) {
        print_tok(tok);
        if (g + 1 == N_GEN)
            break;
        XTime_GetTime(&a);
        tok = step(tok, pos, 1);
        XTime_GetTime(&b);
    }
    // XTime counts at half the CPU clock.
    printf("\r\nlast step: %lu core cycles, %lu bus cycles, %.2f s\r\n",
           (unsigned long)Xil_In32(REG(FPGAI_CORE_CYCLES)),
           (unsigned long)Xil_In32(REG(FPGAI_BUS_CYCLES)),
           (double)(b - a) / COUNTS_PER_SECOND);
    return 0;
}
"""


STAGE_C = r"""// GENERATED by board_zybo.py. One pipeline stage of a multi-board run
// (cluster.py), on a Zynq: the stage's layers in the PL, the network on
// the ARM. Build it from Vitis's "lwIP Echo Server" template, which brings
// the lwIP BSP settings and platform_zynq.c; replace that template's
// main.c with this file and add fpgai_layout.h. xilffs must be enabled
// in the BSP as well.
//
// The boards are GALS stages: each on its own clock, sharing nothing but
// messages over UDP, in gals.py's format (A5 5A, type, position, token,
// flags, payload, CRC32). A hidden state is larger than one Ethernet
// frame, so it travels as parts, type 3, each carrying its first word
// index and word count. The first stage holds the prompt and prints the
// text; the last stage runs the head and answers every position with a
// token message, which the first stage waits for before the next one.
#include <stdio.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "ff.h"
#include "platform.h"
#include "platform_config.h"
#include "netif/xadapter.h"
#include "lwip/init.h"
#include "lwip/udp.h"
#include "lwip/ip_addr.h"
#include "fpgai_layout.h"

#define REG(o) (FPGAI_REGS + (o))
#define PART_WORDS 600
// IP_MY and the rest are four octets: expand them before IP4_ADDR counts.
#define SET_IP(ip, ...) IP4_ADDR(ip, __VA_ARGS__)
enum { T_TOKEN = 1, T_HIDDEN = 2, T_PART = 3 };

static FATFS fs;
static struct netif nif;
static struct udp_pcb *pcb;
static ip_addr_t next_ip, first_ip;
static s16 hid[FPGAI_D];
static int hid_have;
static volatile int got_token, got_hidden;
static volatile u32 m_pos, m_tok, m_flags;
static volatile s16 m_best;
static u8 txb[1500];

static u32 crc32(const u8 *p, int n)
{
    u32 c = 0xFFFFFFFFU;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ (0xEDB88320U & (0U - (c & 1U)));
    }
    return ~c;
}

static int build(int type, u32 pos, u32 tok, u32 flags, const u8 *pay, int n)
{
    u8 *b = txb;
    b[0] = 0xA5; b[1] = 0x5A; b[2] = type;
    b[3] = pos; b[4] = pos >> 8;
    b[5] = tok; b[6] = tok >> 8; b[7] = tok >> 16; b[8] = flags;
    memcpy(b + 9, pay, n);
    u32 c = crc32(b + 2, 7 + n);
    for (int k = 0; k < 4; k++)
        b[9 + n + k] = c >> (8 * k);
    return 13 + n;
}

static void send_to(ip_addr_t *dst, int n)
{
    struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, n, PBUF_RAM);
    if (!p)
        return;
    memcpy(p->payload, txb, n);
    udp_sendto(pcb, p, dst, UDP_PORT);
    pbuf_free(p);
}

static void rx(void *arg, struct udp_pcb *u, struct pbuf *p,
               const ip_addr_t *addr, u16_t port)
{
    static u8 b[1500];
    int n = p->tot_len > (int)sizeof b ? (int)sizeof b : p->tot_len;
    pbuf_copy_partial(p, b, n, 0);
    pbuf_free(p);
    (void)arg; (void)u; (void)addr; (void)port;
    if (n < 13 || b[0] != 0xA5 || b[1] != 0x5A)
        return;
    u32 c = b[n - 4] | b[n - 3] << 8 | b[n - 2] << 16 | (u32)b[n - 1] << 24;
    if (crc32(b + 2, n - 6) != c)
        return;                          // dropped; the sender repeats on timeout
    u32 pos = b[3] | b[4] << 8, tok = b[5] | b[6] << 8 | b[7] << 16, fl = b[8];
    if (b[2] == T_TOKEN) {
        m_pos = pos; m_tok = tok; m_flags = fl; m_best = (s16)(b[9] | b[10] << 8);
        got_token = 1;
    } else if (b[2] == T_PART) {
        int off = b[9] | b[10] << 8, cnt = b[11] | b[12] << 8;
        if (off + cnt > FPGAI_D || 13 + 2 * cnt + 4 > n)
            return;
        for (int i = 0; i < cnt; i++)
            hid[off + i] = (s16)(b[13 + 2 * i] | b[14 + 2 * i] << 8);
        hid_have += cnt;
        if (hid_have >= FPGAI_D) {
            m_pos = pos; m_tok = tok; m_flags = fl; hid_have = 0;
            got_hidden = 1;
        }
    }
}

__attribute__((unused)) static void send_hidden(u32 pos, u32 tok, u32 flags)
{
    static u8 pay[4 + 2 * PART_WORDS];
    for (int off = 0; off < FPGAI_D; off += PART_WORDS) {
        int cnt = FPGAI_D - off < PART_WORDS ? FPGAI_D - off : PART_WORDS;
        pay[0] = off; pay[1] = off >> 8; pay[2] = cnt; pay[3] = cnt >> 8;
        for (int i = 0; i < cnt; i++) {
            pay[4 + 2 * i] = (u16)hid[off + i];
            pay[5 + 2 * i] = (u16)hid[off + i] >> 8;
        }
        send_to(&next_ip, build(T_PART, pos, tok, flags, pay, 4 + 2 * cnt));
    }
}

static void run(u32 tok, u32 pos, int emb, int head)
{
    Xil_Out32(REG(FPGAI_TOK), tok);
    Xil_Out32(REG(FPGAI_POS), pos);
    Xil_Out32(REG(FPGAI_CTRL), 1 | (head ? 2 : 0) | (emb ? 4 : 0));
    while (!(Xil_In32(REG(FPGAI_STATUS)) & 2))
        ;
}

__attribute__((unused)) static void load_hidden(void)
{
    Xil_Out32(REG(FPGAI_XADDR), 0);
    for (int i = 0; i < FPGAI_D; i++)
        Xil_Out32(REG(FPGAI_XDATA), (u16)hid[i]);
}

__attribute__((unused)) static void read_hidden(void)
{
    Xil_Out32(REG(FPGAI_XADDR), 0);
    for (int i = 0; i < FPGAI_D; i++)
        hid[i] = (s16)Xil_In32(REG(FPGAI_XDATA));
}

static int load(const char *name, UINTPTR addr, u32 want)
{
    FIL f;
    UINT n;
    u32 got = 0;
    if (f_open(&f, name, FA_READ) != FR_OK) {
        printf("missing %s\r\n", name);
        return -1;
    }
    if (want && f_size(&f) != want) {
        printf("%s is %lu bytes, this bitstream wants %lu\r\n", name,
               (unsigned long)f_size(&f), (unsigned long)want);
        return -1;
    }
    while (f_read(&f, (void *)(addr + got), 1 << 20, &n) == FR_OK && n)
        got += n;
    f_close(&f);
    printf("%s: %lu bytes\r\n", name, (unsigned long)got);
    return (int)got;
}

#if STAGE_FIRST
static const u32 *vocab;
static void print_tok(u32 t)
{
    u32 n = vocab[0];
    const char *s = (const char *)(vocab + n + 2);
    if (t < n)
        printf("%.*s", (int)(vocab[t + 2] - vocab[t + 1]), s + vocab[t + 1]);
    fflush(stdout);
}
#endif

static void wait_for(volatile int *flag)
{
    while (!*flag)
        xemacif_input(&nif);
    *flag = 0;
}

int main(void)
{
    __attribute__((unused)) static u32 prompt[MAX_POS + 1];
    ip_addr_t ip, mask, gw;
    unsigned char mac[6] = {0x00, 0x0a, 0x35, 0x00, 0x01, 0x10 + STAGE_INDEX};
    init_platform();
    printf("\r\nfpgai stage %d of %d on the " FPGAI_BOARD ": layers %d-%d\r\n",
           STAGE_INDEX, STAGE_COUNT, STAGE_L0, STAGE_L1 - 1);
    if (Xil_In32(REG(FPGAI_ID)) != FPGAI_ID_VALUE ||
        Xil_In32(REG(FPGAI_STAGE)) != STAGE_VALUE ||
        Xil_In32(REG(FPGAI_KVEND)) != KVEND) {
        printf("the bitstream is not this stage's\r\n");
        return 1;
    }
    if (f_mount(&fs, "0:/", 1) != FR_OK ||
        load("weights8.bin", WBASE, WBYTES) < 0 ||
        load("cparams.bin", CBASE, CBYTES) < 0)
        return 1;
#if STAGE_FIRST
    if (load("vocab.bin", VOCAB_BASE, 0) < 0 ||
        load("prompt.bin", (UINTPTR)prompt, 0) < 0)
        return 1;
    vocab = (const u32 *)VOCAB_BASE;
#endif
    memset((void *)KBASE, 0, KVEND - KBASE);
    Xil_DCacheFlush();

    lwip_init();
    SET_IP(&ip, IP_MY);
    IP4_ADDR(&mask, 255, 255, 255, 0);
    IP4_ADDR(&gw, 0, 0, 0, 0);
    SET_IP(&next_ip, IP_NEXT);
    SET_IP(&first_ip, IP_FIRST);
    if (!xemac_add(&nif, &ip, &mask, &gw, mac, PLATFORM_EMAC_BASEADDR)) {
        printf("no Ethernet\r\n");
        return 1;
    }
    netif_set_default(&nif);
    platform_enable_interrupts();
    netif_set_up(&nif);
    pcb = udp_new();
    udp_bind(pcb, IP_ADDR_ANY, UDP_PORT);
    udp_recv(pcb, rx, NULL);
    printf("listening on UDP %d\r\n", UDP_PORT);

#if STAGE_FIRST
    // Drive the decode: embed each token, run this stage's layers, pass
    // the hidden state on, and wait for the last stage's token.
    u32 n = prompt[0], tok = prompt[1];
    for (u32 pos = 0; pos < n + N_GEN - 1 && pos < MAX_POS; pos++) {
        int head = pos >= n - 1;
        if (pos < n) {
            tok = prompt[pos + 1];
            print_tok(tok);
        }
        run(tok, pos, 1, 0);
        read_hidden();
        send_hidden(pos, tok, head);
        wait_for(&got_token);
        if (head) {
            tok = m_tok;
            print_tok(tok);
        }
    }
    printf("\r\n");
#else
    // Wait for a hidden state, run this stage's layers on it, and pass on
    // the result: the next stage's input, or from the last, the token.
    for (;;) {
        wait_for(&got_hidden);
        load_hidden();
        int head = STAGE_LAST && (m_flags & 1);
        run(m_tok, m_pos, 0, head);
#if STAGE_LAST
        u8 pay[2];
        s16 best = (s16)Xil_In32(REG(FPGAI_BEST));
        pay[0] = (u16)best; pay[1] = (u16)best >> 8;
        send_to(&first_ip, build(T_TOKEN, m_pos + 1,
                                 head ? Xil_In32(REG(FPGAI_NEXT_TOK)) : 0,
                                 m_flags, pay, 2));
#else
        read_hidden();
        send_hidden(m_pos, m_tok, m_flags);
#endif
    }
#endif
    return 0;
}
"""


def write_sd(L, work, sd, prompt, tokzr, first=True):
    os.makedirs(sd, exist_ok=True)
    w8 = zybo.write_w8(work)
    dst = os.path.join(sd, "weights8.bin")
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(w8):
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(w8, dst)
        except OSError:
            shutil.copyfile(w8, dst)
    with open(os.path.join(work, "cparams.hex")) as f, \
            open(os.path.join(sd, "cparams.bin"), "wb") as g:
        for line in f:
            if line.strip():
                g.write(struct.pack("<Q", int(line, 16)))
    if not first:
        return                  # only the first stage prints, and holds the prompt
    # vocab.bin: n, n+1 offsets, then every token's bytes.
    n = max(tokzr.inv) + 1
    blob, offs = bytearray(), [0]
    for t in range(n):
        s = tokzr.inv.get(t, "")
        blob += bytes(tokzr.u2b.get(c, 63) for c in s)
        offs.append(len(blob))
    with open(os.path.join(sd, "vocab.bin"), "wb") as g:
        g.write(struct.pack("<%dI" % (n + 2), n, *offs) + bytes(blob))
    with open(os.path.join(sd, "prompt.bin"), "wb") as g:
        g.write(struct.pack("<%dI" % (len(prompt) + 1), len(prompt), *prompt))


README = """# fpgai on the {title}

GENERATED by `board_zybo.py`. The generated Qwen2.5-0.5B sequencer in
the Zynq's PL, fed from DDR; the ARM loads the model from the SD card
and drives it a position at a time.

## Build

1. `vivado -mode batch -source build.tcl` on an x86 host with
   {board_files}; {license}. Writes `fpgai.xsa` and the reports.
   Without Vivado, `open/build_open.sh` writes the PL's bitstream.
2. In Vitis: a platform from `fpgai.xsa` (standalone, ps7_cortexa9_0),
   with `xilffs` enabled in the BSP; an empty C application with
   `sw/main.c` and `sw/fpgai_layout.h`.
3. Copy `sd/*` to a FAT32 SD card, {boot} (or make a BOOT.bin with
   the FSBL), and open {uart} at 115200.

## Memory map (DDR, 1 GB)

| region | address | bytes |
|---|---|---|
| the ARM's program | 0x00100000 | below the weights |
| weights8.bin, int8 | 0x{wb:08X} | {wbytes} |
| cparams.bin, column constants | 0x{cb:08X} | {cbytes} |
| K cache | 0x{kb:08X} | {kbytes} |
| V cache | 0x{vb:08X} | {vbytes} |
| vocab.bin | 0x{end:08X} | text of each token |

The RMSNorm gains are in block RAM, from `rtl/gains.hex`.

## Registers (AXI-Lite on GP0 at 0x{regs:08X})

| offset | name | meaning |
|---|---|---|
{regtab}

A step: write TOK and POS, write CTRL = 1 (| 2 for the head), poll
STATUS until bit 1, read NEXT_TOK. The software checks ID and the
layout registers against its header first, so SD files and a bitstream
from different builds are caught before anything runs.

## What is and is not checked

Simulated here: the register block, and a full one-layer build of this
same design run only through AXI-Lite, against a DDR model with 30
cycles of latency on every port, matching the direct testbench's token
and core cycles, both with every port always ready and with every port
stalling and gapping its beats at random (`--jitter`). Not run: Vivado, so the block design, the AXI
interconnects, timing at {mhz} MHz and the software have not met real
hardware. The core clock is gated with a BUFGCE while a port's line is
still on the bus; that path is the first thing to read in the timing
report.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "build_qfull"),
                    help="a qwen_full.py build")
    ap.add_argument("--board", default="zybo_z7_20",
                    choices=sorted(k for k, v in boards.PACKAGES.items() if v["ps7"]))
    ap.add_argument("--out", default=None, help="default: the board's folder")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--latency", type=int, default=30)
    ap.add_argument("--sim", action="store_true",
                    help="simulate the wrapper on the build's own steps")
    ap.add_argument("--no-sd", action="store_true")
    ap.add_argument("--jitter", action="store_true",
                    help="DDR ports stall and gap at random in the simulation")
    a = ap.parse_args()
    package(a.work, a.board, a.out, a.prompt, a.tokens, not a.no_sd,
            sim=a.sim, latency=a.latency, jitter=a.jitter)


def package(work, board="zybo_z7_20", out=None, prompt="The capital of France is",
            tokens=16, sd=True, stage=None, sim=False, latency=30, jitter=False):
    """Write a board's package from a qwen_full.py build. stage: one
    stage of a multi-board run, from cluster.py (its layers, its place in
    the chain, its peers' addresses); the build must be a stage build."""
    pk = boards.PACKAGES[board]
    out = out or os.path.join(ROOT, pk["out"])
    L = zybo.layout(work, BASE)
    if stage:
        L["stage"] = stage
    # The weights, constants, KV cache and the vocabulary all sit in the
    # PS's DDR; a board whose DDR is smaller cannot hold this build.
    if L["end"] + (4 << 20) > pk["ddr_bytes"]:
        sys.exit("%s: this build needs %d MB of DDR, the board has %d MB"
                 % (pk["title"], (L["end"] + (4 << 20)) >> 20, pk["ddr_bytes"] >> 20))
    rtl = os.path.join(out, "rtl")
    os.makedirs(rtl, exist_ok=True)
    # Not the testbenches, nor the copies a --sim run leaves in the build.
    srcs = [f for f in os.listdir(work) if f.endswith(".v")
            and not f.startswith(("tb_", "bz_")) and f != "qwen_zybo.v"]
    for f in srcs:
        shutil.copyfile(os.path.join(work, f), os.path.join(rtl, f))
    shutil.copyfile(os.path.join(work, "gains.hex"), os.path.join(rtl, "gains.hex"))
    with open(os.path.join(rtl, "qwen_zybo.v"), "w") as f:
        f.write(zybo.render_top(L["W"], L["gn"], L["cb"], L["kb"], L["vb"], 4, L["wb"]))
    with open(os.path.join(rtl, "fpgai_zybo.v"), "w") as f:
        f.write(render_wrapper(L))
    with open(os.path.join(out, "build.tcl"), "w") as f:
        f.write(render_build(pk))
    with open(os.path.join(out, "board.xdc"), "w") as f:
        f.write(render_xdc(pk))
    od = os.path.join(out, "open")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "fpgai_ps7.v"), "w") as f:
        f.write(render_ps7_top(led=bool(pk.get("led"))))
    with open(os.path.join(od, "build_open.sh"), "w") as f:
        f.write(OPEN_SH.format(mhz=MHZ, part=pk["part"], chip=pk["chip"],
                               title=pk["title"]))
    os.chmod(os.path.join(od, "build_open.sh"), 0o755)
    with open(os.path.join(od, "open.xdc"), "w") as f:
        f.write(render_xdc(pk, open_flow=True))
    with open(os.path.join(od, "nextpnr-xilinx-dsp.patch"), "w") as f:
        f.write(DSP_PATCH)
    sw = os.path.join(out, "sw")
    os.makedirs(sw, exist_ok=True)
    with open(os.path.join(sw, "fpgai_layout.h"), "w") as f:
        f.write(render_header(L, tokens, pk["title"], stage))
    with open(os.path.join(sw, "main.c"), "w") as f:
        f.write(STAGE_C if stage else MAIN_C)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(
            wb=L["wb"], wbytes=L["words"] * 16, cb=L["cb"], cbytes=L["cn"] * 8,
            kb=L["kb"], kbytes=L["kn"] * 32, vb=L["vb"], vbytes=L["vn"] * 32,
            end=L["end"], regs=REG_BASE, mhz=MHZ, title=pk["title"],
            board_files=pk["board_files"], license=pk["license"],
            boot=pk["boot"], uart=pk["uart"],
            regtab="\n".join("| 0x%02X | %s | %s |" % r
                              for r in REGS + (STAGE_REGS if stage else []))))
    if sd:
        import qwen_real
        tk = qwen_real.Tokenizer()
        write_sd(L, work, os.path.join(out, "sd"), tk.encode(prompt), tk,
                 first=not stage or stage["emb"])
    print("package in", out)
    if sim:
        simd = os.path.join(out, "sim")
        os.makedirs(simd, exist_ok=True)
        with open(os.path.join(simd, "tb_fpgai_zybo.v"), "w") as f:
            f.write(tb_text(L, 4, latency, int(jitter)))
        # The DDR model reads weights8.bin and cparams.hex from the build.
        shutil.copyfile(os.path.join(simd, "tb_fpgai_zybo.v"),
                        os.path.join(work, "tb_fpgai_zybo.v"))
        for f in ("qwen_zybo.v", "fpgai_zybo.v"):
            shutil.copyfile(os.path.join(rtl, f), os.path.join(work, "bz_" + f))
        zybo.simulate(work, "tb_fpgai_zybo.v", ["bz_qwen_zybo.v", "bz_fpgai_zybo.v"])


if __name__ == "__main__":
    main()
