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
import textwrap

import boards
import zybo

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = 0x08000000               # weights; below it the ARM's program
FPGAI_ID = 0xF96A0001
# Without an SD card, sw/load_jtag.tcl writes the files over JTAG and then
# this word at JTAG_MARK, after the files' sizes; prompt.bin goes 64 bytes
# past it. The vocabulary below it has 16 MB.
JTAG_MAGIC = 0xF96A10AD
VOCAB_ROOM = 16 << 20
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
# A tensor-parallel rank's package adds these instead.
TP_REGS = [
    (0x38, "GADDR", "element of the vector being gathered to access next (auto-increments)"),
    (0x3C, "GDATA", "that element, 16 bits signed: read this rank's slice, write the others'"),
    (0x40, "TP", "bits 7:0 this rank, 15:8 the ranks"),
    (0x44, "GATHER", "bit 0 the core waits on a gather not yet answered, bits 3:1 which "
                     "(1 the attention context, 2 o's or down's output, 3 the gated "
                     "product, 4 the head's best); write 1 once the other ranks' slices are in"),
    (0x48, "GMOVE", "write n, with bit 16 set for into the core: the PL moves n words from "
                    "element GADDR of the vector being gathered to DDR at GBUF, or back; "
                    "bit 0 reads 1 while it moves"),
    (0x4C, "GBUF", "DDR address of the gather buffer: element i at byte 2i"),
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
    if "x_addr" in W:
        return stage_wrapper(v, L)
    return tp_wrapper(v, L) if "gx_addr" in W else v



def stage_wrapper(v, L):
    """The register block for one pipeline stage of a multi-board run:
    XADDR and XDATA load the hidden state the stage starts from and read
    back the one it ends with, CTRL bit 2 runs the embedding instead, and
    STAGE says which layers this bitstream holds. The core's clock is
    held while a line fills, so an XDATA access holds its AXI response
    until the core has taken the edges it needs: two to take a write,
    three to have the word read."""
    return _window(v, L, "x")


def tp_wrapper(v, L):
    """The register block for one rank of a tensor-parallel run: GADDR
    and GDATA read this rank's slice of the vector being gathered and
    write the other ranks' slices into it, a word an access, waiting for
    the core's edges as XDATA does; GATHER says whether and what the core
    is waiting for, and a write of 1 lets it go on once the slices are in;
    TP says which rank of how many this bitstream is."""
    return _window(v, L, "gx")


def _window(v, L, px):
    """A word-at-a-time window into the core's vectors, at 0x38 and 0x3C,
    for a stage (px "x") or a tensor-parallel rank (px "gx")."""
    xa = L["W"][px + "_addr"]
    stage = px == "x"
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
    if stage:
        st = L.get("stage", dict(l0=0, l1=0, emb=1, head=1))
        rep("5'h00: s_axi_rdata <= {30'd0, head_en, 1'b0};",
            "5'h00: s_axi_rdata <= {29'd0, emb_en, head_en, 1'b0};")
        ident = "          5'h10: s_axi_rdata <= 32'h%08X;\n" % (
            st["l0"] | st["l1"] << 8 | st["emb"] << 16 | st["head"] << 17)
    else:
        r, T = L["tp"]["rank"], L["tp"]["ranks"]
        ident = ("          5'h10: s_axi_rdata <= 32'h%08X;\n"
                 "          5'h11: s_axi_rdata <= {28'd0, g_vec, g_req && !g_done};\n"
                 "          5'h12: s_axi_rdata <= {31'd0, dma_busy || dma_go};\n"
                 "          5'h13: s_axi_rdata <= 32'h%08X;\n" % (r | T << 8, L["gb"]))
    rep("""          default: s_axi_rdata <= 32'd0;""",
        """          5'h0E: s_axi_rdata <= {%d'd0, %s_addr};
%s          default: s_axi_rdata <= 32'd0;""" % (32 - xa, px, ident))
    if stage:
        rep("""          4'h0: begin head_en <= wd[1];""",
            """          5'h00: begin head_en <= wd[1]; emb_en <= wd[2];""")
    else:
        rep("""          4'h0: begin head_en <= wd[1];""", """          5'h00: begin head_en <= wd[1];""")
    rep("""          4'h1: tok <= wd[""", """          5'h01: tok <= wd[""")
    rep("""          4'h2: pos <= wd[""", """          5'h02: pos <= wd[""")
    # A gather's answer is held until the core drops its request.
    go = "" if stage else ("          5'h11: if (wd[0]) g_done <= 1'b1;\n"
                           "          5'h12: begin dma_go <= 1'b1; dma_in <= wd[16]; "
                           "dma_n <= wd[%d:0]; end\n" % xa)
    drop = "" if stage else ("      if (g_done && !g_req) g_done <= 1'b0;\n"
                             "      if (dma_go && dma_busy) dma_go <= 1'b0;\n")
    rep("""          default: ;
        endcase
      end
      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;""", """          5'h0E: {px}_addr <= wd[{xam}:0];
          5'h0F: begin {px}_wdata <= wd[15:0]; {px}_we <= 1'b1; xcc <= core_cycles; xop <= 2'd1; end
{go}          default: ;
        endcase
      end
      // The {what} accesses finish on the core's edges.
      if (xop == 2'd1 && core_cycles - xcc >= 2) begin
        {px}_we <= 1'b0; xop <= 2'd0; s_axi_bvalid <= 1'b1; {px}_addr <= {px}_addr + 1;
      end
      if (xop == 2'd2 && core_cycles - xcc >= 3) begin
        s_axi_rdata <= {{{{16{{{px}_rdata[15]}}}}, {px}_rdata}}; s_axi_rvalid <= 1'b1;
        xop <= 2'd0; {px}_addr <= {px}_addr + 1;
      end
{drop}      if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 1'b0;""".format(
        px=px, xam=xa - 1, go=go, drop=drop,
        what="hidden-state" if stage else "gather"))
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
    if stage:
        rst = "      emb_en <= 1'b1; x_we <= 1'b0; x_addr <= 0; x_wdata <= 0; xop <= 2'd0; xcc <= 0;"
        decl = ("  reg emb_en, x_we; reg [%d:0] x_addr; reg signed [15:0] x_wdata;\n"
                "  wire signed [15:0] x_rdata; reg [1:0] xop; reg [31:0] xcc;" % (xa - 1))
        conn = "    .emb_en(emb_en), .x_we(x_we), .x_addr(x_addr), .x_wdata(x_wdata), .x_rdata(x_rdata),\n"
    else:
        rst = ("      gx_we <= 1'b0; gx_addr <= 0; gx_wdata <= 0; xop <= 2'd0; xcc <= 0;"
               " g_done <= 1'b0; dma_go <= 1'b0; dma_in <= 1'b0; dma_n <= 0;")
        decl = ("  reg gx_we, g_done; reg [%d:0] gx_addr; reg signed [15:0] gx_wdata;\n"
                "  wire signed [15:0] gx_rdata; wire g_req; wire [2:0] g_vec;\n"
                "  reg [1:0] xop; reg [31:0] xcc;\n"
                "  reg dma_go, dma_in; reg [%d:0] dma_n; wire dma_busy;" % (xa - 1, xa))
        conn = ("    .g_req(g_req), .g_vec(g_vec), .g_done(g_done), .gx_we(gx_we),\n"
                "    .gx_addr(gx_addr), .gx_wdata(gx_wdata), .gx_rdata(gx_rdata),\n"
                "    .dma_go(dma_go), .dma_in(dma_in), .dma_first(gx_addr), .dma_n(dma_n),\n"
                "    .dma_busy(dma_busy),\n")
    rep("""      tok <= 0; pos <= 0; bus_cycles <= 0;""",
        """      tok <= 0; pos <= 0; bus_cycles <= 0;
""" + rst)
    rep("""  reg start, head_en, done_s;""", """  reg start, head_en, done_s;
""" + decl)
    rep("""    .tok(tok), .pos(pos), .next_tok(next_tok), .best(best), .done(done),
    .busy(busy),
""", """    .tok(tok), .pos(pos), .next_tok(next_tok), .best(best), .done(done),
    .busy(busy),
""" + conn)
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

def _open_synth(pk, top="fpgai_ps7"):
    """Yosys's synthesis for the open flow, split around its DSP mapping.

    Yosys 0.68's xilinx_dsp pass, folding a product's register into the
    DSP's M register when that register has a synchronous reset, ties the
    register's bits above the product's own width to 0 rather than to its
    sign: `if (!rst_n) p <= 0; else p <= a * $signed({1'b0, b});` reads
    back wrong for every negative product. Every MAC and requantizer here
    is written that way, and the first ZC706 bitstream answered token 0
    at every step. Unmapping synchronous resets just before the DSP
    mapping keeps those registers out of the DSPs; the resets come back
    as the flip-flops' R pins.

    A core that fills the part's DSPs (the Zybo's at 32 lanes, 196 of
    220) leaves nextpnr-xilinx no free LUT beside some DSPs to make the
    constants their unused inputs need, and routing fails; so the first
    soft_score_lanes of the attention head's score lanes multiply in LUTs
    instead, retyped before the DSP mapping. Vivado's build keeps every
    multiplier on a DSP."""
    synth = "synth_xilinx -flatten -abc9 -arch xc7 -top %s" % top
    n = pk.get("soft_score_lanes", 0)
    soft = ""
    if n:
        lanes = " ".join("c:*slane?%d?.*" % k for k in range(n)) + " %u" * (n - 1)
        soft = "  select -set soft t:\\$mul %s %%i; chtype -set \\$__soft_mul @soft; \\\n" % lanes
    return ("  %s -run :map_dsp; \\\n  dffunmap -srst-only; \\\n%s  %s -run map_dsp:"
            % (synth, soft, synth))


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

# Project X-Ray's fasm2frames adds a glue bit ten rows above any column-62
# GFAN0 ground tie, a rule observed on one part; on the XC7Z045 that tile
# can fall off the grid (Y354) and the lookup aborts the conversion. This
# skips the glue where its tile does not exist.
XRAY_PATCH = "diff --git a/utils/fasm2frames.py b/utils/fasm2frames.py\nindex 0e76d3f..d46341b 100755\n--- a/utils/fasm2frames.py\n+++ b/utils/fasm2frames.py\n@@ -515,7 +515,10 @@ def run(\n                     x62_gfan_tile = 'INT_L_X{}Y{}'.format(m.group(2), int(m.group(3)) + 10)\n                 break\n \n-        if x62_gfan_tile:\n+        # The rule was observed on one part's column 62; on a part whose\n+        # grid has no tile ten rows up (the XC7Z045 at Y354) there is no\n+        # glue bit to set, and looking one up would abort the conversion.\n+        if x62_gfan_tile and x62_gfan_tile in assembler.grid.tileinfo:\n             feature = '{}.GFAN_TIE_ROOT_GLUE'.format(x62_gfan_tile)\n             for line in fasm.parse_fasm_string(feature):\n                 assembler.add_fasm_line(line, glue_missing)\n"

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
#     (tested at 9553f1ad), git apply prjxray-glue.patch, cmake -B build,
#     cmake --build build
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
{synth}; write_json fpgai.json"
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

# What each model's package has been simulated through (RESULTS.md).
CHECKED = {
    "qwen2.5": ("Qwen2.5-0.5B",
                "this design run only through AXI-Lite against a DDR model with 30 "
                "cycles of latency on every port: one layer matching the direct "
                "testbench's token and core cycles, with every port always ready "
                "and with every port stalling and gapping its beats at random "
                "(`--jitter`); all 24 layers and the head the same way, on the "
                "16-lane build and on the 32-lane one, choosing \" Paris\"; and this "
                "package's own ARM program, compiled unchanged, driving this RTL "
                "from its own SD card's files (`cosim.py --jitter`), printing the "
                "integer model's 16 tokens with its logits."),
    "qwen3": ("Qwen3-0.6B",
              "this design with one layer run only through AXI-Lite against the "
              "stalling DDR model, matching the direct testbench's token and core "
              "cycles; the sequencer itself through all 28 layers and the head on "
              "the direct testbench, choosing \" Paris\" as the integer model does; "
              "and this package's own ARM program, compiled unchanged, driving this "
              "RTL from its own SD card's files (`cosim.py --jitter`), printing the "
              "integer model's 16 tokens with its logits."),
}


def render_header(L, n_gen, title="Zybo Z7-20", stage=None, tp=None):
    lines = ["// GENERATED by board_zybo.py: the DDR layout and registers of",
             "// the bitstream this was generated with.",
             "#ifndef FPGAI_LAYOUT_H", "#define FPGAI_LAYOUT_H",
             "#define FPGAI_REGS      0x%08XU" % REG_BASE]
    lines += ["#define FPGAI_%-10s 0x%02XU  /* %s */" % (n, o, w) for o, n, w in REGS]
    lines += ["#define FPGAI_ID_VALUE  0x%08XU" % FPGAI_ID,
              "#define WBASE           0x%08XU" % L["wb"],
              "#define WBYTES          %dU" % (L["words"] * L["N"]),
              "#define CBASE           0x%08XU" % L["cb"],
              "#define CBYTES          %dU" % (L["cn"] * 8),
              "#define KBASE           0x%08XU" % L["kb"],
              "#define KVEND           0x%08XU" % L["end"],
              "#define VOCAB_BASE      0x%08XU" % L.get("gend", L["end"]),
              "#define MAX_POS         %d" % (1 << L["W"]["pos"]),
              "#define N_GEN           %d" % n_gen,
              "#define CORE_MHZ        %d" % MHZ,
              '#define FPGAI_BOARD     "%s"' % title]
    if not stage and not tp:
        jm = L.get("gend", L["end"]) + VOCAB_ROOM
        lines += ["#define JTAG_MARK       0x%08XU  /* the JTAG loader's marker and sizes */" % jm,
                  "#define JTAG_PROMPT     0x%08XU  /* prompt.bin, loaded over JTAG */" % (jm + 64),
                  "#define JTAG_MAGIC      0x%08XU" % JTAG_MAGIC]
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
    if tp:
        # Every rank's slice of the context, of o's and down's output and
        # of the gated product: words and first word. Even without shares.
        T = tp["ranks"]
        sl = tp.get("slices") or [(tp["HHD"] // T, tp["D"] // T, tp["F"] // T)] * T
        of = [tuple(sum(x[k] for x in sl[:r]) for k in range(3)) for r in range(T)]
        lines += ["#define FPGAI_%-10s 0x%02XU  /* %s */" % (n, o, w) for o, n, w in TP_REGS]
        lines += ["#define FPGAI_HHD       %d" % tp["HHD"],
                  "#define FPGAI_D         %d" % tp["D"],
                  "#define FPGAI_F         %d" % tp["F"],
                  "#define TP_RANK         %d" % tp["rank"],
                  "#define TP_RANKS        %d" % tp["ranks"],
                  "#define TP_VALUE        0x%08XU" % (tp["rank"] | tp["ranks"] << 8),
                  "#define GBUF            0x%08XU" % L.get("gb", 0),
                  "#define TP_SLICES       %s" % ", ".join("{%d, %d, %d}" % tuple(x) for x in sl),
                  "#define TP_OFFSETS      %s" % ", ".join("{%d, %d, %d}" % x for x in of),
                  "#define IP_RANKS        %s" % ", ".join(
                      "{%s}" % ", ".join(str(v) for v in ip) for ip in tp["ips"]),
                  "#define UDP_PORT        %d" % tp.get("port", 5000)]
    lines += ["#endif", ""]
    return "\n".join(lines)


MAIN_C = r"""// GENERATED by board_zybo.py. Bare-metal driver for the fpgai_zybo PL.
// Build it in Vitis as an application on fpgai.xsa with the xilffs
// library (FatFs) enabled in the BSP. It reads the SD card's files into
// DDR, or without a card waits for sw/load_jtag.tcl to write them there
// over JTAG, runs the prompt a position at a time, then generates N_GEN
// tokens, printing each one's text on the USB-UART at 115200.
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

// No SD card: the same files over JTAG. sw/load_jtag.tcl, run in XSCT
// while this waits, writes each to its address, then the sizes of the
// first two and last the marker. The flushes make the marker the
// debugger wrote visible here whichever way it went, through this core's
// caches or straight to DDR; the full flush after it puts every file in
// DDR, where the PL reads them.
static int wait_jtag(void)
{
    volatile u32 *mark = (volatile u32 *)JTAG_MARK;
    printf("no SD card: waiting for the files over JTAG; in XSCT run\r\n"
           "  source sw/load_jtag.tcl\r\n");
    mark[0] = 0;
    Xil_DCacheFlushRange((UINTPTR)mark, 64);
    while (mark[0] != JTAG_MAGIC)
        Xil_DCacheFlushRange((UINTPTR)mark, 64);
    Xil_DCacheFlush();
    if (mark[1] != WBYTES || mark[2] != CBYTES) {
        printf("weights8.bin is %lu bytes and cparams.bin %lu, this bitstream "
               "wants %lu and %lu\r\n", (unsigned long)mark[1],
               (unsigned long)mark[2], (unsigned long)WBYTES,
               (unsigned long)CBYTES);
        return -1;
    }
    printf("files loaded over JTAG\r\n");
    return 0;
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
    if (f_mount(&fs, "0:/", 1) == FR_OK) {
        if (load("weights8.bin", WBASE, WBYTES) < 0 ||
            load("cparams.bin", CBASE, CBYTES) < 0 ||
            load("vocab.bin", VOCAB_BASE, 0) < 0 ||
            load("prompt.bin", (UINTPTR)prompt, 0) < 0)
            return 1;
    } else {
        if (wait_jtag() < 0)
            return 1;
        memcpy(prompt, (const void *)JTAG_PROMPT, sizeof prompt);
    }
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
//
// A datagram that is lost or fails its CRC is dropped, and the first
// stage sends its hidden state again when no answer comes within
// RESEND_MS. Running a position twice is harmless: its KV cache entries
// are written again with the same values, so every later stage simply
// recomputes a repeated position and passes it on.
#include <stdio.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xtime_l.h"
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
#define NPARTS ((FPGAI_D + PART_WORDS - 1) / PART_WORDS)
// Longer than a whole step through every stage (about a second for
// Qwen3 over two boards), so only a lost message triggers it.
#ifndef RESEND_MS
#define RESEND_MS 5000
#endif
// IP_MY and the rest are four octets: expand them before IP4_ADDR counts.
#define SET_IP(ip, ...) IP4_ADDR(ip, __VA_ARGS__)
enum { T_TOKEN = 1, T_HIDDEN = 2, T_PART = 3 };

static FATFS fs;
static struct netif nif;
static struct udp_pcb *pcb;
static ip_addr_t next_ip, first_ip;
static s16 hid[FPGAI_D];
static u32 part_pos = 0xFFFFFFFFU;
static u8 part_got[NPARTS];
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
        return;                          // dropped: the first stage resends
    u32 pos = b[3] | b[4] << 8, tok = b[5] | b[6] << 8 | b[7] << 16, fl = b[8];
    if (b[2] == T_TOKEN) {
        m_pos = pos; m_tok = tok; m_flags = fl; m_best = (s16)(b[9] | b[10] << 8);
        got_token = 1;
    } else if (b[2] == T_PART) {
        int off = b[9] | b[10] << 8, cnt = b[11] | b[12] << 8;
        if (off % PART_WORDS || off + cnt > FPGAI_D || 13 + 2 * cnt + 4 > n)
            return;
        // Parts are collected per position; a repeat of one only
        // rewrites it, and the state is whole when every part is in.
        if (pos != part_pos) {
            memset(part_got, 0, sizeof part_got);
            part_pos = pos;
        }
        for (int i = 0; i < cnt; i++)
            hid[off + i] = (s16)(b[13 + 2 * i] | b[14 + 2 * i] << 8);
        part_got[off / PART_WORDS] = 1;
        int all = 1;
        for (int k = 0; k < NPARTS; k++)
            all &= part_got[k];
        if (all) {
            m_pos = pos; m_tok = tok; m_flags = fl;
            part_pos = 0xFFFFFFFFU;
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

// Poll the network until flag is set; 0 if ms pass first (ms 0: forever).
static int wait_for(volatile int *flag, u32 ms)
{
    XTime t0, t;
    XTime_GetTime(&t0);
    while (!*flag) {
        xemacif_input(&nif);
        XTime_GetTime(&t);
        if (ms && (t - t0) > (XTime)ms * (COUNTS_PER_SECOND / 1000))
            return 0;
    }
    *flag = 0;
    return 1;
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
        Xil_In32(REG(FPGAI_KVEND)) != (u32)KVEND) {
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
        do
            send_hidden(pos, tok, head);
        while (!wait_for(&got_token, RESEND_MS) || m_pos != pos + 1);
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
        wait_for(&got_hidden, 0);
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


TP_C = r"""// GENERATED by board_zybo.py. One rank of a tensor-parallel run (tp.py)
// on a Zynq: this rank's slice of every layer in the PL, the gathers on
// the ARM over UDP. Build it from Vitis's "lwIP Echo Server" template as
// the stage program is built: replace its main.c with this file, add
// fpgai_layout.h, and enable xilffs in the BSP.
//
// Every rank runs every position. The core stops at each gather with
// GATHER bit 0 set and the vector's number in bits 3:1. The PL moves this
// rank's slice out to the gather buffer in DDR (GMOVE), the ARM sends it
// to every other rank, writes theirs into the buffer as they arrive, the
// PL moves the whole vector back into the core, and the ARM writes 1 to
// GATHER. The buffer is on the PL's cache-coherent port, so the ARM reads
// and writes it as ordinary memory. At the
// head each rank offers its best logit and that token, and every rank
// takes the same winner: the largest logit, the lower rank on a tie,
// which is the lower token, as the integer model's argmax.
//
// A rank's share need not be the others': the planner sizes each by its
// board's speed, and TP_SLICES and TP_OFFSETS say where every rank's
// slice of each vector lies. Gathers are numbered from 1 in the order
// every rank meets them, and a slice travels in parts of at most
// PART_WORDS words, each a datagram in
// gals.py's framing (A5 5A, type, position, token, flags, payload,
// CRC32). A rank can be one gather ahead of another and never more, so
// parts that come early are kept for the next gather, and this rank's
// previous slice is kept for a rank one behind. A rank still missing
// parts after RESEND_MS asks their sender again (type 5), and the answer
// is the slice once more: a lost or corrupt datagram costs one timeout.
#include <stdio.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xtime_l.h"
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
#define MAX2(a, b) ((a) > (b) ? (a) : (b))
#ifndef FPGAI_BARRIER
#define FPGAI_BARRIER() __asm__ volatile ("dsb" ::: "memory")
#endif
#define VMAX MAX2(MAX2(FPGAI_HHD, FPGAI_D), MAX2(FPGAI_F, 3 * TP_RANKS))
#define SMAX VMAX
#define PMAX ((SMAX + PART_WORDS - 1) / PART_WORDS)
// Longer than the slowest rank takes between two gathers, so only a
// lost datagram triggers it.
#ifndef RESEND_MS
#define RESEND_MS 250
#endif
#ifndef LINGER_MS
#define LINGER_MS 3000
#endif
enum { T_SLICE = 4, T_ASK = 5 };

struct gather {
    u32 gid;
    int vec;                      // the vector's number, 0 until known
    s16 v[VMAX];                  // the whole vector, each rank's slice at its offset
    u8 got[TP_RANKS][PMAX];
};

static FATFS fs;
static struct netif nif;
static struct udp_pcb *pcb;
static const u8 ips[TP_RANKS][4] = {IP_RANKS};
static const u16 slices[TP_RANKS][3] = {TP_SLICES}, offsets[TP_RANKS][3] = {TP_OFFSETS};

// Rank r's slice of vector vec: its words, and where it starts. At the
// head each rank offers three words: its best logit and that token.
static int words_of(int r, int vec) { return vec == 4 ? 3 : slices[r][vec - 1]; }
static int start_of(int r, int vec) { return vec == 4 ? 3 * r : offsets[r][vec - 1]; }

// The PL moves n words from element first of the vector being gathered
// out to the gather buffer, or with into back into the core.
#define gbuf ((volatile s16 *)GBUF)
static void move(int into, int first, int n)
{
    FPGAI_BARRIER();                     // the buffer's words before the move
    Xil_Out32(REG(FPGAI_GADDR), first);
    Xil_Out32(REG(FPGAI_GMOVE), (u32)n | (into ? 1U << 16 : 0));
    while (Xil_In32(REG(FPGAI_GMOVE)) & 1)
        ;
    FPGAI_BARRIER();
}
static ip_addr_t peer[TP_RANKS];
static struct gather gb[2], *cur = &gb[0], *nxt = &gb[1];
static s16 mine[SMAX], prev[SMAX];
static u32 mine_gid, prev_gid;
static int mine_vec, mine_n, prev_vec, prev_n;
static u32 g_tok, n_asks, n_gathers;
static s16 g_best;
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

static int build(int type, const u8 *pay, int n)
{
    u8 *b = txb;
    memset(b, 0, 9);
    b[0] = 0xA5; b[1] = 0x5A; b[2] = type;
    memcpy(b + 9, pay, n);
    u32 c = crc32(b + 2, 7 + n);
    for (int k = 0; k < 4; k++)
        b[9 + n + k] = c >> (8 * k);
    return 13 + n;
}

static void send_to(int r, int n)
{
    struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, n, PBUF_RAM);
    if (!p)
        return;
    memcpy(p->payload, txb, n);
    udp_sendto(pcb, p, &peer[r], UDP_PORT);
    pbuf_free(p);
}

static void put32(u8 *b, u32 v) { b[0] = v; b[1] = v >> 8; b[2] = v >> 16; b[3] = v >> 24; }
static u32 get32(const u8 *b) { return b[0] | b[1] << 8 | b[2] << 16 | (u32)b[3] << 24; }

// A slice to rank r: gid, vector, sender, slice words, then its parts.
static void send_slice(int r, u32 gid, int vec, int n, const s16 *w)
{
    static u8 pay[14 + 2 * PART_WORDS];
    for (int off = 0; off < n; off += PART_WORDS) {
        int cnt = n - off < PART_WORDS ? n - off : PART_WORDS;
        put32(pay, gid);
        pay[4] = vec; pay[5] = TP_RANK;
        pay[6] = n; pay[7] = n >> 8; pay[8] = off; pay[9] = off >> 8;
        pay[10] = cnt; pay[11] = cnt >> 8; pay[12] = 0; pay[13] = 0;
        for (int i = 0; i < cnt; i++) {
            pay[14 + 2 * i] = (u16)w[off + i];
            pay[15 + 2 * i] = (u16)w[off + i] >> 8;
        }
        send_to(r, build(T_SLICE, pay, 14 + 2 * cnt));
    }
}

static void rx(void *arg, struct udp_pcb *u, struct pbuf *p,
               const ip_addr_t *addr, u16_t port)
{
    static u8 b[1500];
    int n = p->tot_len > (int)sizeof b ? (int)sizeof b : p->tot_len;
    pbuf_copy_partial(p, b, n, 0);
    pbuf_free(p);
    (void)arg; (void)u; (void)addr; (void)port;
    if (n < 13 + 6 || b[0] != 0xA5 || b[1] != 0x5A)
        return;
    if (crc32(b + 2, n - 6) != get32(b + n - 4))
        return;                          // dropped: its receiver asks again
    const u8 *q = b + 9;
    u32 gid = get32(q);
    int src = q[5];
    if (src >= TP_RANKS || src == TP_RANK)
        return;
    if (b[2] == T_ASK) {
        // A rank missing this rank's slice: the current one or the last.
        if (gid == mine_gid && mine_gid)
            send_slice(src, mine_gid, mine_vec, mine_n, mine);
        else if (gid == prev_gid && prev_gid)
            send_slice(src, prev_gid, prev_vec, prev_n, prev);
        return;
    }
    if (b[2] != T_SLICE || n < 13 + 14)
        return;
    struct gather *g = gid == cur->gid ? cur : gid == nxt->gid ? nxt : 0;
    int vec = q[4], sn = q[6] | q[7] << 8, off = q[8] | q[9] << 8, cnt = q[10] | q[11] << 8;
    if (!g || vec < 1 || vec > 4 || sn != words_of(src, vec) || off % PART_WORDS
        || off + cnt > sn || 13 + 14 + 2 * cnt > n || (g->vec && g->vec != vec))
        return;
    g->vec = vec;
    for (int i = 0; i < cnt; i++)
        g->v[start_of(src, vec) + off + i] = (s16)(q[14 + 2 * i] | q[15 + 2 * i] << 8);
    g->got[src][off / PART_WORDS] = 1;
}

static int complete(const struct gather *g, int r)
{
    for (int k = 0; k * PART_WORDS < words_of(r, g->vec); k++)
        if (!g->got[r][k])
            return 0;
    return 1;
}

static XTime now_ms(void)
{
    XTime t;
    XTime_GetTime(&t);
    return t / (COUNTS_PER_SECOND / 1000);
}

// One gather: this rank's slice out to every other rank, theirs in.
static void gather(int vec)
{
    int n = words_of(TP_RANK, vec), at = start_of(TP_RANK, vec);
    // The next gather becomes this one, with any parts that came early.
    struct gather *t = cur;
    cur = nxt;
    nxt = t;
    memset(nxt->got, 0, sizeof nxt->got);
    nxt->gid = cur->gid + 1;
    nxt->vec = 0;
    if (cur->vec && cur->vec != vec) {
        printf("rank %d: early parts of gather %lu disagree\r\n", TP_RANK,
               (unsigned long)cur->gid);
        memset(cur->got, 0, sizeof cur->got);
    }
    cur->vec = vec;
    memcpy(prev, mine, sizeof mine);
    prev_gid = mine_gid; prev_vec = mine_vec; prev_n = mine_n;
    if (vec == 4) {
        u32 tk = Xil_In32(REG(FPGAI_NEXT_TOK));
        mine[0] = (s16)Xil_In32(REG(FPGAI_BEST));
        mine[1] = (s16)(tk & 0xFFFF);
        mine[2] = (s16)(tk >> 16);
    } else {
        move(0, at, n);
        for (int i = 0; i < n; i++)
            mine[i] = gbuf[at + i];
    }
    mine_gid = cur->gid; mine_vec = vec; mine_n = n;
    memcpy(cur->v + at, mine, 2 * n);
    for (int r = 0; r < TP_RANKS; r++)
        if (r != TP_RANK)
            send_slice(r, mine_gid, vec, n, mine);
    XTime t0 = now_ms();
    for (;;) {
        int all = 1;
        for (int r = 0; r < TP_RANKS; r++)
            if (r != TP_RANK && !complete(cur, r))
                all = 0;
        if (all)
            break;
        xemacif_input(&nif);
        if (now_ms() - t0 > RESEND_MS) {
            static u8 ask[6];
            put32(ask, cur->gid);
            ask[4] = vec; ask[5] = TP_RANK;
            for (int r = 0; r < TP_RANKS; r++)
                if (r != TP_RANK && !complete(cur, r)) {
                    send_to(r, build(T_ASK, ask, 6));
                    n_asks++;
                }
            t0 = now_ms();
        }
    }
    if (vec == 4) {
        // The largest logit; on a tie the lower rank, whose tokens are lower.
        g_best = cur->v[0];
        g_tok = (u16)cur->v[1] | (u32)(u16)cur->v[2] << 16;
        for (int r = 1; r < TP_RANKS; r++)
            if (cur->v[3 * r] > g_best) {
                g_best = cur->v[3 * r];
                g_tok = (u16)cur->v[3 * r + 1] | (u32)(u16)cur->v[3 * r + 2] << 16;
            }
    } else {
        // The others' slices into the buffer, then the whole vector into
        // the core: this rank's own slice goes back as it came out.
        int tot = 0;
        for (int r = 0; r < TP_RANKS; r++) {
            int rn = words_of(r, vec), ra = start_of(r, vec);
            tot += rn;
            if (r != TP_RANK)
                for (int i = 0; i < rn; i++)
                    gbuf[ra + i] = cur->v[ra + i];
        }
        move(1, 0, tot);
    }
    n_gathers++;
    Xil_Out32(REG(FPGAI_GATHER), 1);
}

// One position on this rank, every gather on the way served.
static void run(u32 tok, u32 pos, int head)
{
    Xil_Out32(REG(FPGAI_TOK), tok);
    Xil_Out32(REG(FPGAI_POS), pos);
    Xil_Out32(REG(FPGAI_CTRL), 1 | (head ? 2 : 0));
    for (;;) {
        u32 g = Xil_In32(REG(FPGAI_GATHER));
        if (g & 1)
            gather((g >> 1) & 7);
        else if (Xil_In32(REG(FPGAI_STATUS)) & 2)
            break;
        else
            xemacif_input(&nif);         // a peer's early parts, or its asks
    }
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

#if TP_RANK == 0
static const u32 *vocab;
static void print_tok(u32 t)
{
    u32 n = vocab[0];
    const char *s = (const char *)(vocab + n + 2);
    if (t < n)
        printf("%.*s", (int)(vocab[t + 2] - vocab[t + 1]), s + vocab[t + 1]);
    fflush(stdout);
}
#else
#define print_tok(t) ((void)(t))
#endif

int main(void)
{
    static u32 prompt[MAX_POS + 1];
    ip_addr_t ip, mask, gw;
    unsigned char mac[6] = {0x00, 0x0a, 0x35, 0x00, 0x02, 0x10 + TP_RANK};
    init_platform();
    printf("\r\nfpgai rank %d of %d on the " FPGAI_BOARD "\r\n", TP_RANK, TP_RANKS);
    if (Xil_In32(REG(FPGAI_ID)) != FPGAI_ID_VALUE ||
        Xil_In32(REG(FPGAI_TP)) != TP_VALUE ||
        Xil_In32(REG(FPGAI_KVEND)) != (u32)KVEND) {
        printf("the bitstream is not this rank's\r\n");
        return 1;
    }
    if (f_mount(&fs, "0:/", 1) != FR_OK ||
        load("weights8.bin", WBASE, WBYTES) < 0 ||
        load("cparams.bin", CBASE, CBYTES) < 0 ||
        load("prompt.bin", (UINTPTR)prompt, 0) < 0)
        return 1;
#if TP_RANK == 0
    if (load("vocab.bin", VOCAB_BASE, 0) < 0)
        return 1;
    vocab = (const u32 *)VOCAB_BASE;
#endif
    memset((void *)KBASE, 0, KVEND - KBASE);
    Xil_DCacheFlush();

    lwip_init();
    IP4_ADDR(&ip, ips[TP_RANK][0], ips[TP_RANK][1], ips[TP_RANK][2], ips[TP_RANK][3]);
    IP4_ADDR(&mask, 255, 255, 255, 0);
    IP4_ADDR(&gw, 0, 0, 0, 0);
    for (int r = 0; r < TP_RANKS; r++)
        IP4_ADDR(&peer[r], ips[r][0], ips[r][1], ips[r][2], ips[r][3]);
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
    cur->gid = 0;
    nxt->gid = 1;
    printf("listening on UDP %d\r\n", UDP_PORT);

    XTime a = 0, b = 0;
    u32 n = prompt[0], tok = 0;
    for (u32 pos = 0; pos < n + N_GEN - 1 && pos < MAX_POS; pos++) {
        int head = pos >= n - 1;
        if (pos < n) {
            tok = prompt[pos + 1];
            print_tok(tok);
        }
        XTime_GetTime(&a);
        run(tok, pos, head);
        XTime_GetTime(&b);
        if (head) {
            tok = g_tok;
            print_tok(tok);
        }
    }
    printf("\r\nrank %d: %lu gathers, %lu asks, last step %.3f s\r\n", TP_RANK,
           (unsigned long)n_gathers, (unsigned long)n_asks,
           (double)(b - a) / COUNTS_PER_SECOND);
    // A rank that lost this one's last slice asks for it: keep answering.
    XTime t0 = now_ms();
    while (now_ms() - t0 < LINGER_MS)
        xemacif_input(&nif);
    return 0;
}
"""


def write_sd(L, work, sd, prompt, tokzr, first=True, vocab=True):
    os.makedirs(sd, exist_ok=True)
    w8 = zybo.write_w8(work)
    dst = os.path.join(sd, "weights8.bin")
    # The build's image itself, not one of the same size: write_w8 makes
    # a new file whenever the image changes, so a stale link is caught.
    same = os.path.exists(dst) and (os.path.samefile(dst, w8) or (
        os.path.getsize(dst) == os.path.getsize(w8)
        and os.path.getmtime(dst) >= os.path.getmtime(w8)))
    if not same:
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
    if vocab:
        # vocab.bin: n, n+1 offsets, then every token's bytes.
        n = max(tokzr.inv) + 1
        blob, offs = bytearray(), [0]
        for t in range(n):
            s = tokzr.inv.get(t, "")
            blob += bytes(tokzr.u2b.get(c, 63) for c in s)
            offs.append(len(blob))
        assert 4 * (n + 2) + len(blob) <= VOCAB_ROOM, "vocab.bin past JTAG_MARK"
        with open(os.path.join(sd, "vocab.bin"), "wb") as g:
            g.write(struct.pack("<%dI" % (n + 2), n, *offs) + bytes(blob))
    with open(os.path.join(sd, "prompt.bin"), "wb") as g:
        g.write(struct.pack("<%dI" % (len(prompt) + 1), len(prompt), *prompt))


README = """# fpgai on the {title}

GENERATED by `board_zybo.py`. The generated {model} sequencer, {lanes}
lanes wide, in the Zynq's PL, fed from DDR; the ARM loads the model
from the SD card and drives it a position at a time.
{role}
## Build

1. `vivado -mode batch -source build.tcl` on an x86 host with
   {board_files}; {license}. Writes `fpgai.xsa` and the reports.
   Without Vivado, `open/build_open.sh` writes the PL's bitstream.
2. In Vitis: a platform from `fpgai.xsa` (standalone, ps7_cortexa9_0),
   with `xilffs` enabled in the BSP; an empty C application with
   `sw/main.c` and `sw/fpgai_layout.h`.
3. Copy `sd/*` to a FAT32 SD card, {boot} (or make a BOOT.bin with
   the FSBL), and open {uart} at 115200.{jtag}

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

Simulated here: the register block, and {checked} {open_flow}Not run: Vivado,
so its block design, its interconnects and its timing, and the software
on real hardware. The core clock is gated with a BUFGCE while a port's
line is still on the bus; that path is the first thing to read in the
timing report.
"""


LOAD_TCL = """# GENERATED by board_zybo.py: the SD card's files over JTAG, for a
# board without a card. Start the application first: with no card it
# prints "waiting for the files over JTAG". Then, in Vitis's XSCT console
# (or xsct on its own):
#
#   source <this package>/sw/load_jtag.tcl
#
# It halts the core, writes the files from ../sd beside this script into
# DDR at the addresses fpgai_layout.h gives, writes their sizes and the
# marker the program waits for, and resumes it. JTAG is slower than the
# card: the weights take minutes.
set sd [file normalize [file join [file dirname [file normalize [info script]]] .. sd]]
if {{[catch {{targets -set -nocase -filter {{name =~ "ARM*#0"}}}}]}} {{
    connect
    targets -set -nocase -filter {{name =~ "ARM*#0"}}
}}
catch {{stop}}
foreach {{name addr}} {{{files}}} {{
    puts "$name to $addr"
    dow -data [file join $sd $name] $addr
}}
mwr 0x{size_at:08X} [list [file size [file join $sd weights8.bin]] [file size [file join $sd cparams.bin]]]
mwr 0x{mark:08X} 0x{magic:08X}
con
puts "loaded; the program goes on on the UART"
"""


def render_load_tcl(L):
    """sw/load_jtag.tcl for a single board's package."""
    jm = L.get("gend", L["end"]) + VOCAB_ROOM
    files = [("weights8.bin", L["wb"]), ("cparams.bin", L["cb"]),
             ("vocab.bin", L.get("gend", L["end"])), ("prompt.bin", jm + 64)]
    return LOAD_TCL.format(files=" ".join("%s 0x%08X" % f for f in files),
                           size_at=jm + 4, mark=jm, magic=JTAG_MAGIC)


def _readme(role, text):
    """A stage's or a rank's program talks UDP: its application comes
    from the lwIP template, not an empty one."""
    if not role:
        return text
    a = ("   with `xilffs` enabled in the BSP; an empty C application with\n"
         "   `sw/main.c` and `sw/fpgai_layout.h`.")
    assert a in text
    return text.replace(a, "   with `xilffs` enabled in the BSP; an application from the lwIP Echo\n"
                           "   Server template, its `main.c` replaced by `sw/main.c`, and\n"
                           "   `sw/fpgai_layout.h`.")


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


def model_name():
    import qwen_real
    return qwen_real.MODEL


def package(work, board="zybo_z7_20", out=None, prompt="The capital of France is",
            tokens=16, sd=True, stage=None, sim=False, latency=30, jitter=False,
            model=None, checked=None, tp=None, simulator="iverilog"):
    """Write a board's package from a qwen_full.py build. stage: one
    stage of a multi-board run, from cluster.py (its layers, its place in
    the chain, its peers' addresses); the build must be a stage build.
    tp: one rank of a split of the weights (tp.py): rank, ranks, the
    gathered vectors' lengths HHD, D and F, every rank's address, and
    slices, every rank's words of each (even shares without it); the build
    must be that rank's.
    model and checked name the model and say what was verified, for a
    build that is not one of the checkpoints CHECKED describes (spec2rtl.py
    passes its own report); such a package quotes no routed clocks."""
    pk = boards.PACKAGES[board]
    out = out or os.path.join(ROOT, pk["out"])
    L = zybo.layout(work, BASE)
    if L["N"] != pk["lanes"]:
        sys.exit("%s: its core is %d lanes wide, this build's %d; rebuild with "
                 "qwen_full.py --lanes %d" % (pk["title"], pk["lanes"], L["N"], pk["lanes"]))
    if stage:
        L["stage"] = stage
    if tp:
        if "gx_addr" not in L["W"]:
            sys.exit("%s: a rank's package needs a rank's build (qwen_full's tp)" % pk["title"])
        L["tp"] = tp
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
        f.write(zybo.render_top(L["W"], L["gn"], L["cb"], L["kb"], L["vb"], 4, L["wb"],
                                gb=L.get("gb")))
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
        f.write(OPEN_SH.format(mhz=MHZ, part=pk["part"], chip=pk["chip"], synth=_open_synth(pk),
                               title=pk["title"]))
    os.chmod(os.path.join(od, "build_open.sh"), 0o755)
    with open(os.path.join(od, "open.xdc"), "w") as f:
        f.write(render_xdc(pk, open_flow=True))
    with open(os.path.join(od, "nextpnr-xilinx-dsp.patch"), "w") as f:
        f.write(DSP_PATCH)
    with open(os.path.join(od, "prjxray-glue.patch"), "w") as f:
        f.write(XRAY_PATCH)
    sw = os.path.join(out, "sw")
    os.makedirs(sw, exist_ok=True)
    with open(os.path.join(sw, "fpgai_layout.h"), "w") as f:
        f.write(render_header(L, tokens, pk["title"], stage, tp))
    with open(os.path.join(sw, "main.c"), "w") as f:
        f.write(STAGE_C if stage else TP_C if tp else MAIN_C)
    if not stage and not tp:
        with open(os.path.join(sw, "load_jtag.tcl"), "w") as f:
            f.write(render_load_tcl(L))
    if stage:
        role = ("\nThis is stage %d of %d of a split of the layers (cluster.py, gals.py): "
                "layers %d to %d%s%s. The stages talk UDP on port %d; `sw/main.c` is "
                "built from Vitis's lwIP Echo Server template, as its header says, and "
                "the first stage prints the text.\n"
                % (stage["index"], stage["count"], stage["l0"], stage["l1"] - 1,
                   ", with the embedding" if stage["emb"] else "",
                   ", with the head" if stage["head"] else "", stage.get("port", 5000)))
    elif tp:
        role = ("\nThis is rank %d of %d of a split of the weights (tp.py): every layer, "
                "with this rank's heads and its share of every matrix's columns and "
                "of the vocabulary. The ranks gather each other's slices over UDP on "
                "port %d at %s; `sw/main.c` is built from Vitis's lwIP Echo Server "
                "template, as its header says. Every rank's SD card holds its own "
                "weights and the prompt, rank 0's also the vocabulary, and rank 0 "
                "prints the text.\n"
                % (tp["rank"], tp["ranks"], tp.get("port", 5000),
                   ", ".join(".".join(map(str, ip)) for ip in tp["ips"])))
    else:
        role = ""
    if role:
        role = "\n" + textwrap.fill(role.strip(), 72) + "\n"
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(_readme(role, README.format(
            wb=L["wb"], wbytes=L["words"] * L["N"], cb=L["cb"], cbytes=L["cn"] * 8,
            kb=L["kb"], kbytes=L["kn"] * 2 * L["N"], vb=L["vb"], vbytes=L["vn"] * 2 * L["N"],
            end=L["end"], regs=REG_BASE, mhz=MHZ, title=pk["title"], lanes=L["N"],
            model=model or CHECKED[model_name()][0],
            checked=checked if checked is not None else CHECKED[model_name()][1],
            board_files=pk["board_files"], license=pk["license"],
            boot=pk["boot"], uart=pk["uart"],
            open_flow=("The open flow in `open/` places and routes it on this part: "
                       "%.1f MHz core, %.1f MHz bus, both past %d MHz. "
                       % (pk["open_flow_mhz"][model_name()] + (MHZ,))
                       if checked is None and model_name() in pk.get("open_flow_mhz", {})
                       else ""),
            role=role,
            jtag="" if stage or tp else (
                "\n   Without a card: run the application, which waits, then\n"
                "   `source sw/load_jtag.tcl` in Vitis's XSCT console writes the\n"
                "   same files into DDR over JTAG (minutes for the weights)."),
            regtab="\n".join("| 0x%02X | %s | %s |" % r
                              for r in REGS + (STAGE_REGS if stage else TP_REGS if tp
                                               else [])))))
    if sd:
        import qwen_real
        tk = qwen_real.Tokenizer()
        write_sd(L, work, os.path.join(out, "sd"), tk.encode(prompt), tk,
                 first=not stage or stage["emb"], vocab=not tp or tp["rank"] == 0)
    print("package in", out)
    if sim:
        zybo.write_w8(work)                 # the DDR model's weight image
        simd = os.path.join(out, "sim")
        os.makedirs(simd, exist_ok=True)
        with open(os.path.join(simd, "tb_fpgai_zybo.v"), "w") as f:
            f.write(tb_text(L, 4, latency, int(jitter)))
        # The DDR model reads weights8.bin and cparams.hex from the build.
        shutil.copyfile(os.path.join(simd, "tb_fpgai_zybo.v"),
                        os.path.join(work, "tb_fpgai_zybo.v"))
        for f in ("qwen_zybo.v", "fpgai_zybo.v"):
            shutil.copyfile(os.path.join(rtl, f), os.path.join(work, "bz_" + f))
        zybo.simulate(work, "tb_fpgai_zybo.v", ["bz_qwen_zybo.v", "bz_fpgai_zybo.v"], simulator)


if __name__ == "__main__":
    main()
