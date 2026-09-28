"""The Qwen sequencer as a Zybo Z7-20 package.

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
    return """// GENERATED by board_zybo.py: do not edit by hand.
// The Qwen sequencer for the Zybo Z7-20's PL. The ARM drives it through
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


BUILD = """# GENERATED by board_zybo.py. Build the Zybo Z7-20 bitstream and platform:
#   vivado -mode batch -source build.tcl
# Needs an x86 Vivado (2020.2 or later) with Digilent's board files, for
# the PS7's DDR and MIO preset. Writes fpgai.xsa for Vitis (see sw/).
create_project -force fpgai ./vivado -part xc7z020clg400-1
if {{[catch {{set_property board_part digilentinc.com:zybo-z7-20:part0:1.1 [current_project]}}]}} {{
  puts "ERROR: install Digilent's board files (github.com/Digilent/vivado-boards)"
  exit 1
}}
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
create_bd_port -dir O busy_led
connect_bd_net [get_bd_pins fpgai/busy_led] [get_bd_ports busy_led]
assign_bd_address
set seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps7/Data] -filter {{NAME =~ "*fpgai*"}}]
set_property offset 0x{regs:08X} $seg
set_property range 64K $seg
validate_bd_design
save_bd_design

set w [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $w
set_property top system_wrapper [current_fileset]
add_files -fileset constrs_1 ./zybo.xdc
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
open_run impl_1
report_timing_summary -file timing.rpt
report_utilization -file utilization.rpt
write_hw_platform -fixed -include_bit -force ./fpgai.xsa
puts "platform: ./fpgai.xsa"
"""

XDC = """# GENERATED by board_zybo.py. The Zybo Z7-20's LD0 shows the core busy.
set_property -dict {{PACKAGE_PIN M14 IOSTANDARD LVCMOS33}} [get_ports busy_led]
"""

REG_BASE = 0x43C00000


def render_header(L, n_gen):
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
              "#endif", ""]
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
    printf("\r\nfpgai: Qwen2.5-0.5B on the Zybo PL\r\n");
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
    for (int g = 0; g < N_GEN && pos < MAX_POS; g++, pos++) {
        print_tok(tok);
        XTime_GetTime(&a);
        tok = step(tok, pos, 1);
        XTime_GetTime(&b);
    }
    print_tok(tok);
    // XTime counts at half the CPU clock.
    printf("\r\nlast step: %lu core cycles, %lu bus cycles, %.2f s\r\n",
           (unsigned long)Xil_In32(REG(FPGAI_CORE_CYCLES)),
           (unsigned long)Xil_In32(REG(FPGAI_BUS_CYCLES)),
           (double)(b - a) / COUNTS_PER_SECOND);
    return 0;
}
"""


def write_sd(L, work, sd, prompt, tokzr):
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


README = """# fpgai on the Zybo Z7-20

GENERATED by `board_zybo.py`. The generated Qwen2.5-0.5B sequencer in
the Zynq's PL, fed from DDR; the ARM loads the model from the SD card
and drives it a position at a time.

## Build

1. `vivado -mode batch -source build.tcl` on an x86 host with Digilent's
   board files. Writes `fpgai.xsa` and the reports.
2. In Vitis: a platform from `fpgai.xsa` (standalone, ps7_cortexa9_0),
   with `xilffs` enabled in the BSP; an empty C application with
   `sw/main.c` and `sw/fpgai_layout.h`.
3. Copy `sd/*` to a FAT32 SD card, boot the board from JTAG (or make a
   BOOT.bin with the FSBL), and open the USB-UART at 115200.

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
    ap.add_argument("--out", default=os.path.join(ROOT, "board_zybo"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--latency", type=int, default=30)
    ap.add_argument("--sim", action="store_true",
                    help="simulate the wrapper on the build's own steps")
    ap.add_argument("--no-sd", action="store_true")
    ap.add_argument("--jitter", action="store_true",
                    help="DDR ports stall and gap at random in the simulation")
    a = ap.parse_args()
    work, out = a.work, a.out
    L = zybo.layout(work, BASE)
    rtl = os.path.join(out, "rtl")
    os.makedirs(rtl, exist_ok=True)
    srcs = [f for f in os.listdir(work) if f.endswith(".v")
            and not f.startswith("tb_") and f != "qwen_zybo.v"]
    for f in srcs:
        shutil.copyfile(os.path.join(work, f), os.path.join(rtl, f))
    shutil.copyfile(os.path.join(work, "gains.hex"), os.path.join(rtl, "gains.hex"))
    with open(os.path.join(rtl, "qwen_zybo.v"), "w") as f:
        f.write(zybo.render_top(L["W"], L["gn"], L["cb"], L["kb"], L["vb"], 4, L["wb"]))
    with open(os.path.join(rtl, "fpgai_zybo.v"), "w") as f:
        f.write(render_wrapper(L))
    with open(os.path.join(out, "build.tcl"), "w") as f:
        f.write(BUILD.format(mhz=MHZ, regs=REG_BASE))
    with open(os.path.join(out, "zybo.xdc"), "w") as f:
        f.write(XDC.format())
    sw = os.path.join(out, "sw")
    os.makedirs(sw, exist_ok=True)
    with open(os.path.join(sw, "fpgai_layout.h"), "w") as f:
        f.write(render_header(L, a.tokens))
    with open(os.path.join(sw, "main.c"), "w") as f:
        f.write(MAIN_C)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(
            wb=L["wb"], wbytes=L["words"] * 16, cb=L["cb"], cbytes=L["cn"] * 8,
            kb=L["kb"], kbytes=L["kn"] * 32, vb=L["vb"], vbytes=L["vn"] * 32,
            end=L["end"], regs=REG_BASE, mhz=MHZ,
            regtab="\n".join("| 0x%02X | %s | %s |" % r for r in REGS)))
    if not a.no_sd:
        import qwen_real
        tk = qwen_real.Tokenizer()
        write_sd(L, work, os.path.join(out, "sd"), tk.encode(a.prompt), tk)
    print("package in", out)
    if a.sim:
        simd = os.path.join(out, "sim")
        os.makedirs(simd, exist_ok=True)
        with open(os.path.join(simd, "tb_fpgai_zybo.v"), "w") as f:
            f.write(tb_text(L, 4, a.latency, int(a.jitter)))
        # The DDR model reads weights8.bin and cparams.hex from the build.
        shutil.copyfile(os.path.join(simd, "tb_fpgai_zybo.v"),
                        os.path.join(work, "tb_fpgai_zybo.v"))
        for f in ("qwen_zybo.v", "fpgai_zybo.v"):
            shutil.copyfile(os.path.join(rtl, f), os.path.join(work, "bz_" + f))
        zybo.simulate(work, "tb_fpgai_zybo.v", ["bz_qwen_zybo.v", "bz_fpgai_zybo.v"])


if __name__ == "__main__":
    main()
