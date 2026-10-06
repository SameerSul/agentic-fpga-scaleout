"""A board package's ARM program on the netlist its bitstream is made from.

cosim.py runs a package's sw/main.c against its RTL. This runs it
against what Yosys makes of that RTL for the open flow: the synthesis
commands of the package's own open/build_open.sh, with the register
block's wrapper (fpgai_zybo) as the top instead of the PS7's, written out
as a netlist of Xilinx primitives and simulated in the same harness. A
step's token, logit and KV cache then say whether synthesis kept the
design's meaning, which the RTL's simulations cannot.

The first ZC706 bitstream answered token 0 at every step; this harness
answers the same, and the cause was Yosys's (board_zybo._open_synth).

Primitives are Yosys's own simulation models (share/yosys/xilinx/
cells_sim.v), except two: its RAMB36E1 and RAMB18E1 carry timing only,
with outputs nothing drives, so a netlist read 0 from every block RAM;
here they are behavioral (TDP and SDP, the widths, the write modes, byte
enables, INIT and INITP, the output register), and BUFGCE gates as the
silicon does, its enable passing while the clock is low.

Run: python3 gatesim.py board_zc706_qwen3        the diagnostics' step,
     against the RTL's (sdboot/diag.c), K and V layer by layer
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _cells_sim():
    yosys = shutil.which("yosys")
    share = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(yosys))), "share",
                         "yosys", "xilinx", "cells_sim.v")
    if not os.path.exists(share):
        share = subprocess.run(["yosys-config", "--datdir"], capture_output=True,
                               text=True).stdout.strip() + "/xilinx/cells_sim.v"
    return open(share).read()


def _bram(name, abits, nd, np_):
    """A behavioral RAMB36E1 (abits 15) or RAMB18E1 (14), nd data and np_
    parity bits, INIT words of 256 bits."""
    ni, npi = nd // 256, np_ // 256
    dw = 32 if name == "RAMB36E1" else 16
    pw, web, wa = dw // 8, (8 if name == "RAMB36E1" else 4), (4 if name == "RAMB36E1" else 2)
    ports = ["output CASCADEOUTA", "output CASCADEOUTB", "output [7:0] ECCPARITY",
             "output [8:0] RDADDRECC", "output SBITERR", "output DBITERR", "input CASCADEINA",
             "input CASCADEINB", "input INJECTDBITERR", "input INJECTSBITERR"] \
        if name == "RAMB36E1" else []
    ports += ["output [%d:0] DOADO" % (dw - 1), "output [%d:0] DOBDO" % (dw - 1),
              "output [%d:0] DOPADOP" % (pw - 1), "output [%d:0] DOPBDOP" % (pw - 1),
              "input ENARDEN", "input CLKARDCLK", "input RSTRAMARSTRAM", "input RSTREGARSTREG",
              "input REGCEAREGCE", "input ENBWREN", "input CLKBWRCLK", "input RSTRAMB",
              "input RSTREGB", "input REGCEB", "input [%d:0] ADDRARDADDR" % (15 if name == "RAMB36E1" else 13),
              "input [%d:0] ADDRBWRADDR" % (15 if name == "RAMB36E1" else 13),
              "input [%d:0] DIADI" % (dw - 1), "input [%d:0] DIBDI" % (dw - 1),
              "input [%d:0] DIPADIP" % (pw - 1), "input [%d:0] DIPBDIP" % (pw - 1),
              "input [%d:0] WEA" % (wa - 1), "input [%d:0] WEBWE" % (web - 1)]
    L = ["module %s (\n  %s);" % (name, ",\n  ".join(ports))]
    L += ["  parameter [255:0] INIT_%02X = 256'h0;" % i for i in range(ni)]
    L += ["  parameter [255:0] INITP_%02X = 256'h0;" % i for i in range(npi)]
    for p, v in (("RAM_MODE", '"TDP"'), ("READ_WIDTH_A", "0"), ("READ_WIDTH_B", "0"),
                 ("WRITE_WIDTH_A", "0"), ("WRITE_WIDTH_B", "0"), ("DOA_REG", "0"),
                 ("DOB_REG", "0"), ("WRITE_MODE_A", '"WRITE_FIRST"'),
                 ("WRITE_MODE_B", '"WRITE_FIRST"'), ("SRVAL_A", "0"), ("SRVAL_B", "0"),
                 ("INIT_A", "0"), ("INIT_B", "0"), ("RAM_EXTENSION_A", '"NONE"'),
                 ("RAM_EXTENSION_B", '"NONE"'), ("RDADDR_COLLISION_HWCONFIG", '"DELAYED_WRITE"'),
                 ("SIM_COLLISION_CHECK", '"ALL"'), ("SIM_DEVICE", '"7SERIES"'),
                 ("EN_ECC_READ", '"FALSE"'), ("EN_ECC_WRITE", '"FALSE"'), ("INIT_FILE", '"NONE"'),
                 ("IS_CLKARDCLK_INVERTED", "0"), ("IS_CLKBWRCLK_INVERTED", "0"),
                 ("IS_ENARDEN_INVERTED", "0"), ("IS_ENBWREN_INVERTED", "0"),
                 ("IS_RSTRAMARSTRAM_INVERTED", "0"), ("IS_RSTRAMB_INVERTED", "0"),
                 ("IS_RSTREGARSTREG_INVERTED", "0"), ("IS_RSTREGB_INVERTED", "0")):
        L.append("  parameter %s = %s;" % (p, v))
    L.append("  reg [%d:0] md; reg [%d:0] mp;" % (nd - 1, np_ - 1))
    L.append("  initial begin")
    L += ["    md[%d +: 256] = INIT_%02X;" % (256 * i, i) for i in range(ni)]
    L += ["    mp[%d +: 256] = INITP_%02X;" % (256 * i, i) for i in range(npi)]
    L.append("  end")
    W2 = 2 * dw
    L.append("""  // A width's data and parity bits; an entry is the address's bits above
  // log2 of its data width. In SDP the read port is A's address, clock and
  // enable at READ_WIDTH_A, the write port B's at WRITE_WIDTH_B, and the
  // word spans both ports' data pins, A's low.
  function integer dbits(input integer wd); dbits = wd >= 9 ? wd - wd / 9 : wd; endfunction
  function integer pbits(input integer wd); pbits = wd >= 9 ? wd / 9 : 0; endfunction
  function integer lg(input integer v); integer k; begin lg = 0; for (k = 1; k < v; k = k * 2) lg = lg + 1; end endfunction
  localparam SDP = RAM_MODE == "SDP";
  localparam RDA = dbits(READ_WIDTH_A), RPA = pbits(READ_WIDTH_A);
  localparam RDB = dbits(READ_WIDTH_B), RPB = pbits(READ_WIDTH_B);
  localparam WDA = dbits(WRITE_WIDTH_A), WPA = pbits(WRITE_WIDTH_A);
  localparam WDB = dbits(WRITE_WIDTH_B), WPB = pbits(WRITE_WIDTH_B);
  wire [{ab}:0] aa = ADDRARDADDR[{ab}:0], ab = ADDRBWRADDR[{ab}:0];
  wire clka = CLKARDCLK ^ IS_CLKARDCLK_INVERTED[0], clkb = CLKBWRCLK ^ IS_CLKBWRCLK_INVERTED[0];
  wire ena = ENARDEN ^ IS_ENARDEN_INVERTED[0], enb = ENBWREN ^ IS_ENBWREN_INVERTED[0];
  wire rsta = RSTRAMARSTRAM ^ IS_RSTRAMARSTRAM_INVERTED[0], rstb = RSTRAMB ^ IS_RSTRAMB_INVERTED[0];
  wire [{w2}:0] wdat = {{DIBDI, DIADI}};
  wire [{p2}:0] wpar = {{DIPBDIP, DIPADIP}};
  reg [{w2}:0] oa, ob, ra, rb; reg [{p2}:0] opa, opb, rpa, rpb;
  initial begin oa = INIT_A; ob = INIT_B; ra = INIT_A; rb = INIT_B; opa = 0; opb = 0; rpa = 0; rpb = 0; end
  integer k;
  // Reads take the word before this edge's writes (READ_FIRST); NO_CHANGE
  // keeps the output while the port writes.
  always @(posedge clka) if (ena) begin : porta
    reg [{w2}:0] d; reg [{p2}:0] p;
    if (READ_WIDTH_A != 0) begin
      d = 0; p = 0;
      for (k = 0; k < RDA; k = k + 1) d[k] = md[(aa >> lg(RDA)) * RDA + k];
      for (k = 0; k < RPA; k = k + 1) p[k] = mp[(aa >> lg(RDA)) * RPA + k];
      if (rsta) begin oa <= SRVAL_A; opa <= 0; end
      else if (!(WRITE_MODE_A == "NO_CHANGE" && !SDP && |WEA)) begin oa <= d; opa <= p; end
    end
    if (WRITE_WIDTH_A != 0 && !SDP) begin
      for (k = 0; k < WDA; k = k + 1) if (WEA[(WDA >= 8 ? k / 8 : 0) % {wa}]) md[(aa >> lg(WDA)) * WDA + k] <= DIADI[k];
      for (k = 0; k < WPA; k = k + 1) if (WEA[k % {wa}]) mp[(aa >> lg(WDA)) * WPA + k] <= DIPADIP[k];
    end
  end
  always @(posedge clkb) if (enb) begin : portb
    reg [{w2}:0] d; reg [{p2}:0] p;
    if (READ_WIDTH_B != 0 && !SDP) begin
      d = 0; p = 0;
      for (k = 0; k < RDB; k = k + 1) d[k] = md[(ab >> lg(RDB)) * RDB + k];
      for (k = 0; k < RPB; k = k + 1) p[k] = mp[(ab >> lg(RDB)) * RPB + k];
      if (rstb) begin ob <= SRVAL_B; opb <= 0; end
      else if (!(WRITE_MODE_B == "NO_CHANGE" && |WEBWE)) begin ob <= d; opb <= p; end
    end
    if (WRITE_WIDTH_B != 0) begin
      for (k = 0; k < WDB; k = k + 1) if (WEBWE[(WDB >= 8 ? k / 8 : 0) % {web}]) md[(ab >> lg(WDB)) * WDB + k] <= SDP ? wdat[k] : DIBDI[k % {dw}];
      for (k = 0; k < WPB; k = k + 1) if (WEBWE[k % {web}]) mp[(ab >> lg(WDB)) * WPB + k] <= SDP ? wpar[k] : DIPBDIP[k % {pw}];
    end
  end
  always @(posedge clka) if (REGCEAREGCE) begin ra <= oa; rpa <= opa; end
  always @(posedge clkb) if (REGCEB) begin rb <= ob; rpb <= opb; end
  wire [{w2}:0] qa = DOA_REG ? ra : oa, qb = DOB_REG ? rb : ob;
  wire [{p2}:0] qpa = DOA_REG ? rpa : opa, qpb = DOB_REG ? rpb : opb;
  assign DOADO = qa[{dw1}:0];
  assign DOBDO = SDP ? qa[{w2}:{dw}] : qb[{dw1}:0];
  assign DOPADOP = qpa[{pw1}:0];
  assign DOPBDOP = SDP ? qpa[{p2}:{pw}] : qpb[{pw1}:0];""".format(
        ab=abits - 1, w2=W2 - 1, p2=2 * pw - 1, wa=wa, web=web, dw=dw, pw=pw, dw1=dw - 1,
        pw1=pw - 1))
    if name == "RAMB36E1":
        L.append("  assign CASCADEOUTA = 0; assign CASCADEOUTB = 0; assign ECCPARITY = 0;\n"
                 "  assign RDADDRECC = 0; assign SBITERR = 0; assign DBITERR = 0;")
    L.append("endmodule")
    return "\n".join(L) + "\n"


BUFGCE = """// BUFGCE as the silicon has it: CE passes while I is low, so an enable
// that changes on I's falling edge gates the next rising edge.
module BUFGCE(output O, input CE, input I);
  reg en = 1'b1;
  always @* if (!I) en = CE;
  assign O = I & en;
endmodule
"""


def primitives(netlist):
    """The simulation models of the primitives netlist instantiates."""
    used = set(re.findall(r"^\s+([A-Z][A-Z0-9_]+)\s+(?:#\(|\\?\S+\s*\()", netlist, re.M))
    out = ["`timescale 1ns/1ps"]
    for m in re.finditer(r"^module\s+(\w+)\b.*?^endmodule", _cells_sim(), re.S | re.M):
        if m.group(1) in used - {"RAMB36E1", "RAMB18E1", "BUFGCE"}:
            out.append(m.group(0))
    out += [_bram("RAMB36E1", 15, 32768, 4096), _bram("RAMB18E1", 14, 16384, 2048), BUFGCE]
    return "\n\n".join(out) + "\n"


def synth_commands(pkg):
    """The synthesis commands of the package's open/build_open.sh, with the
    register block's wrapper as the top."""
    sh = open(os.path.join(pkg, "open", "build_open.sh")).read()
    m = re.search(r'yosys -q -l yosys\.log -p "read_verilog [^;]*;(.*?); write_json', sh, re.S)
    cmds = re.sub(r"\\\n", " ", m.group(1)).replace("\\$", "$")
    return re.sub(r"-top fpgai_ps7\b", "-top fpgai_zybo", " ".join(cmds.split()))


def netlist(pkg, work, log=print):
    """The package's RTL through its open flow's synthesis, as Verilog."""
    syn = os.path.join(work, "syn")
    shutil.rmtree(syn, ignore_errors=True)
    os.makedirs(syn)
    rtl = os.path.join(pkg, "rtl")
    for f in os.listdir(rtl):
        shutil.copyfile(os.path.join(rtl, f), os.path.join(syn, f))
    srcs = " ".join(sorted(f for f in os.listdir(syn) if f.endswith(".v")))
    script = "read_verilog %s; %s; write_verilog -noattr netlist.v" % (srcs, synth_commands(pkg))
    r = subprocess.run(["yosys", "-q", "-l", "yosys.log", "-p", script], cwd=syn,
                       capture_output=True, text=True, errors="replace")
    if r.returncode:
        raise RuntimeError("yosys failed:\n" + r.stderr[-3000:] + r.stdout[-3000:])
    log("  netlist %s" % os.path.join(syn, "netlist.v"))
    return os.path.join(syn, "netlist.v")


def gate_package(pkg, work, main=None, log=print):
    """A package whose rtl/ is its netlist and the primitives' models, its
    sw/ the package's (or main in place of sw/main.c), for cosim.build."""
    net = netlist(pkg, work, log)
    gp = os.path.join(work, "pkg")
    shutil.rmtree(gp, ignore_errors=True)
    os.makedirs(os.path.join(gp, "rtl"))
    shutil.copytree(os.path.join(pkg, "sw"), os.path.join(gp, "sw"))
    if main:
        shutil.copyfile(main, os.path.join(gp, "sw", "main.c"))
    text = open(net).read()
    shutil.copyfile(net, os.path.join(gp, "rtl", "netlist.v"))
    with open(os.path.join(gp, "rtl", "primitives.v"), "w") as f:
        f.write(primitives(text))
    shutil.copyfile(os.path.join(pkg, "rtl", "gains.hex"), os.path.join(gp, "rtl", "gains.hex"))
    return gp


def diag_compare(pkg, work, log=print):
    """sdboot/diag.c's step on the RTL and on the netlist: the token, the
    logit and every K and V value, layer by layer."""
    import sdboot
    rtl = sdboot.diag_record(pkg, os.path.join(work, "rtl"), log)
    gp = gate_package(pkg, os.path.join(work, "gate"), os.path.join(ROOT, "sdboot", "diag.c"), log)
    gate = sdboot.diag_record(gp, os.path.join(work, "gate", "run"), log, sd=os.path.join(pkg, "sd"),
                              layers=rtl["layers"])
    a, b = dict(rtl["w"]), dict(gate["w"])
    nl, half = rtl["layers"], rtl["vbase"]
    per = half // nl
    rows = []
    for o, v in a.items():
        g = (o // per) if o < half else nl + (o - half) // per
        rows.append((g, b.get(o) == v))
    bad = sorted({g for g, ok in rows if not ok})
    return {"rtl": (rtl["next"], rtl["best"]), "gate": (gate["next"], gate["best"]),
            "values": len(a), "same": sum(ok for _, ok in rows),
            "layers_differ": ["%s%d" % ("KV"[g >= nl], g % nl) for g in bad]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("package")
    ap.add_argument("--work", default=os.path.join(ROOT, "build_gatesim"))
    a = ap.parse_args()
    r = diag_compare(os.path.abspath(a.package), os.path.abspath(a.work))
    print("RTL: token %d, logit %d; netlist: token %d, logit %d" % (r["rtl"] + r["gate"]))
    print("%d of %d K and V values the same%s" % (
        r["same"], r["values"],
        "" if not r["layers_differ"] else "; differ in " + " ".join(r["layers_differ"])))
    sys.exit(0 if r["rtl"] == r["gate"] and r["same"] == r["values"] else 1)


if __name__ == "__main__":
    main()
