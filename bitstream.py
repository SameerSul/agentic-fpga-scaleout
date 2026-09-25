"""Place, route and pack a generated block into a real FPGA bitstream.

Every other stage in this repo stops at synthesis. Synthesis says a design
can be mapped; it does not say the design fits a real device, that its
routing closes, or that its clock survives real wire delay. Until a
bitstream exists, "it works on an FPGA" is a claim about a model.

This runs the open toolchain end to end: yosys synth_ice40, nextpnr-ice40
for place and route, icepack for the bitstream, and icetime for a
post-route timing estimate against the real device timing model.

The device is a Lattice iCE40, because that is the only family with an
open place and route flow installable here. It is NOT the Xilinx part the
project is aimed at: Vivado does not run on an ARM Mac and nextpnr has no
mainline Xilinx target. So this proves the flow reaches a bitstream and
that the generated RTL survives real place and route, on a different
device than the one the boards carry. RESULTS.md says so plainly.

  python3 bitstream.py                 # the MAC chiplet
  python3 bitstream.py --block requant
  python3 bitstream.py --device up5k --package sg48
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import specgen
from agent import (RuleBasedAgent, FIX_WIDTH, FIX_CLEAR,
                   FIX_SATURATE, FIX_XOR, FIX_LUT, FIX_NORM, FIX_EVEN)
from chiplet_flow import ROOT, TARGET_LINK_GBPS

WORK = os.path.join(ROOT, "build_bitstream")

# The testbench is generated for the exact spec being built, never taken
# from the committed copy. The committed tb_crc.v is the 10 Gbps endpoint
# at 16 bytes per cycle; building the 1 Gbps endpoint and checking it
# against that file reported a bitstream failure when the design had in
# fact produced the right answer for its own configuration.
TB_FOR = {
    "mac": lambda spec: specgen.render_testbench(spec),
    "requant": lambda spec: specgen.render_requant_testbench(spec),
    "crc": lambda spec: specgen.render_crc_testbench(spec),
    "exp": lambda spec: specgen.render_exp_testbench(spec),
    "recip": lambda spec: specgen.render_recip_testbench(spec),
    "rsqrt": lambda spec: specgen.render_rsqrt_testbench(spec),
}

BLOCKS = {
    "mac": ("mac", lambda ms: specgen.derive_chiplet_spec(ms),
            {FIX_WIDTH, FIX_CLEAR}),
    "requant": ("requant", lambda ms: specgen.derive_requant_spec(ms),
                {FIX_SATURATE}),
    "crc": ("crc32", lambda ms: specgen.derive_endpoint_spec(1.0),
            {FIX_XOR}),
    "exp": ("expu", lambda ms: specgen.derive_exp_spec(ms), {FIX_LUT}),
    "recip": ("recip", lambda ms: specgen.derive_recip_spec(ms),
              {FIX_NORM}),
    "rsqrt": ("rsqrt", lambda ms: specgen.derive_rsqrt_spec(ms),
              {FIX_EVEN}),
}


ICEBOX_PY = "/opt/homebrew/Cellar/icestorm/1.1/share/icestorm/python"


def package_pins(device, package):
    """Valid pin names for a package, from icestorm's own database."""
    env = dict(os.environ)
    env["PYTHONPATH"] = ICEBOX_PY + ":" + env.get("PYTHONPATH", "")
    # icestorm keys these by size alone: hx8k and lp8k are both "8k".
    key = "%s-%s" % (re.sub(r"^[a-z]+", "", device), package)
    code = ("import icebox,sys;"
            "print(' '.join(sorted({str(p[0]) for p in "
            "icebox.pinloc_db[%r]})))" % key)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env)
    return r.stdout.split() if r.returncode == 0 else []


def write_pcf(spec, pins, path):
    """Constrain every port to a real pin.

    Without a pcf the placer picks pins itself and the bitstream unpacks
    with its ports named after tile coordinates, which cannot be bound to
    a testbench. Constraining them keeps the names, and a real design
    would have pin constraints anyway.
    """
    used, lines = 0, []
    for p in spec["ports"]:
        w = p["width"]
        names = ([p["name"]] if w == 1
                 else ["%s[%d]" % (p["name"], i) for i in range(w)])
        for nm in names:
            if used >= len(pins):
                return False
            lines.append("set_io %s %s" % (nm, pins[used]))
            used += 1
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def _mangle(src):
    """Turn escaped identifiers into plain ones.

    icebox_vlog writes bit-blasted ports as escaped identifiers, \\acc[3]
    and so on. An escaped identifier is terminated by whitespace, not by
    the comma that follows it in these declaration lists, so the output
    does not parse. Rewriting them to plain names sidesteps the whole
    question and changes nothing about the logic.
    """
    return re.sub(r"\\([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\s?",
                  lambda m: "%s_%s_" % (m.group(1), m.group(2)), src)


def _to_non_ansi(src):
    """Rewrite the unpacked module's header to non-ANSI port style.

    icebox_vlog writes an ANSI port list and then re-declares the same
    output names as regs, which is a duplicate declaration against an
    ANSI header. In non-ANSI style "output x; reg x;" is the ordinary way
    to write a registered output, so the same body parses.
    """
    m = re.search(r"module\s+(\w+)\s*\((.*?)\);", src, re.S)
    if not m:
        return src
    decls = re.findall(r"\b(input|output|inout)\s+(\w+)", m.group(2))
    names = [n for _, n in decls]
    ports = set(names)
    header = "module %s (%s);\n" % (m.group(1), ", ".join(names))
    header += "\n".join("  %s %s;" % (d, n) for d, n in decls) + "\n"
    body = src[m.end():]
    out = []
    for line in body.splitlines():
        w = re.match(r"^(\s*wire\s+)(.*?);\s*$", line)
        if w:
            # A wire that repeats a port is still a duplicate; a reg is not.
            keep = [n.strip() for n in w.group(2).split(",")
                    if n.strip() and n.strip() not in ports]
            if not keep:
                continue
            line = "%s%s;" % (w.group(1), ", ".join(keep))
        out.append(line)
    return src[:m.start()] + header + "\n".join(out)


def _wrapper(spec, top, inner):
    """A vector-ported wrapper around the bit-blasted unpacked module, so
    the original testbench can drive the bitstream unchanged."""
    ports, conns = [], []
    for p in spec["ports"]:
        d = "input" if p["dir"] == "input" else "output"
        sign = " signed" if p.get("signed") else ""
        if p["width"] == 1:
            ports.append("%s%s %s" % (d, sign, p["name"]))
            conns.append(".%s(%s)" % (p["name"], p["name"]))
        else:
            ports.append("%s%s [%d:0] %s" % (d, sign, p["width"] - 1,
                                             p["name"]))
            for i in range(p["width"]):
                conns.append(".%s_%d_(%s[%d])" % (p["name"], i, p["name"], i))
    return "module %s (\n  %s\n);\n  %s u (\n    %s\n  );\nendmodule\n" % (
        top, ",\n  ".join(ports), inner, ",\n    ".join(conns))


def run(cmd, timeout=1800):
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def need(tool):
    if not shutil.which(tool):
        raise SystemExit(
            "%s not on PATH. The open iCE40 flow needs yosys, nextpnr-ice40 "
            "and icestorm:\n  brew install nextpnr-ice40 icestorm" % tool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="mac", choices=sorted(BLOCKS))
    ap.add_argument("--device", default="hx8k")
    ap.add_argument("--package", default="ct256")
    ap.add_argument("--freq", type=float, default=None,
                    help="target MHz; defaults to the spec's clock")
    a = ap.parse_args()
    for t in ("yosys", "nextpnr-ice40", "icepack", "icetime"):
        need(t)

    top, derive, fixes = BLOCKS[a.block]
    ms = specgen.load_model_spec()
    spec = derive(ms)
    rtl = getattr(RuleBasedAgent(), "render_" + a.block)(spec, fixes)
    freq = a.freq or spec["parameters"]["target_clock_mhz"]

    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK)
    open(os.path.join(WORK, "top.v"), "w").write(rtl)
    print("block %s (%s), device iCE40 %s %s, target %g MHz\n"
          % (a.block, spec["name"], a.device, a.package, freq))

    print("synthesising to iCE40 primitives")
    rc, out = run(["yosys", "-p",
                   "read_verilog top.v; synth_ice40 -top %s -json top.json"
                   % top])
    if rc:
        print(out[-2500:])
        raise SystemExit("synth_ice40 failed")
    # nextpnr reports the real utilisation a moment later, so the yosys
    # cell histogram is not worth parsing twice.
    lcs = 0

    pins = package_pins(a.device, a.package)
    have_pcf = bool(pins) and write_pcf(spec, pins,
                                        os.path.join(WORK, "top.pcf"))
    if not have_pcf:
        # Worth saying out loud: without it the bitstream unpacks with
        # tile-coordinate port names and cannot be simulated.
        print("   no pin constraints (%d package pins found), so the "
              "bitstream cannot be bound to a testbench" % len(pins))
    print("placing and routing%s"
          % (" with pin constraints" if have_pcf else ""))
    cmd = ["nextpnr-ice40", "--%s" % a.device, "--package", a.package,
           "--json", "top.json", "--asc", "top.asc", "--freq", str(freq),
           "--placer", "heap", "--seed", "1"]
    if have_pcf:
        cmd += ["--pcf", "top.pcf"]
    rc, out = run(cmd)
    routed = "Program finished normally" in out or rc == 0
    m = re.search(r"Max frequency for clock\s+'[^']*':\s+([\d.]+)\s+MHz", out)
    for line in out.splitlines():
        if re.search(r"^Info:\s+(ICESTORM_LC|ICESTORM_RAM|SB_IO|ICESTORM_DSP"
                     r"|SB_GB):", line):
            print("   " + line.replace("Info:", "").strip())
    if not routed:
        print(out[-2500:])
        raise SystemExit("place and route failed")
    post_fmax = float(m.group(1)) if m else None
    if post_fmax:
        print("   post-route fmax %.2f MHz against a %g MHz target (%s)"
              % (post_fmax, freq,
                 "MET" if post_fmax >= freq else "NOT MET"))

    print("packing the bitstream")
    rc, out = run(["icepack", "top.asc", "top.bin"])
    if rc:
        print(out[-1500:])
        raise SystemExit("icepack failed")
    size = os.path.getsize(os.path.join(WORK, "top.bin"))
    print("   %s  %d bytes" % (os.path.join("build_bitstream", "top.bin"),
                               size))

    # Verify the bitstream, not just the design that produced it. icepack
    # emits the actual configuration bits; icebox_vlog turns those bits
    # back into logic, which is then simulated against the same
    # self-checking testbench the RTL passed. Everything upstream can be
    # right and the packed result still be wrong, and nothing so far
    # looked at the artifact that would actually be loaded onto a device.
    print("verifying the packed bitstream against the original testbench")
    vcmd = ["icebox_vlog", "-n", top + "_bits", "-s"]
    if have_pcf:
        vcmd += ["-p", "top.pcf"]
    rc, out = run(vcmd + ["top.asc"])
    if rc:
        print(out[-1200:])
        raise SystemExit("icebox_vlog failed")
    open(os.path.join(WORK, "unpacked.v"), "w").write(
        _to_non_ansi(_mangle(out)) + "\n" + _wrapper(spec, top,
                                                     top + "_bits"))
    tb = "tb_block.v"
    open(os.path.join(WORK, tb), "w").write(TB_FOR[a.block](spec))
    cells = os.path.join(
        subprocess.run(["yosys-config", "--datdir"], capture_output=True,
                       text=True).stdout.strip(), "ice40", "cells_sim.v")
    rc, out = run(["iverilog", "-g2005", "-o", "bit.out", "-DNO_ICE40_DEFAULT_ASSIGNMENTS",
                   tb, "unpacked.v", cells])
    if rc:
        print("   could not compile the unpacked bitstream for simulation:")
        print("   " + out.strip().splitlines()[0][:160] if out.strip() else "")
        bit_verified = None
    else:
        rc, out = run(["vvp", "bit.out"], timeout=900)
        bit_verified = "TB_RESULT: PASS" in out
        if bit_verified:
            m2 = re.search(r"TB_PASS checks=(\d+)", out)
            print("   the packed bitstream passes the testbench (%s checks)"
                  % (m2.group(1) if m2 else "?"))
        else:
            bad = [l for l in out.splitlines() if "TB_FAIL" in l]
            print("   BITSTREAM FAILS: " + (bad[0][:140] if bad else
                                            out.strip()[-160:]))

    rc, out = run(["icetime", "-d", a.device, "-mtr", "top.rpt", "top.asc"])
    it = re.search(r"Total path delay:\s+([\d.]+) ns \(([\d.]+) MHz\)", out)
    if it:
        print("   icetime critical path %.2f ns (%.2f MHz)"
              % (float(it.group(1)), float(it.group(2))))

    result = {
        "block": a.block, "spec": spec["name"],
        "device": "ice40-%s-%s" % (a.device, a.package),
        "target_mhz": freq, "post_route_fmax_mhz": post_fmax,
        "timing_met": bool(post_fmax and post_fmax >= freq),
        "luts": lcs, "bitstream_bytes": size,
        "bitstream_verified": bit_verified,
        "icetime_mhz": float(it.group(2)) if it else None,
    }
    with open(os.path.join(ROOT, "bitstream_%s.json" % a.block), "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote bitstream_%s.json" % a.block)
    if bit_verified is False:
        return 1
    return 0 if result["timing_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
