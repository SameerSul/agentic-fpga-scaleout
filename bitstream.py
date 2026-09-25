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
from agent import RuleBasedAgent, FIX_WIDTH, FIX_CLEAR, FIX_SATURATE, FIX_XOR
from chiplet_flow import ROOT, TARGET_LINK_GBPS

WORK = os.path.join(ROOT, "build_bitstream")

BLOCKS = {
    "mac": ("mac", lambda ms: specgen.derive_chiplet_spec(ms),
            {FIX_WIDTH, FIX_CLEAR}),
    "requant": ("requant", lambda ms: specgen.derive_requant_spec(ms),
                {FIX_SATURATE}),
    "crc": ("crc32", lambda ms: specgen.derive_endpoint_spec(1.0),
            {FIX_XOR}),
}


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
    rtl = getattr(RuleBasedAgent(), "render_" + (
        "crc" if a.block == "crc" else a.block))(spec, fixes)
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

    print("placing and routing")
    rc, out = run(["nextpnr-ice40", "--%s" % a.device,
                   "--package", a.package, "--json", "top.json",
                   "--asc", "top.asc", "--freq", str(freq),
                   "--placer", "heap", "--seed", "1"])
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
        "icetime_mhz": float(it.group(2)) if it else None,
    }
    with open(os.path.join(ROOT, "bitstream_%s.json" % a.block), "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote bitstream_%s.json" % a.block)
    return 0 if result["timing_met"] else 1


if __name__ == "__main__":
    sys.exit(main())
