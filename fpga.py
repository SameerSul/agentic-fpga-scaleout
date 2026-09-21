"""FPGA synthesis backend: map the generated RTL onto real device primitives.

The generic-liberty flow in chiplet_flow.py answers the ASIC question (cells
and area against a standard-cell library). This module answers the FPGA
question, which is the one the deployment path actually cares about: how
many LUTs, flip-flops, DSP slices, and block RAMs does this design consume
on a real part, and is the RTL free of constructs that break on FPGA?

Two findings drive the board model in boards.py:

  the MAC chiplet infers a DSP slice, so the compute array is DSP-bound,
  not LUT-bound, and a device's DSP count sets the instance ceiling;

  the CRC endpoint is pure LUT logic, so link count scales against the
  LUT budget instead.

Neither is visible in generic-cell synthesis, where both blocks look like
undifferentiated gates. Uses yosys synth_xilinx; no vendor tools required.
Pure Python 3 stdlib."""
import os
import re
import subprocess

# Cell-name prefixes to resource buckets. INV counts as a LUT: yosys leaves
# inverters unmapped and vendor tools absorb them into a LUT input, so
# counting them keeps the estimate conservative rather than optimistic.
BUCKETS = (
    ("luts", re.compile(r"^(LUT[1-6]|INV)$")),
    ("ffs", re.compile(r"^FD[RSCP]E?$")),
    ("dsps", re.compile(r"^DSP\d+E?\d*$")),
    ("brams", re.compile(r"^RAMB\d+")),
    ("urams", re.compile(r"^URAM\d+")),
    ("carry", re.compile(r"^CARRY\d+$")),
    ("muxf", re.compile(r"^MUXF\d+$")),
)
IGNORE = re.compile(r"^(BUFG|IBUF|OBUF|BUFGCE)")
CELL_LINE = re.compile(r"^\s+(\d+)\s+([A-Z][A-Z0-9_]*)\s*$")
LATCH = re.compile(r"Latch inferred for signal\s+`([^']+)'")
MULTIDRIVE = re.compile(r"multiple conflicting drivers.*?`([^']+)'", re.I)
LOOP = re.compile(r"found logic loop|combinational loop", re.I)

FAMILIES = ("xcup", "xcu", "xc7")


def _run(args, cwd, timeout=300):
    r = subprocess.run(args, capture_output=True, text=True,
                       timeout=timeout, cwd=cwd)
    return r.returncode, r.stdout + r.stderr


def synth_fpga(rtl_file, top, build_dir, family="xcup"):
    """Synthesize one block for an FPGA family and return its resource use.

    rtl_file is a bare filename inside build_dir: every tool call runs with
    cwd=build_dir because the project path contains a space, which yosys
    script parsing does not tolerate unquoted.

    -noiopad is deliberate: these blocks are instantiated inside a larger
    design, so counting I/O buffers for their ports would charge pins the
    real system never spends."""
    script = ("read_verilog {rtl}; synth_xilinx -family {fam} -noiopad "
              "-top {top}; check; stat".format(
                  rtl=rtl_file, fam=family, top=top))
    rc, out = _run(["yosys", "-p", script], build_dir)
    if rc != 0:
        return {"status": "fail", "family": family,
                "errors": [l for l in out.splitlines() if "ERROR" in l][:5]}

    # The final stat block for the top module: take the last occurrence, so
    # per-submodule tables earlier in the log cannot shadow the totals.
    tail = out.rsplit("=== " + top + " ===", 1)[-1]
    res = {k: 0 for k, _ in BUCKETS}
    unknown = {}
    for line in tail.splitlines():
        m = CELL_LINE.match(line)
        if not m:
            continue
        n, cell = int(m.group(1)), m.group(2)
        if IGNORE.match(cell):
            continue
        for key, pat in BUCKETS:
            if pat.match(cell):
                res[key] += n
                break
        else:
            unknown[cell] = unknown.get(cell, 0) + n
    res.update(status="pass", family=family, top=top,
               lint=lint_findings(out))
    if unknown:
        res["unmapped_cells"] = unknown
    return res


def lint_findings(log):
    """FPGA-hostile constructs, parsed from the synthesis log.

    These are the failures that pass simulation and then misbehave or refuse
    to build on real hardware, so they belong in the agent's feedback loop
    next to testbench mismatches rather than being discovered at bring-up."""
    out = []
    for name in LATCH.findall(log):
        out.append({"kind": "inferred_latch", "signal": name,
                    "hint": "incomplete assignment in a combinational block; "
                            "drive every branch or make the signal clocked"})
    for name in MULTIDRIVE.findall(log):
        out.append({"kind": "multiple_drivers", "signal": name,
                    "hint": "one signal assigned from more than one block"})
    if LOOP.search(log):
        out.append({"kind": "combinational_loop",
                    "hint": "feedback path with no register in it"})
    return out


def summarize(res):
    if res.get("status") != "pass":
        return "FAIL"
    s = "%d LUT, %d FF, %d DSP" % (res["luts"], res["ffs"], res["dsps"])
    if res.get("brams") or res.get("urams"):
        s += ", %d BRAM" % res["brams"]
    if res.get("lint"):
        s += ", %d lint" % len(res["lint"])
    return s
