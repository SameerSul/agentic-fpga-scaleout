"""Every input: the blocks whose input space is small enough to simulate
whole, checked against their golden model on all of it.

A testbench checks the vectors it drives, and mutation testing says how
well; neither says a design is right on the inputs nobody drove. For the
exponential (a 13-bit input, non-positive: 4096 values) and SiLU (13 bits:
8192) every input fits in one run of the block's own testbench, so the
check is complete: a design that passes equals the golden model, and so
the reference design, on every input there is.

  python3 exhaustive.py GATES_DIR     the blocks in a gates directory
"""
import json
import os
import shutil
import sys

import formal

ROOT = os.path.dirname(os.path.abspath(__file__))


def inputs(spec):
    """Every input of the block, or None for a block too wide for this."""
    import specgen
    p = spec["parameters"]
    top = spec["top_module"]
    if top == "expu":
        return list(range(0, -(1 << (p["in_width"] - 1)) - 1, -1)), specgen.render_exp_testbench
    if top == "silu":
        w = p["width"]
        return list(range(-(1 << (w - 1)), 1 << (w - 1))), specgen.render_silu_testbench
    return None, None


def check(spec, rtl_path, deps=(), work=None, timeout=1200):
    """Run the block's testbench on every input. Returns {"status":
    "pass" | "fail" | "n/a" | "error", "inputs": n}."""
    xs, render = inputs(spec)
    if xs is None:
        return {"status": "n/a"}
    work = work or os.path.join(ROOT, "build_exhaustive", spec["top_module"])
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    with open(os.path.join(work, "tb.v"), "w") as f:
        f.write(render(spec, xs))
    srcs = ["tb.v"]
    for i, s in enumerate([rtl_path] + list(deps)):
        d = "s%d_%s" % (i, os.path.basename(s))
        shutil.copyfile(s, os.path.join(work, d))
        srcs.append(d)
    rc, out = formal._run(["iverilog", "-g2005", "-o", "ex.vvp"] + srcs, work, timeout)
    if rc != 0:
        return {"status": "error", "detail": out.strip().splitlines()[-3:]}
    rc, out = formal._run(["vvp", "-n", "ex.vvp"], work, timeout)
    ok = "TB_RESULT: PASS" in out
    fail = next((l for l in out.splitlines() if "TB_FAIL" in l), "")
    return {"status": "pass" if ok else "fail", "inputs": len(xs), "detail": fail[:300]}


if __name__ == "__main__":
    import re
    import contracts
    g = sys.argv[1]
    for rep in sorted(f for f in os.listdir(g) if re.match(r"report_[^.]+\.json$", f)):
        spec = json.load(open(os.path.join(g, rep)))["spec"]
        tag = rep[7:-5]
        rtl = os.path.join(g, tag + ".v")
        res = check(spec, rtl, contracts.supplied(g, rtl, skip=tag + ".v"))
        if res["status"] != "n/a":
            print(tag, res)
