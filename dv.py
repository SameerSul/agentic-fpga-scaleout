"""Mutation testing for the generated DV.

A converged run only means the RTL passed the testbench. It says nothing
about whether the testbench could have caught a bug, and a testbench that
cannot fail is not verification. This injects known defects into RTL that
already passes, and reports how many the testbench kills.

Buckets, kept separate on purpose:

  killed      the testbench reported TB_RESULT: FAIL. This is DV working.
  equivalent  yosys proved the mutant behaves identically to the original, so
              no testbench could ever kill it. Excluded from the score.
  compile     iverilog rejected the mutant. Real, but the compiler caught it,
              not the testbench, so it is not evidence about DV quality.
  SURVIVED    the mutant simulated clean and the testbench said PASS. This is
              a hole: a bug the flow would sign off on.
  n/a         the operator did not match this RTL, so nothing was tested.

A mutant that changes no behaviour can never be killed by any testbench, and
counting it as a DV failure would understate the score. Every survivor is
therefore put to yosys for a sequential equivalence proof before it is called
a hole, so the number that remains is the number that is really wrong.

  python3 dv.py                              (defaults to build/mac.v + tb_mac.v)
  python3 dv.py --rtl build/mac_bench.v --tb tb_mac.v
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DVDIR = os.path.join(ROOT, "build_dv")   # never the flow's build dir


def _widen(m):
    """Halve the declared width of the widest vector: the classic truncated
    accumulator, which is exactly what the wide_product test exists for."""
    hi = int(m.group(1))
    return "[%d:0]" % (hi // 2) if hi >= 3 else m.group(0)


def mut_trunc(src):
    widths = [int(w) for w in re.findall(r"\[(\d+):0\]", src)]
    if not widths:
        return src
    top = max(widths)
    return re.sub(r"\[%d:0\]" % top, "[%d:0]" % (top // 2), src)


def _sub_first_real_add(src):
    """Flip the first addition that is part of the datapath. Loop induction
    (i = i + 1) is skipped: turning it into a subtraction makes the loop
    never terminate, which hangs the simulator and measures nothing about
    the testbench."""
    out = []
    done = False
    for line in src.splitlines(True):
        if (not done and " + " in line
                and "for (" not in line.replace("for(", "for (")
                and not re.search(r"\b(\w+)\s*=\s*\1\s*\+\s*1\b", line)):
            line = line.replace(" + ", " - ", 1)
            done = True
        out.append(line)
    return "".join(out)


def mut_nba(src):
    # Non-blocking to blocking inside sequential blocks: a real scheduling
    # race. Guard against '<=' used as a comparison by requiring a lone
    # identifier (optionally indexed) on the left.
    return re.sub(r"(^\s*[A-Za-z_]\w*(?:\s*\[[^\]]+\])?\s*)<=", r"\1 =",
                  src, flags=re.M)


OPS = [
    ("acc_truncated", mut_trunc,
     "halve the widest vector: accumulator loses its high bits"),
    ("adder_to_subtractor", lambda s: _sub_first_real_add(s),
     "first datapath addition becomes a subtraction"),
    ("reset_dead", lambda s: re.sub(r"[!~]\s*rst_n", "1'b0", s, count=1),
     "reset branch can never be taken"),
    ("clear_dead", lambda s: re.sub(r"if\s*\(\s*clear", "if (1'b0 && clear",
                                    s, count=1),
     "synchronous clear never fires"),
    ("valid_ungated", lambda s: re.sub(r"if\s*\(\s*(m_)?valid",
                                       r"if (1'b1 || \1valid", s, count=1),
     "accumulate ignores the valid qualifier"),
    ("clock_edge_flipped", lambda s: s.replace("posedge clk", "negedge clk", 1),
     "one sequential block samples on the wrong edge"),
    ("product_off_by_one",
     lambda s: re.sub(r"(\b\w+\s*\*\s*\w+\b)", r"(\1 + 1)", s, count=1),
     "multiplier result is off by one"),
    ("nonblocking_to_blocking", mut_nba,
     "sequential assignments become blocking: scheduling race"),
    ("signedness_dropped", lambda s: re.sub(r"\bsigned\b\s*", "", s),
     "signed datapath becomes unsigned: every negative operand is read as "
     "a large positive one"),
]


def survivor_verdict(orig, mutant, top, extra=()):
    """Classify a survivor: equivalent, a hole, or not provable in time.

    The third case is real and has to be said rather than folded into
    one of the others. A design holding a memory expands to thousands of
    flip-flops once memory_map runs, and the solver does not finish. To
    call that a DV hole would be to invent a defect; to call it
    equivalent would be to assume one away.
    """
    if prove_equivalent(orig, mutant, top, extra):
        return "equivalent"
    return "unproven" if _last_equiv_timed_out[0] else "hole"


_last_equiv_timed_out = [False]


def prove_equivalent(orig, mutant, top, extra=()):
    """Ask yosys whether the mutant is sequentially equivalent to the
    original. Only a proof reclassifies a survivor; anything inconclusive
    leaves it counted as a hole, because the honest default is to assume the
    testbench missed something."""
    if not shutil.which("yosys"):
        return False
    with open(os.path.join(DVDIR, "gold.v"), "w") as f:
        f.write(orig)
    with open(os.path.join(DVDIR, "gate.v"), "w") as f:
        f.write(mutant)
    # A block that instantiates others cannot be elaborated without
    # them. Without this the proof fails for want of a module rather
    # than for want of equivalence, and every mutant of a composite
    # block is misreported as a DV hole.
    deps = ""
    for i, src in enumerate(extra):
        name = "eqdep%d.v" % i
        shutil.copy(src, os.path.join(DVDIR, name))
        deps += " " + name
    # memory_map turns an inferred ROM into plain logic. Without it a
    # design with a lookup table cannot be proven at all: the SAT solver
    # reports "no SAT model available" for the $mem cell and every mutant
    # in such a design is misreported as a surviving DV hole.
    script = ("read_verilog gold.v{d}; prep -top {t} -flatten; memory_map; "
              "opt -full; design -stash gold; read_verilog gate.v{d}; "
              "prep -top {t} -flatten; memory_map; opt -full; "
              "design -stash gate; "
              "design -copy-from gold -as gold {t}; "
              "design -copy-from gate -as gate {t}; "
              "equiv_make gold gate equiv; prep -top equiv; "
              "equiv_simple; equiv_induct; equiv_status -assert"
              ).format(t=top, d=deps)
    rc, out = run(["yosys", "-p", script], DVDIR, timeout=90)
    # Both conditions: equiv_status -assert sets the exit code, and the
    # message confirms cells were actually compared rather than the
    # selection coming up empty and passing vacuously.
    _last_equiv_timed_out[0] = (rc == 124)
    return (rc == 0 and "Equivalence successfully proven" in out
            and re.search(r"Found [1-9]\d* \$equiv cells", out) is not None)


def run(cmd, cwd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def evaluate(name, src, tb_path, top, deps=()):
    os.makedirs(DVDIR, exist_ok=True)
    rtl = os.path.join(DVDIR, "mutant.v")
    with open(rtl, "w") as f:
        f.write(src)
    rc, out = run(["iverilog", "-g2005", "-o", "mutant.out", tb_path,
                   "mutant.v"] + list(deps), DVDIR)
    if rc != 0:
        return "compile", ""
    rc, out = run(["vvp", "mutant.out"], DVDIR)
    if rc == 124:
        return "hang", ""
    if "TB_RESULT: FAIL" in out:
        m = re.search(r"TB_FAIL test=(\S+)", out)
        return "killed", m.group(1) if m else ""
    if "TB_RESULT: PASS" in out:
        return "SURVIVED", ""
    return "no_verdict", out.strip().splitlines()[-1][:50] if out else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", default=os.path.join("build", "mac.v"))
    ap.add_argument("--tb", default="tb_mac.v")
    ap.add_argument("--top", default=None,
                    help="top module; detected from the RTL when omitted")
    a = ap.parse_args()

    rtl_path = a.rtl if os.path.isabs(a.rtl) else os.path.join(ROOT, a.rtl)
    tb_path = a.tb if os.path.isabs(a.tb) else os.path.join(ROOT, a.tb)
    for p in (rtl_path, tb_path):
        if not os.path.exists(p):
            raise SystemExit("missing %s" % p)
    src = open(rtl_path).read()
    # Read the top module out of the RTL. Taking it from a flag invites a
    # silent mismatch that makes the equivalence proof fail for the wrong
    # reason and reports equivalent mutants as DV holes.
    mods = re.findall(r"^\s*module\s+([A-Za-z_]\w*)", src, re.M)
    if not mods:
        raise SystemExit("no module declaration found in %s" % rtl_path)
    top = a.top or mods[0]
    if len(mods) > 1 and not a.top:
        print("note: %d modules in file, taking '%s' as top" % (len(mods), top))

    print("rtl %s\ntb  %s\n" % (os.path.relpath(rtl_path, ROOT),
                                os.path.relpath(tb_path, ROOT)))
    # The unmutated design must pass, or every later result is meaningless.
    base, _ = evaluate("baseline", src, tb_path, top)
    if base != "SURVIVED":
        raise SystemExit("baseline RTL does not pass its own testbench (%s): "
                         "mutation scores would be meaningless" % base)
    print("baseline passes its testbench\n")

    print("%-24s %-10s %s" % ("mutation", "verdict", "caught by"))
    print("-" * 62)
    tally = {}
    survivors = []
    for name, fn, desc in OPS:
        mutant = fn(src)
        if mutant == src:
            verdict, why = "n/a", "operator did not match this RTL"
        else:
            verdict, why = evaluate(name, mutant, tb_path, top)
            if verdict == "killed":
                why = "test '%s'" % why if why else "testbench"
            elif verdict == "compile":
                why = "iverilog, not the testbench"
            elif verdict == "SURVIVED":
                if prove_equivalent(src, mutant, top):
                    verdict = "equivalent"
                    why = "no behaviour change: yosys proved it"
                else:
                    why = "NOTHING: signed off with this bug in it"
                    survivors.append((name, desc))
        tally[verdict] = tally.get(verdict, 0) + 1
        print("%-24s %-10s %s" % (name, verdict, why))

    applied = sum(v for k, v in tally.items() if k != "n/a")
    killed = tally.get("killed", 0)
    surv = tally.get("SURVIVED", 0)
    testable = killed + surv
    print("\n%d operators applied, %d skipped as not applicable"
          % (applied, tally.get("n/a", 0)))
    if testable:
        print("testbench mutation score: %d/%d killed (%.0f%%)"
              % (killed, testable, 100.0 * killed / testable))
    print("compiler caught %d, %d proven equivalent (both excluded above)"
          % (tally.get("compile", 0), tally.get("equivalent", 0)))
    if survivors:
        print("\nDV HOLES, each one a bug this flow would sign off on:")
        for n, d in survivors:
            print("  %-22s %s" % (n, d))
    shutil.rmtree(DVDIR, ignore_errors=True)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
