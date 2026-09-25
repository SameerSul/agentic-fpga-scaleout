"""Consistency harness: how reliably does an agent take a spec to signoff?

One successful run proves nothing. This runs the same job N times with a fresh
agent each time and reports the convergence rate, the iteration count, and for
an LLM-backed agent the number of model calls it took. It writes to its own
job files so the committed profiles are never disturbed.

  python3 bench.py --agent swarm --runs 5
  python3 bench.py --agent rules --runs 3     (free, deterministic, a smoke test)
"""
import argparse
import json
import os
import statistics
import sys
import time

import specgen
from chiplet_flow import run_flow, make_agent, ROOT

def job_files(tag):
    """Namespaced job files so two benches can run at once. Paired with
    CHIPLET_BUILD_DIR, which does the same for the scratch directory."""
    return {"spec_file": "spec_%s_bench.json" % tag,
            "tb_file": "tb_%s_bench.v" % tag,
            "rtl_file": "rtl_%s_bench.v" % tag,
            "profile_file": "profile_%s_bench.json" % tag,
            "report_file": "report_%s_bench.json" % tag}


JOB = job_files("default")


def one_run(kind, max_iters):
    global JOB
    specgen.generate(spec_file=JOB["spec_file"], tb_file=JOB["tb_file"])
    agent = make_agent(kind)
    t0 = time.time()
    report, profile = run_flow(JOB, verbose=False, agent=agent,
                               max_iters=max_iters)
    calls = getattr(agent, "calls", None)
    total = sum(calls.values()) if isinstance(calls, dict) else calls

    def first_fail(entry):
        for stage in ("sim", "synth", "timing", "fpga"):
            r = entry.get(stage)
            if r and r.get("status") == "fail":
                return stage
        return None

    # Every gate failure the run hit, not just the last: the shape of the
    # retry path is what says whether the swarm is reliable or just lucky.
    stumbles = [s for s in (first_fail(e) for e in report["history"]) if s]
    failed = None
    if not report["converged"]:
        failed = first_fail(report["history"][-1]) or "max_iters"
    return {"converged": report["converged"],
            "iterations": report["iterations_used"],
            "calls": total, "roles": calls if isinstance(calls, dict) else None,
            "seconds": time.time() - t0,
            "failed_at": failed, "stumbles": stumbles,
            "cells": (profile or {}).get("cell_count")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="swarm",
                    choices=["rules", "llm", "swarm"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--tag", default="default",
                    help="namespace for this bench's job files")
    a = ap.parse_args()
    global JOB
    JOB = job_files(a.tag)

    print("agent=%s runs=%d max_iters=%d\n" % (a.agent, a.runs, a.max_iters))
    print("%-4s %-10s %5s %6s %7s  %s"
          % ("run", "result", "iter", "calls", "sec", "note"))
    print("-" * 58)
    rows = []
    for i in range(1, a.runs + 1):
        try:
            r = one_run(a.agent, a.max_iters)
        except Exception as e:
            r = {"converged": False, "iterations": 0, "calls": None,
                 "roles": None, "seconds": 0.0, "stumbles": [],
                 "failed_at": "error:%s" % str(e)[:40], "cells": None}
        rows.append(r)
        print("%-4d %-10s %5d %6s %7.1f  %s"
              % (i, "converged" if r["converged"] else "FAILED",
                 r["iterations"],
                 "-" if r["calls"] is None else r["calls"],
                 r["seconds"],
                 r["failed_at"] or ("retried: " + ",".join(r["stumbles"])
                                    if r["stumbles"] else "clean first pass")))

    ok = [r for r in rows if r["converged"]]
    print("\nconvergence rate: %d/%d (%.0f%%)"
          % (len(ok), len(rows), 100.0 * len(ok) / len(rows)))
    if ok:
        it = [r["iterations"] for r in ok]
        print("iterations on success: min %d, median %.1f, max %d"
              % (min(it), statistics.median(it), max(it)))
        cl = [r["cells"] for r in ok if r["cells"]]
        if cl:
            print("synthesized cell count: min %d, max %d, spread %.0f%%"
                  % (min(cl), max(cl), 100.0 * (max(cl) - min(cl)) / min(cl)))
        cs = [r["calls"] for r in ok if r["calls"]]
        if cs:
            print("model calls on success: median %.1f" % statistics.median(cs))
            roles = {}
            for r in ok:
                for k, v in (r["roles"] or {}).items():
                    roles[k] = roles.get(k, 0) + v
            if roles:
                print("calls by role (total): "
                      + ", ".join("%s %d" % kv for kv in sorted(roles.items())))
        clean = sum(1 for r in ok if not r["stumbles"])
        print("first-pass clean (no gate retry): %d/%d" % (clean, len(ok)))
        allst = [s for r in rows for s in r["stumbles"]]
        if allst:
            print("gates that forced a retry: "
                  + ", ".join("%s x%d" % (g, allst.count(g))
                              for g in sorted(set(allst))))
    bad = [r["failed_at"] for r in rows if not r["converged"]]
    if bad:
        print("failures:", ", ".join(bad))

    for f in JOB.values():
        p = os.path.join(ROOT, f)
        if os.path.exists(p) and f != JOB["rtl_file"]:
            os.remove(p)
    json.dump(rows, open(os.path.join(ROOT, a.out), "w"), indent=2)
    print("\nper-run detail written to %s" % a.out)
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
