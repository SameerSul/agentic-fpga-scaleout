"""Point the agent swarm at every generated block, for the current model.

Until now the LLM path was measured on the MAC and the endpoint only,
and the seven newer blocks had been written exclusively by the
deterministic agent. That made "an LLM writes this RTL" a claim about
two blocks and an assumption about the rest. This runs the swarm at all
of them and reports what actually converged.

  python3 swarm_all.py [--agent swarm|llm] [--iters 8]
"""
import argparse
import json
import os
import shutil
import sys
import time

import chiplet_flow as cf

BLOCKS = [
    ("chiplet",  cf.CHIPLET_JOB),
    ("requant",  cf.REQUANT_JOB),
    ("exp",      cf.EXP_JOB),
    ("recip",    cf.RECIP_JOB),
    ("rsqrt",    cf.RSQRT_JOB),
    ("matvec",   cf.MATVEC_JOB),
    ("wmem",     cf.WMEM_JOB),
    ("softmax",  cf.SOFTMAX_JOB),
    ("mlp",      cf.MLP_JOB),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="swarm",
                    choices=["swarm", "llm", "rules"])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    ms = cf.specgen.load_model_spec()
    print("model %s, agent %s\n" % (ms["name"], a.agent))
    print("%-9s %-10s %5s %6s %8s  %s"
          % ("block", "result", "iter", "calls", "sec", "signed off as"))
    print("-" * 74)
    rows = []
    for name, job in BLOCKS:
        if a.only and name != a.only:
            continue
        # Own file names so a run never disturbs the committed artifacts.
        j = dict(job)
        for k in ("spec_file", "tb_file", "rtl_file", "profile_file",
                  "report_file"):
            j[k] = j[k].replace(".", "_sw.", 1)
        agent = cf.make_agent(a.agent)
        t0 = time.time()
        try:
            rep, prof = cf.run_flow(j, verbose=False, agent=agent,
                                    max_iters=a.iters)
        except Exception as e:
            print("%-9s %-10s %5s %6s %8s  %s"
                  % (name, "ERROR", "-", "-", "-", str(e)[:28]))
            rows.append({"block": name, "converged": False,
                         "error": str(e)[:200]})
            continue
        calls = getattr(agent, "calls", None)
        ncall = sum(calls.values()) if isinstance(calls, dict) else calls
        note = ""
        if prof:
            note = "%.0f MHz, %d cells" % (prof["fmax_estimate_mhz"],
                                           prof["cell_count"])
        else:
            last = (rep["history"] or [{}])[-1]
            for st in ("sim", "synth", "timing", "fpga"):
                r = last.get(st)
                if r and r.get("status") == "fail":
                    note = "stuck at " + st
                    mm = (r.get("mismatches") or [{}])[0]
                    if mm:
                        note += ": " + " ".join(
                            "%s=%s" % kv for kv in list(mm.items())[:3])
                    elif r.get("errors"):
                        note += ": " + r["errors"][0][:44]
                    break
        print("%-9s %-10s %5d %6s %8.0f  %s"
              % (name, "converged" if rep["converged"] else "FAILED",
                 rep["iterations_used"], "-" if ncall is None else ncall,
                 time.time() - t0, note))
        sys.stdout.flush()
        rows.append({"block": name, "converged": rep["converged"],
                     "iterations": rep["iterations_used"],
                     "calls": ncall, "roles": calls if isinstance(calls, dict)
                     else None, "note": note})
        # Only the file entries: a job dict also carries flags and
        # tuples, and joining a path onto a bool raises.
        for k in ("spec_file", "tb_file", "rtl_file", "profile_file",
                  "report_file"):
            p = os.path.join(cf.ROOT, j[k])
            if os.path.exists(p):
                os.remove(p)

    ok = [r for r in rows if r["converged"]]
    print("\n%d/%d blocks signed off by the %s agent"
          % (len(ok), len(rows), a.agent))
    if len(ok) < len(rows):
        print("did not converge: " + ", ".join(r["block"] for r in rows
                                               if not r["converged"]))
    out = os.path.join(cf.ROOT, "swarm_all_%s.json" % a.agent)
    json.dump({"model": ms["name"], "agent": a.agent, "blocks": rows},
              open(out, "w"), indent=2)
    print("written to %s" % os.path.basename(out))
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
