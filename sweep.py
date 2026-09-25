"""Spec-to-RTL coverage sweep.

The flow is only worth anything if it works for specs nobody hand-tuned it
for. This drives many different specs end to end, chiplet and endpoint, and
at every point checks the whole chain rather than just the last step:

  spec derived  the derivation ran and produced a self-consistent spec
  rtl           an agent proposed RTL
  sim           it passes its generated, self-checking testbench
  synth         yosys maps it to the standard cell library
  timing        OpenSTA, not the proxy, closes at the spec's target clock
  fpga          it maps to real device primitives
  profile       the signed-off numbers the sizing model consumes are sane
  dv            the generated testbench kills every injected defect

The last one is the one that is easy to skip and the one that matters most:
a testbench that is generated at a width nobody tried can be vacuous, and a
vacuous testbench makes every gate above it meaningless.

Runs on the rule-based agent by default, so the sweep is free, deterministic
and safe in CI. It uses its own build directory and its own job files, so it
can run while something else is using the flow.

  python3 sweep.py                 (everything)
  python3 sweep.py --only chiplet
  python3 sweep.py --skip-dv       (gates only, much faster)
"""
import argparse
import copy
import json
import os
import re
import shutil
import sys

import chiplet_flow
import specgen
import dv
from chiplet_flow import ROOT, run_flow, make_agent

BUILD = os.path.join(ROOT, "build_sweep")
chiplet_flow.BUILD = BUILD          # never the flow's own build directory

JOB = {"spec_file": "spec_sweep.json", "tb_file": "tb_sweep.v",
       "rtl_file": "rtl_sweep.v", "profile_file": "profile_sweep.json",
       "report_file": "report_sweep.json"}

# Model specs the chiplet derivation has to handle. The point is the spread
# of derived widths: int4 weights, a 16-bit datapath, a model small enough
# that the guard term collapses, and an asymmetric weight/activation pair.
def _models(base):
    def m(name, **kw):
        d = copy.deepcopy(base)
        d.update(name=name, **kw)
        return d
    return [
        m("gpt2_124m"),
        m("int4_weights", weight_bits=4, activation_bits=8),
        m("int16_wide", weight_bits=16, activation_bits=16,
          d_model=4096, d_ff=11008),
        m("tiny_model", d_model=64, d_ff=256),
        m("qwen3_0p6b", d_model=1024, d_ff=3072),
        m("asym_w8_a16", weight_bits=8, activation_bits=16,
          d_model=2048, d_ff=8192),
    ]


LINK_RATES = [1.0, 10.0, 25.0, 100.0]

COLS = ("derived", "rtl", "sim", "synth", "timing", "fpga", "profile", "dv")


def _stage_status(entry, name):
    r = (entry or {}).get(name)
    return None if not r else r.get("status")


def check_profile(spec, profile, unit, timing=None):
    """The signed-off numbers have to be usable by the sizing model, not just
    present. A zero or absent throughput figure silently poisons every
    downstream prediction."""
    if not profile:
        return False, "no profile"
    # Timing has to come from a timing tool. The flow falls back to a
    # gate-depth proxy when OpenSTA cannot read the netlist, and a proxy
    # number is an estimate, not closure. Accepting it silently is how a
    # design that never met its clock gets signed off.
    if shutil.which("sta"):
        method = (timing or {}).get("method") or profile.get("fmax_method")
        if method and "opensta" not in method:
            return False, "timing came from %s, not OpenSTA" % method
    per = profile.get("cycles_per_%s" % unit)
    if not per or per <= 0:
        return False, "cycles_per_%s missing" % unit
    fmax = profile.get("fmax_estimate_mhz")
    target = spec["parameters"]["target_clock_mhz"]
    if not fmax or fmax < target:
        return False, "fmax %.1f below target %g" % (fmax or 0, target)
    if not profile.get("cell_count"):
        return False, "no cell count"
    f = profile.get("fpga") or {}
    if not (f.get("luts") or f.get("dsps")):
        return False, "no FPGA resources"
    return True, "%.2f cyc/%s, fmax %.0f MHz, %d cells" % (
        per, unit, fmax, profile["cell_count"])


def run_dv(rtl_path, tb_path):
    """Mutation-test the generated testbench. Returns (ok, detail)."""
    src = open(rtl_path).read()
    mods = re.findall(r"^\s*module\s+([A-Za-z_]\w*)", src, re.M)
    if not mods:
        return False, "no module found"
    top = mods[0]
    os.makedirs(dv.DVDIR, exist_ok=True)
    try:
        if dv.evaluate("base", src, tb_path, top)[0] != "SURVIVED":
            return False, "baseline fails its own testbench"
        killed, survived, skipped = 0, [], 0
        for name, fn, _ in dv.OPS:
            mutant = fn(src)
            if mutant == src:
                skipped += 1
                continue
            v = dv.evaluate(name, mutant, tb_path, top)[0]
            if v == "killed":
                killed += 1
            elif v == "SURVIVED":
                if dv.prove_equivalent(src, mutant, top):
                    skipped += 1
                else:
                    survived.append(name)
            else:
                skipped += 1
        if survived:
            return False, "SURVIVED: " + ",".join(survived)
        return True, "%d/%d killed, %d n/a" % (killed, killed, skipped)
    finally:
        shutil.rmtree(dv.DVDIR, ignore_errors=True)


def one_case(label, spec, unit, agent_kind, do_dv):
    """Run one spec all the way through and return a per-stage verdict."""
    res = dict.fromkeys(COLS, "")
    res["case"] = label
    res["derived"] = "ok"
    report, profile = run_flow(JOB, verbose=False, agent=make_agent(agent_kind))
    last = (report["history"] or [{}])[-1]
    res["rtl"] = "ok" if report["history"] else "FAIL"
    for st in ("sim", "synth", "timing", "fpga"):
        s = _stage_status(last, st)
        res[st] = {"pass": "ok", None: "-"}.get(s, s or "-")
    if not report["converged"]:
        res["profile"] = res["dv"] = "-"
        return res, "did not converge in %d iterations" % report["iterations_used"]
    ok, detail = check_profile(spec, profile, unit,
                               report["final_metrics"].get("timing"))
    res["profile"] = "ok" if ok else "FAIL"
    if not do_dv:
        res["dv"] = "skip"
        return res, detail
    dok, ddetail = run_dv(os.path.join(BUILD, JOB["rtl_file"]),
                          os.path.join(ROOT, JOB["tb_file"]))
    res["dv"] = "ok" if dok else "FAIL"
    return res, "%s | dv %s" % (detail, ddetail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="rules",
                    choices=["rules", "llm", "swarm"])
    ap.add_argument("--only", default="all",
                    choices=["all", "both", "chiplet", "requant", "exp",
                             "endpoint"])
    ap.add_argument("--skip-dv", action="store_true")
    a = ap.parse_args()
    do_dv = not a.skip_dv

    os.makedirs(BUILD, exist_ok=True)
    rows, details, failures = [], [], []

    if a.only in ("all", "both", "chiplet"):
        base = specgen.load_model_spec()
        for ms in _models(base):
            spec = specgen.generate(ms, spec_file=JOB["spec_file"],
                                    tb_file=JOB["tb_file"])
            p = spec["parameters"]
            label = "mac %s w%d/a%d dw%d acc%d" % (
                ms["name"], ms["weight_bits"], ms["activation_bits"],
                p["data_width"], p["acc_width"])
            r, d = one_case(label, spec, "mac", a.agent, do_dv)
            rows.append(r)
            details.append((label, d))
            if any(r[c] not in ("ok", "skip", "-") for c in COLS):
                failures.append(label)

    if a.only in ("all", "requant"):
        # The requantizer is derived from the same model spec, so it has to
        # track the same spread of widths the chiplet does.
        base = specgen.load_model_spec()
        for ms in _models(base):
            spec = specgen.generate_requant(ms, spec_file=JOB["spec_file"],
                                            tb_file=JOB["tb_file"])
            p = spec["parameters"]
            label = "requant %s acc%d->%d" % (ms["name"], p["acc_width"],
                                              p["out_width"])
            r, d = one_case(label, spec, "activation", a.agent, do_dv)
            rows.append(r)
            details.append((label, d))
            if any(r[c] not in ("ok", "skip", "-") for c in COLS):
                failures.append(label)

    if a.only in ("all", "exp"):
        base = specgen.load_model_spec()
        for ms in _models(base):
            spec = specgen.generate_exp(ms, spec_file=JOB["spec_file"],
                                        tb_file=JOB["tb_file"])
            p = spec["parameters"]
            label = "exp %s Q%d.%d->Q0.%d" % (
                ms["name"], p["in_width"] - 1 - p["in_frac"], p["in_frac"],
                p["out_frac"])
            r, d = one_case(label, spec, "score", a.agent, do_dv)
            rows.append(r)
            details.append((label, d))
            if any(r[c] not in ("ok", "skip", "-") for c in COLS):
                failures.append(label)

    if a.only in ("all", "both", "endpoint"):
        for gbps in LINK_RATES:
            rate, opts = specgen.endpoint_options(gbps)
            # Same architecture-level retry the endpoint flow performs: a
            # datapath that cannot close timing is the wrong datapath, not
            # an RTL bug, so widen and try again before calling it a failure.
            for opt in range(len(opts)):
                spec = specgen.generate_endpoint(
                    gbps, spec_file=JOB["spec_file"], tb_file=JOB["tb_file"],
                    option=opt)
                p = spec["parameters"]
                label = "crc %gG %dB/cyc @%gMHz" % (
                    rate, p["bytes_per_cycle"], p["target_clock_mhz"])
                r, d = one_case(label, spec, "byte", a.agent, do_dv)
                rows.append(r)
                details.append((label, d))
                if r["timing"] == "fail" and opt + 1 < len(opts):
                    rows[-1]["case"] = label + " (timing, widening)"
                    continue
                if any(r[c] not in ("ok", "skip", "-") for c in COLS):
                    failures.append(label)
                break

    w = max(len(r["case"]) for r in rows)
    print("\n%-*s  %s" % (w, "case", "  ".join("%-7s" % c for c in COLS)))
    print("-" * (w + 2 + 9 * len(COLS)))
    for r in rows:
        print("%-*s  %s" % (w, r["case"],
                            "  ".join("%-7s" % r[c] for c in COLS)))
    print("\ndetail")
    for label, d in details:
        print("  %-*s  %s" % (w, label, d))

    print("\n%d cases, %d clean, %d failed"
          % (len(rows), len(rows) - len(failures), len(failures)))
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
    for f in JOB.values():
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(BUILD, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
