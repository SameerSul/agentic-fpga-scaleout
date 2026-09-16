"""Orchestrator: agent proposes chiplet RTL, then simulate (iverilog/vvp),
synthesize (yosys + generic liberty), and time (OpenSTA, with a yosys
gate-depth proxy fallback). Parsed failures feed back to the agent. Stops on
full pass or 5 iterations, then writes chiplet_profile.json: the measured
contract (cycles_per_mac, cell_count, area, fmax_estimate_mhz, checks) that
the scaleout fabric consumes. Run: python3 chiplet_flow.py"""
import json, os, re, shutil, subprocess, sys

from agent import RuleBasedAgent

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
LIB = os.path.join(ROOT, "cells.lib")
MAX_ITERS = 5
# Fallback delay per logic level for the gate-depth timing proxy, ns. Roughly
# one INV (0.06 ns) plus wire/load margin in the toy liberty.
PROXY_GATE_NS = 0.12


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=BUILD)
    return r.returncode, r.stdout + r.stderr


def tool(name):
    return shutil.which(name)


def stage_sim(rtl_path):
    sim = os.path.join(BUILD, "sim.out")
    rc, out = run(["iverilog", "-g2005", "-o", sim, os.path.join(ROOT, "tb_mac.v"), rtl_path])
    if rc != 0:
        return {"stage": "sim", "status": "fail", "phase": "compile",
                "errors": out.strip().splitlines()[:10], "mismatches": []}
    rc, out = run(["vvp", sim])
    mismatches = [
        {"test": m[0], "expected_acc": int(m[1]), "got_acc": int(m[2]),
         "expected_vout": m[3], "got_vout": m[4]}
        for m in re.findall(
            r"TB_FAIL test=(\S+) expected_acc=(\d+) got_acc=(\d+) "
            r"expected_vout=(\w) got_vout=(\w)", out)
    ]
    passed = "TB_RESULT: PASS" in out and rc == 0
    checks = re.search(r"TB_PASS checks=(\d+)", out)
    prof = re.search(r"TB_PROFILE macs=(\d+) span_cycles=(\d+) latency_cycles=(\d+)", out)
    res = {"stage": "sim", "status": "pass" if passed else "fail",
           "checks": int(checks.group(1)) if checks else None,
           "mismatches": mismatches}
    if prof:
        macs, span, lat = (int(g) for g in prof.groups())
        res["throughput"] = {"macs": macs, "span_cycles": span,
                             "latency_cycles": lat,
                             "cycles_per_mac": span / macs}
    return res


def stage_synth(rtl_path):
    # Relative names only: all stages run with cwd=BUILD and the project path
    # contains a space, which yosys script parsing does not tolerate unquoted.
    script = ("read_verilog {rtl}; synth -top mac; dfflibmap -liberty {lib}; "
              "abc -liberty {lib}; opt_clean; stat -liberty {lib}; "
              "write_verilog -noattr netlist.v").format(
                  rtl=os.path.basename(rtl_path), lib="cells.lib")
    rc, out = run(["yosys", "-p", script])
    if rc != 0:
        return {"stage": "synth", "status": "fail",
                "errors": [l for l in out.splitlines() if "ERROR" in l][:5]}
    # Yosys 0.68 stat format: "  1044     2066 cells" (count, area, label)
    cells = re.search(r"^\s*(\d+)\s+[\d.]+\s+cells\s*$", out, re.M) \
        or re.search(r"Number of cells:\s+(\d+)", out)
    area = re.search(r"Chip area for module .*?:\s+([\d.]+)", out)
    return {"stage": "synth", "status": "pass",
            "cell_count": int(cells.group(1)) if cells else None,
            "area": float(area.group(1)) if area else None}


def stage_timing(spec):
    period_ns = 1000.0 / spec["parameters"]["target_clock_mhz"]
    if tool("sta"):
        data_inputs = " ".join(p["name"] for p in spec["ports"]
                               if p["dir"] == "input" and p["name"] != "clk")
        tcl = os.path.join(BUILD, "run_sta.tcl")
        with open(tcl, "w") as f:
            f.write("""read_liberty cells.lib
read_verilog netlist.v
link_design mac
create_clock -name clk -period {per} [get_ports clk]
set_input_delay 0.5 -clock clk [get_ports {{{ins}}}]
set_output_delay 0.5 -clock clk [all_outputs]
report_checks -path_delay max
report_worst_slack -max
exit
""".format(per=period_ns, ins=data_inputs))
        try:
            # Relative script name: OpenSTA splits the path on spaces internally
            rc, out = run(["sta", "-no_init", "-exit", os.path.basename(tcl)])
        except subprocess.TimeoutExpired:
            rc, out = 1, "sta timeout"
        m = re.search(r"worst slack (?:max )?(-?[\d.]+)", out)
        if rc == 0 and m:
            slack = float(m.group(1))
            return {"stage": "timing", "status": "pass" if slack >= 0 else "fail",
                    "method": "opensta", "clock_period_ns": period_ns,
                    "worst_slack_ns": slack}
    # Proxy fallback: yosys longest topological path in AND-mapped netlist.
    rc, out = run(["yosys", "-p",
                   "read_verilog mac.v; synth -top mac -flatten; abc -g AND; ltp -noff"])
    m = re.search(r"length=(\d+)", out)
    if rc == 0 and m:
        return {"stage": "timing", "status": "pass", "method": "proxy_gate_depth",
                "note": "OpenSTA unavailable or failed, gate depth is a timing proxy",
                "gate_depth": int(m.group(1)), "clock_period_ns": period_ns}
    return {"stage": "timing", "status": "fail", "method": "none",
            "errors": out.strip().splitlines()[-5:]}


def summarize(res):
    if res is None:
        return "-"
    if res["stage"] == "sim":
        if res["status"] == "pass":
            return "pass ({} checks)".format(res["checks"])
        if res.get("phase") == "compile":
            return "FAIL compile"
        m = res["mismatches"][0] if res["mismatches"] else {}
        return "FAIL {}: exp {} got {}".format(
            m.get("test", "?"), m.get("expected_acc"), m.get("got_acc"))
    if res["stage"] == "synth":
        return "pass ({} cells)".format(res["cell_count"]) if res["status"] == "pass" else "FAIL"
    if res["stage"] == "timing":
        if res["method"] == "opensta":
            return "{} (slack {:+.2f} ns)".format(res["status"], res["worst_slack_ns"])
        if res["method"] == "proxy_gate_depth":
            return "proxy (depth {})".format(res["gate_depth"])
        return "FAIL"
    return res["status"]


def derive_profile(spec, final):
    """Build the machine-readable chiplet profile from measured results only.
    This JSON is the contract between the RTL half and the scaleout half."""
    sim, synth, tim = final["sim"], final["synth"] or {}, final["timing"]
    thr = sim.get("throughput") or {}
    target_mhz = spec["parameters"]["target_clock_mhz"]
    fmax, method = None, "none"
    if tim and tim.get("method") == "opensta" and tim["status"] == "pass":
        # Worst path fits in (period - slack) ns, so the achievable clock is
        # fmax = 1000 / (period - slack).
        fmax = 1000.0 / (tim["clock_period_ns"] - tim["worst_slack_ns"])
        method = "opensta_slack"
    elif tim and tim.get("method") == "proxy_gate_depth":
        fmax = 1000.0 / (tim["gate_depth"] * PROXY_GATE_NS)
        method = "gate_depth_proxy"
    elif tim is None or tim.get("status") == "skipped":
        fmax = float(target_mhz)
        method = "target_assumed_no_timing_tool"
    return {
        "chiplet": spec["name"],
        "top_module": spec["top_module"],
        "cycles_per_mac": thr.get("cycles_per_mac"),
        "latency_cycles": thr.get("latency_cycles"),
        "cell_count": synth.get("cell_count"),
        "area": synth.get("area"),
        "fmax_estimate_mhz": fmax,
        "fmax_method": method,
        "target_clock_mhz": target_mhz,
        "sim_checks_passed": sim.get("checks"),
        "data_width": spec["parameters"]["data_width"],
        "acc_width": spec["parameters"]["acc_width"],
    }


def run_flow(verbose=True):
    """Run the agentic loop to convergence. Returns (report, profile);
    profile is None if the loop did not converge."""
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(BUILD, exist_ok=True)
    shutil.copy(LIB, BUILD)
    spec = json.load(open(os.path.join(ROOT, "spec.json")))
    tools = {t: tool(t) for t in ("iverilog", "vvp", "yosys", "sta")}
    say("Tools:", ", ".join("{}={}".format(k, v or "MISSING") for k, v in tools.items()))
    if not (tools["iverilog"] and tools["vvp"]):
        say("iverilog/vvp required for the demo loop, aborting")
        sys.exit(1)

    agent = RuleBasedAgent()
    history, iterations = [], []
    rows = [("iter", "fixes applied", "sim", "synth", "timing")]
    converged = False

    for it in range(1, MAX_ITERS + 1):
        rtl, fixes = agent.propose(spec, history)
        rtl_path = os.path.join(BUILD, "mac.v")
        with open(rtl_path, "w") as f:
            f.write(rtl)

        sim = stage_sim(rtl_path)
        synth = tim = None
        if sim["status"] == "pass" and tools["yosys"]:
            synth = stage_synth(rtl_path)
            if synth["status"] == "pass":
                tim = stage_timing(spec)
        elif sim["status"] == "pass":
            synth = {"stage": "synth", "status": "skipped", "note": "yosys missing"}

        record = {"iteration": it, "fixes_applied": fixes,
                  "sim": sim, "synth": synth, "timing": tim}
        iterations.append(record)
        for r in (sim, synth, tim):
            if r and r["status"] == "fail":
                history.append(dict(r, iteration=it))
        rows.append((str(it), ",".join(fixes) or "none",
                     summarize(sim), summarize(synth), summarize(tim)))

        ok = lambda r: r is not None and r["status"] in ("pass", "skipped")
        if ok(sim) and ok(synth) and (tim is None and synth["status"] == "skipped" or ok(tim)):
            converged = True
            break

    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    say()
    for i, r in enumerate(rows):
        say("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if i == 0:
            say("-" * (sum(widths) + 2 * (len(widths) - 1)))

    final = iterations[-1]
    report = {
        "spec": spec, "tools": tools, "converged": converged,
        "iterations_used": len(iterations), "history": iterations,
        "final_metrics": {
            "sim_pass": final["sim"]["status"] == "pass",
            "sim_checks": final["sim"].get("checks"),
            "cell_count": (final["synth"] or {}).get("cell_count"),
            "area": (final["synth"] or {}).get("area"),
            "timing": final["timing"],
        },
    }
    rpath = os.path.join(ROOT, "report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    profile = None
    if converged:
        profile = derive_profile(spec, final)
        ppath = os.path.join(ROOT, "chiplet_profile.json")
        with open(ppath, "w") as f:
            json.dump(profile, f, indent=2)
        say("\nCONVERGED in {} iteration(s), report written to {}".format(
            len(iterations), rpath))
        say("chiplet profile written to {}".format(ppath))
    else:
        say("\nDID NOT CONVERGE in {} iteration(s), report written to {}".format(
            len(iterations), rpath))
    return report, profile


def main():
    report, profile = run_flow(verbose=True)
    sys.exit(0 if report["converged"] else 2)


if __name__ == "__main__":
    main()
