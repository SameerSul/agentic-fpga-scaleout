"""Orchestrator: agent proposes RTL, then simulate (iverilog/vvp), synthesize
(yosys + generic liberty), and time (OpenSTA, with a yosys gate-depth proxy
fallback). Parsed failures feed back to the agent. Stops on full pass or 5
iterations, then writes a measured profile JSON: the contract the scaleout
fabric consumes.

Two generated blocks share this one flow, defined as jobs: the compute
chiplet (spec_mac.json + tb_mac.v -> chiplet_profile.json, cycles_per_mac)
and the fabric endpoint (spec_crc.json + tb_crc.v -> fabric_profile.json,
cycles_per_byte). The chiplet spec and testbench are not checked-in inputs:
jobs marked derive_from_model regenerate them from model_spec.json via
specgen.py before every run, so the hardware always tracks the model.

The proposing agent is pluggable: the deterministic RuleBasedAgent by
default, or a real LLM (llm_agent.py) with --agent llm or CHIPLET_AGENT=llm.
Run: python3 chiplet_flow.py [--agent rules|llm|swarm]"""
import json, os, re, shutil, subprocess, sys

from agent import RuleBasedAgent
import fpga
import specgen

ROOT = os.path.dirname(os.path.abspath(__file__))
# Overridable so independent flows (the sweep, a test run, a bench) can
# work in parallel without fighting over one scratch directory.
BUILD = os.path.join(ROOT, os.environ.get("CHIPLET_BUILD_DIR", "build"))
LIB = os.path.join(ROOT, "cells.lib")
MAX_ITERS = 5
# Fallback delay per logic level for the gate-depth timing proxy, ns. Roughly
# one INV (0.06 ns) plus wire/load margin in the toy liberty.
PROXY_GATE_NS = 0.12
# Default FPGA family for the deployment-path synthesis stage.
FPGA_FAMILY = os.environ.get("CHIPLET_FPGA_FAMILY", "xcup")

CHIPLET_JOB = {
    "spec_file": "spec_mac.json", "tb_file": "tb_mac.v", "rtl_file": "mac.v",
    "profile_file": "chiplet_profile.json", "report_file": "report.json",
    "derive_from_model": True,
}
FABRIC_JOB = {
    "spec_file": "spec_crc.json", "tb_file": "tb_crc.v", "rtl_file": "crc.v",
    "profile_file": "fabric_profile.json", "report_file": "report_crc.json",
    "derive_from_link": True,
}
MLP_JOB = {
    "spec_file": "spec_mlp.json", "tb_file": "tb_mlp.v",
    "rtl_file": "mlp.v", "profile_file": "mlp_profile.json",
    "report_file": "report_mlp.json",
    "derive_from_model": "mlp",
    "extra_sources": ("mv_dep.v", "mac_dep.v", "rq_dep.v"),
}
SOFTMAX_JOB = {
    "spec_file": "spec_softmax.json", "tb_file": "tb_softmax.v",
    "rtl_file": "softmax.v", "profile_file": "softmax_profile.json",
    "report_file": "report_softmax.json",
    "derive_from_model": "softmax",
    # It instantiates these, so they are part of the design under test
    # rather than companions to it.
    "extra_sources": ("expu_dep.v", "recip_dep.v"),
}
WMEM_JOB = {
    "spec_file": "spec_wmem.json", "tb_file": "tb_wmem.v",
    "rtl_file": "wmem.v", "profile_file": "wmem_profile.json",
    "report_file": "report_wmem.json",
    "derive_from_model": "wmem",
    # The whole subsystem: the memory is checked feeding the sequencer
    # and the MAC, because a read port a cycle out of step with its
    # reader is invisible to either block alone.
    "extra_sources": ("mac_dep.v", "matvec_dep.v"),
}
MATVEC_JOB = {
    "spec_file": "spec_matvec.json", "tb_file": "tb_matvec.v",
    "rtl_file": "matvec.v", "profile_file": "matvec_profile.json",
    "report_file": "report_matvec.json",
    "derive_from_model": "matvec",
    # The sequencer is checked driving the real MAC, not a model of it.
    "extra_sources": ("mac_dep.v",),
}
RSQRT_JOB = {
    "spec_file": "spec_rsqrt.json", "tb_file": "tb_rsqrt.v",
    "rtl_file": "rsqrt.v", "profile_file": "rsqrt_profile.json",
    "report_file": "report_rsqrt.json",
    "derive_from_model": "rsqrt",
}
RECIP_JOB = {
    "spec_file": "spec_recip.json", "tb_file": "tb_recip.v",
    "rtl_file": "recip.v", "profile_file": "recip_profile.json",
    "report_file": "report_recip.json",
    "derive_from_model": "recip",
}
EXP_JOB = {
    "spec_file": "spec_exp.json", "tb_file": "tb_expu.v",
    "rtl_file": "expu.v", "profile_file": "exp_profile.json",
    "report_file": "report_exp.json",
    "derive_from_model": "exp",
}
REQUANT_JOB = {
    "spec_file": "spec_requant.json", "tb_file": "tb_requant.v",
    "rtl_file": "requant.v", "profile_file": "requant_profile.json",
    "report_file": "report_requant.json",
    "derive_from_model": "requant",
}
# Link rate the fabric endpoint is generated for. 10GbE is the default
# because it is what the mid board class actually exposes (SFP+ cages) and
# what a two-board bring-up would be cabled with. 25G is reachable by the
# same derivation but needs a pipelined or matrix-form CRC the rule-based
# agent does not write; see the README.
TARGET_LINK_GBPS = float(os.environ.get("CHIPLET_LINK_GBPS", 10.0))


def make_agent(kind=None):
    """Agent factory: 'rules' (default), 'llm' or 'swarm'. One fresh instance
    per flow run so an agent's attempt memory never leaks between blocks."""
    kind = kind or os.environ.get("CHIPLET_AGENT", "rules")
    if kind == "rules":
        return RuleBasedAgent()
    if kind == "llm":
        from llm_agent import LLMAgent
        return LLMAgent()
    if kind == "swarm":
        from swarm import SwarmAgent
        return SwarmAgent()
    raise SystemExit(
        "unknown agent %r, expected 'rules', 'llm' or 'swarm'" % kind)


def run(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=BUILD)
    return r.returncode, r.stdout + r.stderr


def tool(name):
    return shutil.which(name)


def _parse_kv(line):
    """Parse a machine line like 'test=foo expected_crc=1 got_crc=2' into a
    dict of strings, so one parser serves every testbench."""
    d = {}
    for kv in line.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def stage_sim(job, rtl_path):
    """Compile and run the job's testbench against the proposed RTL.

    A job may name extra sources. An integration testbench instantiates
    more than the block under test, and the point of one is to check a
    block against another generated block rather than against a model of
    it, so those have to be compiled in.
    """
    sim = os.path.join(BUILD, "sim.out")
    srcs = [os.path.join(ROOT, job["tb_file"]), rtl_path]
    srcs += [os.path.join(BUILD, f) for f in job.get("extra_sources", ())]
    rc, out = run(["iverilog", "-g2005", "-o", sim] + srcs)
    if rc != 0:
        return {"stage": "sim", "status": "fail", "phase": "compile",
                "errors": out.strip().splitlines()[:10], "mismatches": []}
    rc, out = run(["vvp", sim])
    mismatches = [_parse_kv(l) for l in re.findall(r"TB_FAIL (.*)", out)]
    passed = "TB_RESULT: PASS" in out and rc == 0
    checks = re.search(r"TB_PASS checks=(\d+)", out)
    res = {"stage": "sim", "status": "pass" if passed else "fail",
           "checks": int(checks.group(1)) if checks else None,
           "mismatches": mismatches}
    prof = re.search(r"TB_PROFILE (.*)", out)
    if prof:
        p = {k: int(v) for k, v in _parse_kv(prof.group(1)).items()}
        # The unit count is whatever the testbench named it. Hardcoding the
        # known unit names means every new block silently divides by None.
        units = next((v for k, v in p.items()
                      if k not in ("span_cycles", "latency_cycles")), None)
        res["throughput"] = dict(p, units=units,
                                 cycles_per_unit=p["span_cycles"] / units)
    return res


def stage_synth(job, spec, rtl_path):
    # Relative names only: all stages run with cwd=BUILD and the project path
    # contains a space, which yosys script parsing does not tolerate unquoted.
    # Extra sources are part of the design when the top instantiates
    # them, so they are read here too; yosys prunes whatever the top
    # does not reach.
    files = " ".join([os.path.basename(rtl_path)]
                     + list(job.get("extra_sources", ())))
    script = ("read_verilog {rtl}; synth -top {top}; dfflibmap -liberty {lib}; "
              "abc -liberty {lib}; opt_clean; stat -liberty {lib}; "
              "write_verilog -noattr netlist.v").format(
                  rtl=files, top=spec["top_module"], lib="cells.lib")
    rc, out = run(["yosys", "-p", script])
    if rc != 0:
        return {"stage": "synth", "status": "fail",
                "errors": [l for l in out.splitlines() if "ERROR" in l][:5]}
    # Yosys 0.68 stat format: "  1044     2066 cells" (count, area, label)
    _strip_signed(os.path.join(BUILD, "netlist.v"))
    cells = re.search(r"^\s*(\d+)\s+[\d.]+\s+cells\s*$", out, re.M) \
        or re.search(r"Number of cells:\s+(\d+)", out)
    area = re.search(r"Chip area for module .*?:\s+([\d.]+)", out)
    return {"stage": "synth", "status": "pass",
            "cell_count": int(cells.group(1)) if cells else None,
            "area": float(area.group(1)) if area else None}


def _strip_signed(path):
    """Remove `signed` from a gate-level netlist before OpenSTA reads it.

    Yosys carries the port signedness through to the netlist, and OpenSTA's
    Verilog reader rejects `input signed [15:0] a;` outright. At gate level
    signedness carries no information at all: the netlist is cells and wires
    and the sign lives in how the logic was built, not in a declaration.
    Without this every signed design fails to read and the flow falls back
    to the gate-depth proxy, reporting a timing number that never came from
    a timing tool.
    """
    try:
        src = open(path).read()
    except OSError:
        return
    out = re.sub(r"\b(input|output|inout|wire|reg)\s+signed\b", r"\1", src)
    if out != src:
        with open(path, "w") as f:
            f.write(out)


def stage_timing(job, spec):
    period_ns = 1000.0 / spec["parameters"]["target_clock_mhz"]
    if tool("sta"):
        data_inputs = " ".join(p["name"] for p in spec["ports"]
                               if p["dir"] == "input" and p["name"] != "clk")
        tcl = os.path.join(BUILD, "run_sta.tcl")
        with open(tcl, "w") as f:
            f.write("""read_liberty cells.lib
read_verilog netlist.v
link_design {top}
create_clock -name clk -period {per} [get_ports clk]
set_input_delay 0.5 -clock clk [get_ports {{{ins}}}]
set_output_delay 0.5 -clock clk [all_outputs]
report_checks -path_delay max
report_worst_slack -max
exit
""".format(top=spec["top_module"], per=period_ns, ins=data_inputs))
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
                   "read_verilog {rtl}; synth -top {top} -flatten; "
                   "abc -g AND; ltp -noff".format(
                       rtl=" ".join([job["rtl_file"]]
                                    + list(job.get("extra_sources", ()))),
                       top=spec["top_module"])])
    m = re.search(r"length=(\d+)", out)
    if rc == 0 and m:
        return {"stage": "timing", "status": "pass", "method": "proxy_gate_depth",
                "note": "OpenSTA unavailable or failed, gate depth is a timing proxy",
                "gate_depth": int(m.group(1)), "clock_period_ns": period_ns}
    return {"stage": "timing", "status": "fail", "method": "none",
            "errors": out.strip().splitlines()[-5:]}


def stage_fpga(job, spec, rtl_path, family=FPGA_FAMILY):
    """Map the block onto real FPGA primitives and gate on FPGA-hostile RTL.

    This is a hard stage, not a report: inferred latches, multiple drivers,
    and combinational loops fail the iteration and go back to the agent as
    structured feedback, because they are exactly the defects that pass
    simulation and then misbehave on a real part."""
    res = fpga.synth_fpga(os.path.basename(rtl_path), spec["top_module"],
                          BUILD, family,
                          extra=job.get("extra_sources", ()))
    if res.get("status") != "pass":
        return {"stage": "fpga", "status": "fail", "family": family,
                "errors": res.get("errors", [])}
    if res["lint"]:
        return {"stage": "fpga", "status": "fail", "family": family,
                "lint": res["lint"],
                "errors": ["%s on %s: %s" % (f["kind"], f.get("signal", "?"),
                                             f["hint"]) for f in res["lint"]]}
    return dict(res, stage="fpga", status="pass")


def summarize(res):
    if res is None:
        return "-"
    if res["stage"] == "sim":
        if res["status"] == "pass":
            return "pass ({} checks)".format(res["checks"])
        if res.get("phase") == "compile":
            return "FAIL compile"
        m = res["mismatches"][0] if res["mismatches"] else {}
        exp = m.get("expected_acc", m.get("expected_crc"))
        got = m.get("got_acc", m.get("got_crc"))
        return "FAIL {}: exp {} got {}".format(m.get("test", "?"), exp, got)
    if res["stage"] == "synth":
        return "pass ({} cells)".format(res["cell_count"]) if res["status"] == "pass" else "FAIL"
    if res["stage"] == "timing":
        if res["method"] == "opensta":
            return "{} (slack {:+.2f} ns)".format(res["status"], res["worst_slack_ns"])
        if res["method"] == "proxy_gate_depth":
            return "proxy (depth {})".format(res["gate_depth"])
        return "FAIL"
    if res["stage"] == "fpga":
        if res["status"] == "pass":
            return "pass ({} LUT, {} FF, {} DSP)".format(
                res["luts"], res["ffs"], res["dsps"])
        if res.get("lint"):
            f = res["lint"][0]
            return "FAIL {} {}".format(f["kind"], f.get("signal", ""))
        return "FAIL synth"
    return res["status"]


def derive_profile(spec, final):
    """Build the machine-readable profile from measured results only. These
    JSONs are the contract between the RTL half and the scaleout half."""
    sim, synth, tim = final["sim"], final["synth"] or {}, final["timing"]
    fpg = final.get("fpga") or {}
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
    unit = spec.get("unit", "mac")
    prof = {
        "block": spec["name"],
        "top_module": spec["top_module"],
        "unit": unit,
        "cycles_per_" + unit: thr.get("cycles_per_unit"),
        "latency_cycles": thr.get("latency_cycles"),
        "cell_count": synth.get("cell_count"),
        "area": synth.get("area"),
        "fmax_estimate_mhz": fmax,
        "fmax_method": method,
        "target_clock_mhz": target_mhz,
        "sim_checks_passed": sim.get("checks"),
    }
    if fpg.get("status") == "pass":
        # Real device primitives: what boards.py budgets against. The MAC
        # infers a DSP slice, the endpoint is pure LUT logic, and those are
        # different resources, so a single capacity number cannot fit both.
        prof["fpga"] = {k: fpg[k] for k in
                        ("family", "luts", "ffs", "dsps", "brams", "urams",
                         "carry", "muxf") if k in fpg}
    # Block-specific fields, keyed on the unit the spec declares rather
    # than on there being exactly two kinds of block.
    if "derivation" in spec:
        prof["derivation"] = spec["derivation"]
    if unit == "mac":
        prof["chiplet"] = spec["name"]
        prof["data_width"] = spec["parameters"]["data_width"]
        prof["acc_width"] = spec["parameters"]["acc_width"]
    elif unit == "layer":
        prof["layer"] = spec["name"]
        prof["bank"] = spec["parameters"]["bank"]
    elif unit == "tile":
        prof["memory"] = spec["name"]
        prof["capacity"] = spec["parameters"]["capacity"]
    elif unit == "column":
        prof["sequencer"] = spec["name"]
        prof["mac_stages"] = spec["parameters"]["mac_stages"]
    elif unit == "row":
        prof["row_unit"] = spec["name"]
        if "shift_bias" in spec["parameters"]:
            prof["shift_bias"] = spec["parameters"]["shift_bias"]
    elif unit == "score":
        prof["exp_unit"] = spec["name"]
        prof["in_frac"] = spec["parameters"]["in_frac"]
        prof["out_frac"] = spec["parameters"]["out_frac"]
    elif unit == "activation":
        prof["requant"] = spec["name"]
        prof["acc_width"] = spec["parameters"]["acc_width"]
        prof["out_width"] = spec["parameters"]["out_width"]
        prof["pipeline_stages"] = spec["parameters"]["pipeline_stages"]
    else:
        prof["bytes_per_cycle"] = spec["parameters"]["bytes_per_cycle"]
        # The link rate the synthesized endpoint can actually sustain:
        # bytes_per_cycle * fmax. Boards cap their transceiver rate at this.
        prof["endpoint_gbps"] = (spec["parameters"]["bytes_per_cycle"] * 8
                                 * fmax / 1000.0)
    return prof


def run_endpoint_flow(target_gbps=None, verbose=True, agent_kind=None):
    """Generate and sign off the fabric endpoint for a link rate.

    Architecture-level feedback: the agent loop fixes the RTL, but a
    datapath that cannot close timing at its clock is not an RTL bug, it is
    the wrong datapath. When a width and clock pair fails timing, this
    advances to the next standard option for the same rate (wider datapath,
    slower clock, then the flat XOR next-state form whose depth does not
    grow with the width) and runs the whole loop again, which is the call a
    human designer makes at exactly that point."""
    target = TARGET_LINK_GBPS if target_gbps is None else target_gbps
    rate, opts = specgen.endpoint_options(target)
    say = print if verbose else (lambda *a, **k: None)
    attempts = []
    for opt in range(len(opts)):
        w, clk, arch = opts[opt]
        say("\nEndpoint datapath option {}/{} for {:g} Gbps: {} bytes/cycle "
            "at {:g} MHz, {} next-state form".format(
                opt + 1, len(opts), rate, w, clk, arch))
        specgen.generate_endpoint(target, spec_file=FABRIC_JOB["spec_file"],
                                  tb_file=FABRIC_JOB["tb_file"], option=opt)
        job = dict(FABRIC_JOB)
        job.pop("derive_from_link")   # already generated for this option
        report, profile = run_flow(job, verbose=verbose,
                                   agent=make_agent(agent_kind))
        attempts.append({"option": opt, "bytes_per_cycle": w,
                         "clock_mhz": clk, "architecture": arch,
                         "converged": report["converged"]})
        if report["converged"]:
            report["datapath_attempts"] = attempts
            return report, profile
        t = report["final_metrics"]["timing"] or {}
        if t.get("worst_slack_ns") is not None:
            say("   timing missed by {:+.2f} ns at {:g} MHz, widening the "
                "datapath and slowing the clock".format(
                    t["worst_slack_ns"], clk))
    report["datapath_attempts"] = attempts
    return report, profile


def run_flow(job=None, verbose=True, agent=None, max_iters=None):
    """Run the agentic loop to convergence for one job (default: the compute
    chiplet). Returns (report, profile); profile is None without convergence."""
    job = job or CHIPLET_JOB
    max_iters = max_iters or int(os.environ.get("CHIPLET_MAX_ITERS", MAX_ITERS))
    say = print if verbose else (lambda *a, **k: None)
    os.makedirs(BUILD, exist_ok=True)
    shutil.copy(LIB, BUILD)
    if job.get("derive_from_model") == "mlp":
        specgen.generate_mlp(spec_file=job["spec_file"],
                             tb_file=job["tb_file"])
        from agent import (RuleBasedAgent, FIX_WIDTH, FIX_CLEAR,
                           FIX_CLRCOL, FIX_MEMLAT, FIX_SATURATE)
        os.makedirs(BUILD, exist_ok=True)
        _ms = specgen.load_model_spec()
        _r = RuleBasedAgent()
        for _fn, _src in (
                ("mac_dep.v", _r.render_mac(
                    specgen.derive_chiplet_spec(_ms),
                    {FIX_WIDTH, FIX_CLEAR})),
                ("mv_dep.v", _r.render_matvec(
                    specgen.derive_matvec_spec(_ms),
                    {FIX_CLRCOL, FIX_MEMLAT})),
                ("rq_dep.v", _r.render_requant(
                    specgen.derive_requant_spec(_ms), {FIX_SATURATE}))):
            with open(os.path.join(BUILD, _fn), "w") as f:
                f.write(_src)
    elif job.get("derive_from_model") == "softmax":
        specgen.generate_softmax(spec_file=job["spec_file"],
                                 tb_file=job["tb_file"])
        from agent import RuleBasedAgent, FIX_LUT, FIX_NORM
        os.makedirs(BUILD, exist_ok=True)
        _ms = specgen.load_model_spec()
        with open(os.path.join(BUILD, "expu_dep.v"), "w") as f:
            f.write(RuleBasedAgent().render_exp(
                specgen.derive_exp_spec(_ms), {FIX_LUT}))
        with open(os.path.join(BUILD, "recip_dep.v"), "w") as f:
            f.write(RuleBasedAgent().render_recip(
                specgen.derive_recip_spec(_ms), {FIX_NORM}))
    elif job.get("derive_from_model") == "wmem":
        specgen.generate_wmem(spec_file=job["spec_file"],
                              tb_file=job["tb_file"])
        from agent import (RuleBasedAgent, FIX_WIDTH, FIX_CLEAR,
                           FIX_CLRCOL, FIX_MEMLAT)
        os.makedirs(BUILD, exist_ok=True)
        _ms = specgen.load_model_spec()
        with open(os.path.join(BUILD, "mac_dep.v"), "w") as f:
            f.write(RuleBasedAgent().render_mac(
                specgen.derive_chiplet_spec(_ms), {FIX_WIDTH, FIX_CLEAR}))
        with open(os.path.join(BUILD, "matvec_dep.v"), "w") as f:
            f.write(RuleBasedAgent().render_matvec(
                specgen.derive_matvec_spec(_ms),
                {FIX_CLRCOL, FIX_MEMLAT}))
    elif job.get("derive_from_model") == "matvec":
        specgen.generate_matvec(spec_file=job["spec_file"],
                                tb_file=job["tb_file"])
        # Render the block it drives, from the same model spec, with the
        # deterministic agent so the dependency is reproducible.
        from agent import RuleBasedAgent, FIX_WIDTH, FIX_CLEAR
        os.makedirs(BUILD, exist_ok=True)
        with open(os.path.join(BUILD, "mac_dep.v"), "w") as f:
            f.write(RuleBasedAgent().render_mac(
                specgen.derive_chiplet_spec(specgen.load_model_spec()),
                {FIX_WIDTH, FIX_CLEAR}))
    elif job.get("derive_from_model") == "rsqrt":
        specgen.generate_rsqrt(spec_file=job["spec_file"],
                               tb_file=job["tb_file"])
    elif job.get("derive_from_model") == "recip":
        specgen.generate_recip(spec_file=job["spec_file"],
                               tb_file=job["tb_file"])
    elif job.get("derive_from_model") == "exp":
        specgen.generate_exp(spec_file=job["spec_file"],
                             tb_file=job["tb_file"])
    elif job.get("derive_from_model") == "requant":
        specgen.generate_requant(spec_file=job["spec_file"],
                                 tb_file=job["tb_file"])
    elif job.get("derive_from_model"):
        specgen.generate(spec_file=job["spec_file"], tb_file=job["tb_file"])
    if job.get("derive_from_link"):
        specgen.generate_endpoint(TARGET_LINK_GBPS,
                                  spec_file=job["spec_file"],
                                  tb_file=job["tb_file"])
    spec = json.load(open(os.path.join(ROOT, job["spec_file"])))
    tools = {t: tool(t) for t in ("iverilog", "vvp", "yosys", "sta")}
    say("Tools:", ", ".join("{}={}".format(k, v or "MISSING") for k, v in tools.items()))
    if not (tools["iverilog"] and tools["vvp"]):
        say("iverilog/vvp required for the demo loop, aborting")
        sys.exit(1)

    agent = agent or make_agent()
    say("Agent:", type(agent).__name__ + (
        " ({}@{})".format(agent.model, agent.backend)
        if hasattr(agent, "backend") else ""))
    history, iterations = [], []
    rows = [("iter", "fixes applied", "sim", "synth", "timing", "fpga")]
    converged = False

    for it in range(1, max_iters + 1):
        rtl, fixes = agent.propose(spec, history)
        rtl_path = os.path.join(BUILD, job["rtl_file"])
        with open(rtl_path, "w") as f:
            f.write(rtl)

        sim = stage_sim(job, rtl_path)
        synth = tim = fpg = None
        if sim["status"] == "pass" and tools["yosys"]:
            synth = stage_synth(job, spec, rtl_path)
            if synth["status"] == "pass":
                tim = stage_timing(job, spec)
                fpg = stage_fpga(job, spec, rtl_path)
        elif sim["status"] == "pass":
            synth = {"stage": "synth", "status": "skipped", "note": "yosys missing"}

        record = {"iteration": it, "fixes_applied": fixes,
                  "sim": sim, "synth": synth, "timing": tim, "fpga": fpg}
        iterations.append(record)
        for r in (sim, synth, tim, fpg):
            if r and r["status"] == "fail":
                history.append(dict(r, iteration=it))
        rows.append((str(it), ",".join(fixes) or "none", summarize(sim),
                     summarize(synth), summarize(tim), summarize(fpg)))

        ok = lambda r: r is not None and r["status"] in ("pass", "skipped")
        if ok(sim) and ok(synth) and (tim is None and synth["status"] == "skipped"
                                      or (ok(tim) and ok(fpg))):
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
        "spec": spec, "tools": tools, "agent": type(agent).__name__,
        "converged": converged,
        "iterations_used": len(iterations), "history": iterations,
        "final_metrics": {
            "sim_pass": final["sim"]["status"] == "pass",
            "sim_checks": final["sim"].get("checks"),
            "cell_count": (final["synth"] or {}).get("cell_count"),
            "area": (final["synth"] or {}).get("area"),
            "timing": final["timing"],
        },
    }
    rpath = os.path.join(ROOT, job["report_file"])
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)

    profile = None
    if converged:
        profile = derive_profile(spec, final)
        profile["agent"] = type(agent).__name__
        ppath = os.path.join(ROOT, job["profile_file"])
        with open(ppath, "w") as f:
            json.dump(profile, f, indent=2)
        say("\nCONVERGED in {} iteration(s), report written to {}".format(
            len(iterations), rpath))
        say("profile written to {}".format(ppath))
    else:
        say("\nDID NOT CONVERGE in {} iteration(s), report written to {}".format(
            len(iterations), rpath))
    return report, profile


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    kind = argv[argv.index("--agent") + 1] if "--agent" in argv else None
    # LLM first cuts are less predictable than the seeded bugs; allow more
    # feedback iterations before giving up.
    iters = 8 if (kind or os.environ.get("CHIPLET_AGENT")) in ("llm", "swarm") else None
    r1, _ = run_flow(CHIPLET_JOB, verbose=True,
                     agent=make_agent(kind), max_iters=iters)
    print()
    r2, _ = run_endpoint_flow(verbose=True, agent_kind=kind)
    sys.exit(0 if r1["converged"] and r2["converged"] else 2)


if __name__ == "__main__":
    main()
