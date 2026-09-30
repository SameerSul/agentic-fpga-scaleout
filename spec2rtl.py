"""Spec in, verified RTL out.

    python3 spec2rtl.py examples/qwen3_shape.json --board zc706
    python3 spec2rtl.py --weights qwen_weights --board zybo_z7_20 --package

The input is a transformer's shape, in model_spec.json's keys; with
--weights, a checkpoint directory instead, whose config.json is the shape.
What comes out, under --out, is the RTL of a decoder for it and report.md,
which says what checked each part:

  1. spec     the shape is checked against what the generator supports, and
              every block's parameters are derived from it (specgen.py).
  2. blocks   every generated block in the design, leaves first, goes
              through the signoff loop (chiplet_flow.py): an agent writes it,
              its self-checking testbench runs it against the bit-exact
              golden model, Yosys synthesizes it, OpenSTA times it at its
              clock and it maps to 7-series primitives, and every tool
              failure goes back to the agent. A composite block's testbench
              compiles the sub-blocks signed off before it, not copies.
  3. design   the whole decode step, embedding to argmax, is generated around
              exactly those files (qwen_full.py) and simulated against the
              integer model (qwen_int.py) on a checkpoint of the spec's
              shape: every chosen token and its logit has to match. The real
              weights with --weights; otherwise random ones, which check the
              arithmetic as well and say nothing about the text.
  4. board    with --package, the Zynq package for --board (board_zybo.py):
              the DDR bridge, the registers, the block design, the ARM
              program; --bridge also runs the simulated design through that
              package's registers against a DDR model that stalls at random.

Any block that does not converge, or any step that differs from the integer
model, fails the run, and report.md says which.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

import boards
import chiplet_flow as cf
import qwen_full
import qwen_int as qi
import qwen_real as qr
import qwen_synth
import specgen

ROOT = os.path.dirname(os.path.abspath(__file__))
SIM = "iverilog"              # run() sets it (vsim.pick)
SEQ_LEN = 256            # the integer model's and the sequencer's context
COUNTER_MAX = 8191       # the sequencer's 13-bit row and column counters


def normalize(spec):
    """The shape, with every default filled in. model_spec.json's keys."""
    s = dict(spec)
    for k in ("n_layer", "d_model", "n_head", "d_ff", "vocab"):
        if k not in s:
            raise ValueError("the spec has no %s" % k)
    s.setdefault("name", "model")
    s.setdefault("n_kv_head", s["n_head"])
    s.setdefault("head_dim", s["d_model"] // s["n_head"])
    s.setdefault("rope_theta", 1000000.0)
    s.setdefault("rms_norm_eps", 1e-6)
    s.setdefault("qk_norm", False)
    s.setdefault("qkv_bias", not s["qk_norm"])
    s.setdefault("weight_bits", 8)
    # The full sequencer runs 16-bit activations: at 8, a real Qwen's
    # residual stream does not fit (RESULTS.md, the integer model).
    s["activation_bits"] = s.get("activation_bits_sequencer", 16)
    return s


def problems(s, lanes):
    """What the generator cannot build for this shape, in words."""
    out = []
    D, F, H, KV, hd, V = (s["d_model"], s["d_ff"], s["n_head"], s["n_kv_head"],
                          s["head_dim"], s["vocab"])
    if s["weight_bits"] != 8:
        out.append("weight_bits %d: the sequencer and the DDR image are int8"
                   % s["weight_bits"])
    if H % KV:
        out.append("n_head %d is not a multiple of n_kv_head %d" % (H, KV))
    if hd % 2:
        out.append("head_dim %d is odd: RoPE turns pairs" % hd)
    if hd < lanes:
        out.append("head_dim %d is narrower than the core's %d lanes; pass "
                   "--lanes %d" % (hd, lanes, max(1, 1 << (hd.bit_length() - 1))))
    for what, rows in (("the query projection", H * hd), ("the key and value "
                       "projections", KV * hd), ("d_model", D), ("d_ff", F),
                       ("the vocabulary", V)):
        if rows % lanes:
            out.append("%s, %d rows, is not a multiple of the %d lanes"
                       % (what, rows, lanes))
    if max(D, F, H * hd) > COUNTER_MAX:
        out.append("a dimension past %d: the sequencer's counters are 13 bits"
                   % COUNTER_MAX)
    if s["qk_norm"] and s["qkv_bias"]:
        out.append("q/k norms and q/k/v biases together: neither Qwen has both, "
                   "and the generator has not been checked on it")
    if H * hd > max(D, F):
        out.append("q (%d wide) is wider than the widest matrix the projection "
                   "is sized for (%d)" % (H * hd, max(D, F)))
    return out


def hf_config(s):
    return {"hidden_size": s["d_model"], "intermediate_size": s["d_ff"],
            "num_attention_heads": s["n_head"],
            "num_key_value_heads": s["n_kv_head"], "head_dim": s["head_dim"],
            "num_hidden_layers": s["n_layer"], "vocab_size": s["vocab"],
            "rms_norm_eps": s["rms_norm_eps"], "rope_theta": s["rope_theta"]}


def checkpoint_name(wdir):
    """What to call a checkpoint: the fetched Qwens by name, otherwise its
    config's own name or its directory's."""
    if os.path.abspath(wdir) == os.path.abspath(qr.WDIR):
        return {"qwen2.5": "Qwen2.5-0.5B", "qwen3": "Qwen3-0.6B"}.get(qr.MODEL, qr.MODEL)
    try:
        cfg = json.load(open(os.path.join(wdir, "config.json")))
        return os.path.basename(cfg.get("_name_or_path", "").rstrip("/")) or \
            os.path.basename(os.path.abspath(wdir))
    except (OSError, ValueError):
        return os.path.basename(os.path.abspath(wdir))


def spec_of(cfg, W, name):
    """A checkpoint's shape, in the same keys, for its report."""
    return normalize({
        "name": name, "n_layer": cfg["num_hidden_layers"],
        "d_model": cfg["hidden_size"], "n_head": cfg["num_attention_heads"],
        "n_kv_head": cfg["num_key_value_heads"], "head_dim": qr.head_dim(cfg),
        "d_ff": cfg["intermediate_size"], "vocab": cfg["vocab_size"],
        "rope_theta": cfg["rope_theta"], "rms_norm_eps": cfg["rms_norm_eps"],
        "qk_norm": "model.layers.0.self_attn.q_norm.weight" in W,
        "qkv_bias": "model.layers.0.self_attn.q_proj.bias" in W})


# ---------------------------------------------------------------- blocks
# Every generated file the design compiles, leaves first: its design file
# name, what it is, its spec, its testbench, the files its testbench
# compiles with it, and the signoff unit. The tables (exp_rom and the rest)
# are written by specgen from the same specs and are checked by every
# testbench that reads them.
def block_plan(im):
    ms = im.ms
    sp = im.sp
    e, rc, rs = (specgen.derive_exp_spec(ms), specgen.derive_recip_spec(ms),
                 specgen.derive_rsqrt_spec(ms))
    plan = [
        ("mac_dep.v", "multiply-accumulate", specgen.derive_chiplet_spec(ms),
         specgen.render_testbench, ()),
        ("rq_dep.v", "requantizer", specgen.derive_requant_spec(ms),
         specgen.render_requant_testbench, ()),
        ("expu_dep.v", "exponential", e, specgen.render_exp_testbench,
         ("exp_rom.v",)),
        ("recip_dep.v", "reciprocal", rc, specgen.render_recip_testbench,
         ("recip_rom.v",)),
        ("rs_dep.v", "inverse square root", rs, specgen.render_rsqrt_testbench,
         ("rsqrt_rom.v",)),
        ("mv_dep.v", "matrix-vector sequencer", specgen.derive_matvec_spec(ms),
         specgen.render_matvec_testbench, ("mac_dep.v",)),
        ("sm_dep.v", "softmax", specgen.derive_softmax_spec(ms),
         specgen.render_softmax_testbench,
         ("expu_dep.v", "recip_dep.v", "exp_rom.v", "recip_rom.v")),
        # The score lanes' own MAC, q times k over head_dim: signed off as
        # the MAC it is, renamed mac_s for the head.
        ("smac_dep.v", "score multiply-accumulate", specgen.derive_score_mac_spec(ms),
         specgen.render_testbench, ()),
        ("b_projn.v", "projection, per column",
         specgen.derive_projn_spec(ms, per_column=True),
         specgen.render_projn_testbench, cf.PROJN_DEPS),
        ("b_attnn.v", "attention head", specgen.derive_attnn_spec(ms),
         specgen.render_attnn_testbench, cf.ATTN_DEPS),
        ("b_rmsnorm.v", "RMSNorm", sp["rmsnorm"],
         specgen.render_rmsnorm_testbench, cf.RMSNORM_DEPS),
        ("b_rope.v", "rotary embedding", sp["rope"],
         specgen.render_rope_testbench, ("rope_rom.v",)),
        ("b_silu.v", "SiLU", sp["silu"], specgen.render_silu_testbench,
         cf.SILU_DEPS),
        ("b_resadd.v", "residual add", sp["resadd"],
         specgen.render_resadd_testbench, ()),
    ]
    if im.qkn:
        # The same norm over one head: signed off as the rmsnorm it is,
        # then renamed for its second instance, as the sequencer names it.
        hn = json.loads(json.dumps(sp["headnorm"]))
        hn["top_module"] = "rmsnorm"
        plan.insert(11, ("b_rmsnorm_hd.v", "RMSNorm over a head", hn,
                         specgen.render_rmsnorm_testbench, cf.RMSNORM_DEPS))
    tables = {"exp_rom.v": specgen.exp_rom(e), "recip_rom.v": specgen.recip_rom(rc),
              "rsqrt_rom.v": specgen.rsqrt_rom(rs),
              "rope_rom.v": specgen.rope_roms(sp["rope"])}
    return plan, tables


def agent_chain(kind):
    """Who writes a block, in order of escalation: (label, factory, the
    iterations it gets). The LLM agent starts with the model in CHIPLET_LLM
    (claude-cli:haiku by default), escalates to Sonnet, and only then falls
    back to the rule-based agent; the report says which one signed each
    block off."""
    if kind != "llm":
        return [(kind, lambda: cf.make_agent(kind), None)]
    from llm_agent import LLMAgent
    from agent import RuleBasedAgent
    first = os.environ.get("CHIPLET_LLM") or "claude-cli:haiku"
    chain = [(first, lambda: LLMAgent(first), 8)]
    if first != "claude-cli:sonnet":
        chain.append(("claude-cli:sonnet", lambda: LLMAgent("claude-cli:sonnet"), 5))
    chain.append(("rules (fallback)", RuleBasedAgent, 5))
    return chain


def rename_module(src, old, new):
    """src's module old named new, however its header is written: an LLM
    writes "module mac(" or "module mac #(" as readily as "module mac (".
    An exact match missed one, left two modules named mac, and no agent
    could compile the attention head that instances both."""
    out, n = re.subn(r"\bmodule\s+%s\b" % re.escape(old), "module " + new, src, count=1)
    if not n:
        raise RuntimeError("no module %s to rename %s" % (old, new))
    return out


def keep_attempt(gates, tag, job, label):
    """An agent that did not converge leaves its report and its last draft
    under its own name, since the next agent in the chain writes over both:
    without them, why a model failed a block cannot be read back."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    for src, dst in ((job["report_file"], "report_%s.%s.json" % (tag, name)),
                     (os.path.join(gates, job["rtl_file"]), "rtl_%s.%s.v" % (tag, name))):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(gates, dst))


def sign_off(im, gates, agent_kind, log, dv=False):
    """Each block through the signoff loop, into gates/: its signed-off
    RTL under its design name. Returns one row per block. The flow's
    scratch directory is gates/ meanwhile, and the caller's afterwards: a
    run that left it pointing here sent every later flow in the process
    to a directory since removed."""
    old = cf.BUILD
    try:
        return _sign_off(im, gates, agent_kind, log, dv)
    finally:
        cf.BUILD = old


def _sign_off(im, gates, agent_kind, log, dv=False):
    plan, tables = block_plan(im)
    os.makedirs(gates, exist_ok=True)
    for fn, src in tables.items():
        with open(os.path.join(gates, fn), "w") as f:
            f.write(src)
    cf.BUILD = gates                  # the flow's scratch directory
    rows = []
    for fn, what, spec, tbf, deps in plan:
        tag = fn[:-2]
        job = {"spec_file": os.path.join(gates, "spec_%s.json" % tag),
               "tb_file": os.path.join(gates, "tb_%s.v" % tag),
               "rtl_file": "rtl_%s.v" % tag,
               "profile_file": os.path.join(gates, "profile_%s.json" % tag),
               "report_file": os.path.join(gates, "report_%s.json" % tag),
               "extra_sources": tuple(deps)}
        with open(job["spec_file"], "w") as f:
            json.dump(spec, f, indent=2)
        with open(job["tb_file"], "w") as f:
            f.write(tbf(spec))
        t0 = time.time()
        attempts, prev, report, err = [], None, None, None
        for label, make, iters in agent_chain(agent_kind):
            ag = make()
            if getattr(prev, "last_rtl", None) and hasattr(ag, "last_rtl"):
                ag.last_rtl = prev.last_rtl     # edit the last attempt, not restart
            try:
                report, profile = cf.run_flow(job, verbose=False, agent=ag, max_iters=iters)
            except RuntimeError as e:
                # An agent that gives no answer (a model's CLI timing out) has
                # failed its attempt: the next in the chain takes the block.
                err = e
                attempts.append([label, 0, False, str(e)[:200]])
                log("  %-26s %s gave no answer (%s); the next agent takes it"
                    % (what, label, str(e)[:60]))
                keep_attempt(gates, tag, job, label)
                prev = ag
                continue
            attempts.append([label, report["iterations_used"], report["converged"]])
            prev = ag
            if report["converged"]:
                break
            keep_attempt(gates, tag, job, label)
        if report is None:
            raise err
        last = report["history"][-1] if report["history"] else {}
        timing = (last.get("timing") or {})
        fpga = (last.get("fpga") or {})
        row = {"file": fn, "block": what, "converged": report["converged"],
               "agent": attempts[-1][0], "attempts": attempts,
               "iterations": report["iterations_used"],
               "fixes": sorted({f for it in report["history"]
                                for f in it["fixes_applied"]}),
               "checks": report["final_metrics"].get("sim_checks"),
               "cells": report["final_metrics"].get("cell_count"),
               "timing": timing.get("method"),
               "fmax": (profile or {}).get("fmax_estimate_mhz"),
               "target": spec["parameters"].get("target_clock_mhz"),
               "luts": ((profile or {}).get("fpga") or fpga).get("luts"),
               "dsps": ((profile or {}).get("fpga") or fpga).get("dsps"),
               "seconds": round(time.time() - t0, 1)}
        if report["converged"]:
            src = open(os.path.join(gates, job["rtl_file"])).read()
            if fn == "b_rmsnorm_hd.v":
                src = rename_module(src, "rmsnorm", "rmsnorm_hd")
            if fn == "smac_dep.v":
                src = rename_module(src, "mac", "mac_s")
            with open(os.path.join(gates, fn), "w") as f:
                f.write(src)
            if dv:
                import sweep
                sweep.BUILD = gates
                ok, detail = sweep.run_dv(os.path.join(gates, job["rtl_file"]),
                                          job["tb_file"], deps)
                row["dv"] = detail if ok else "FAIL: " + detail
        rows.append(row)
        log("  %-26s %s%s in %d iteration%s, %s checks, %s%s  (%.0f s)" % (
            what, "signed off" if report["converged"] else "DID NOT CONVERGE",
            " by " + row["agent"] if len(attempts) > 1 or agent_kind == "llm" else "",
            row["iterations"], "" if row["iterations"] == 1 else "s",
            row["checks"], ("%s %.0f MHz" % (row["timing"], row["fmax"]))
            if row["fmax"] else "no timing", ("; dv " + row["dv"]) if dv and
            "dv" in row else "", row["seconds"]))
        if not report["converged"]:
            break                     # the blocks above it depend on it
    return rows, [fn for fn, *_ in plan] + sorted(tables)


def use_signed_off(work, gates, files):
    """Put the signed-off files in the design, in place of the ones the
    generator wrote, and say which differed from them."""
    differed = []
    for fn in files:
        mine = os.path.join(work, fn)
        ours = open(os.path.join(gates, fn)).read()
        if os.path.exists(mine) and open(mine).read() != ours:
            differed.append(fn)
        with open(mine, "w") as f:
            f.write(ours)
    return differed


# ---------------------------------------------------------------- design
def simulate(work, srcs, n_prompt, log, timeout=None, sim="iverilog"):
    """The direct testbench: every head step's token and logit."""
    import vsim
    try:
        p = vsim.stream(work, srcs, sim, tag="q")
    except RuntimeError as e:
        return None, str(e)[-2000:]
    t0 = time.time()
    got, cycles = [], []
    for line in p.stdout:
        if line.startswith(("STEP", "PROGRESS")):
            log("    %s (%.0f s)" % (line.rstrip(), time.time() - t0))
        if line.startswith("STEP"):
            f = dict(kv.split("=") for kv in line.split()[1:])
            cycles.append(int(f["cycles"]))
            if int(f["pos"]) >= n_prompt - 1:
                got.append((int(f["next"]), int(f["best"])))
    p.wait()
    return (got, cycles), None


def reference(im, want, n_prompt):
    """The integer model's own head steps: its argmax and its logit."""
    im.reset()
    best = []
    for p in range(len(want) - 1):
        lg = im.step(want[p], p, logits=p >= n_prompt - 1)
        if lg is not None:
            best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
    return best


# ---------------------------------------------------------------- the run
def run(spec=None, board="zybo_z7_20", out=None, weights=None, lanes=None,
        sim_layers=None, gen=2, agent="rules", package=False, bridge=False,
        dv=False, prompt="The capital of France is", seed=5, log=print,
        cluster=None, mode="balanced", split="layers", sim="fast"):
    """One board, or with cluster (a list of boards, any mix) the whole
    heterogeneous pipeline: see run_cluster. sim: the simulator (vsim.py);
    "fast", the default, is Verilator where it is installed, and then a
    real model is simulated at every layer, not one."""
    import vsim
    global SIM
    SIM = vsim.pick(sim)
    out = os.path.abspath(out or os.path.join(ROOT, "build_spec2rtl"))
    os.makedirs(out, exist_ok=True)
    rep = {"board": board, "agent": agent, "stages": {}, "simulator": SIM}
    t_all = time.time()
    lanes = lanes or boards.PACKAGES[board]["lanes"]

    # ---- 1. spec
    log("1. spec")
    tok = None
    if weights:
        cfg, W = qr.load(weights)
        s = spec_of(cfg, W, checkpoint_name(weights))
        if os.path.exists(os.path.join(weights, "tokenizer.json")):
            tok = qr.Tokenizer(weights)
    else:
        s = normalize(spec)
    bad = [] if cluster and len(cluster) > 1 else problems(s, lanes)
    rep.update(spec=s, lanes=lanes)
    if bad:
        rep["stages"]["spec"] = {"ok": False, "problems": bad}
        log("  cannot build this shape:\n    " + "\n    ".join(bad))
        return finish(out, rep, t_all, log)
    # Qwen3's q/k norms, Qwen2.5's q/k/v biases, or neither.
    style = "qwen3" if s["qk_norm"] else "qwen2.5" if s["qkv_bias"] else "plain"
    log("  %s: %d layers, d_model %d, %d heads of %d over %d KV heads, d_ff %d, "
        "vocab %d, %s" % (
            s["name"], s["n_layer"], s["d_model"], s["n_head"], s["head_dim"],
            s["n_kv_head"], s["d_ff"], s["vocab"],
            "q/k norms" if s["qk_norm"] else "q/k/v biases"))
    if not weights:
        cfg = hf_config(s)
        t0 = time.time()
        W = qwen_synth.weights(cfg, style, seed)
        log("  random weights of this shape (seed %d) in %.0f s" % (seed, time.time() - t0))
    t0 = time.time()
    if weights and os.path.abspath(weights) == os.path.abspath(qr.WDIR):
        import qwen_cosim
        cal = qwen_cosim.calibration(cfg, W, tok)   # cached beside the weights
    elif tok:
        cal = qi.calibrate(cfg, W, tok, log=lambda *a: None)
    else:
        rnd = random.Random(seed + 1)
        cal = qi.calibrate(cfg, W, qwen_synth._Ids(
            [rnd.randrange(s["vocab"]) for _ in range(12)]), log=lambda *a: None)
    log("  calibrated in %.0f s" % (time.time() - t0))
    if cluster and len(cluster) > 1:
        if split == "weights":
            return run_tp(s, cfg, W, cal, tok, cluster, out, rep, t_all, sim_layers,
                          gen, agent, dv, prompt, seed, log, package, bridge,
                          even=mode == "even")
        done, board = run_cluster(s, cfg, W, cal, tok, cluster, mode, out, rep,
                                  t_all, sim_layers, gen, agent, package, dv,
                                  prompt, seed, log)
        if done:
            return done
        # The plan put every layer on one board: that board's own run.
        lanes = boards.PACKAGES[board]["lanes"]
        rep.update(board=board, lanes=lanes)
        bad = problems(s, lanes)
        if bad:
            rep["stages"]["spec"] = {"ok": False, "problems": bad}
            return finish(out, rep, t_all, log)
    log("  %d lanes for the %s" % (lanes, boards.PACKAGES[board]["title"]))
    NL = s["n_layer"]
    if sim_layers == "all":
        k = NL
    elif sim_layers is None:
        # A real model's size takes hours a layer set in Icarus; one layer
        # checks the same generator and the same blocks. Verilator runs
        # them all in minutes.
        k = NL if s["d_model"] <= 256 or SIM == "verilator" else 1
    else:
        k = max(1, min(int(sim_layers), NL))
    im = qi.IntQwen(dict(cfg, num_hidden_layers=k), W, 16, True, cal,
                    log=lambda *a: None, exact_io=True)
    im.ms["lanes"] = lanes
    rep["stages"]["spec"] = {"ok": True, "seq_len": SEQ_LEN}

    # ---- 2. blocks
    log("2. blocks, through the signoff gates (%s agent)" % agent)
    gates = os.path.join(out, "gates")
    rows, files = sign_off(im, gates, agent, log, dv)
    ok_blocks = all(r["converged"] for r in rows) and \
        len(rows) == len(block_plan(im)[0])
    rep["stages"]["blocks"] = {"ok": ok_blocks, "rows": rows}
    if not ok_blocks:
        return finish(out, rep, t_all, log)

    # ---- 3. design, simulated against the integer model
    log("3. design: the decode step at %d of %d layers, against the integer model"
        % (k, NL))
    rnd = random.Random(seed + 2)
    ids = tok.encode(prompt) if tok else [rnd.randrange(s["vocab"]) for _ in range(4)]
    work = os.path.join(out, "design")
    shutil.rmtree(work, ignore_errors=True)
    want, srcs = qwen_full.build_model(im, ids, gen, work, log=lambda *a: None)
    differed = use_signed_off(work, gates, files)
    ref = reference(im, want, len(ids))
    res, err = simulate(work, srcs, len(ids), log, sim=SIM)
    if err:
        rep["stages"]["design"] = {"ok": False, "error": err}
        return finish(out, rep, t_all, log)
    got, cycles = res
    ok_design = got == ref and len(got) == gen
    pred = [predicted(s, lanes, k, p + 1, len(ids), True, True)
            - predicted(s, lanes, k, p, len(ids), True, True) for p in range(len(cycles))]
    rep["stages"]["design"] = {
        "ok": ok_design, "layers": k, "of": NL, "prompt": ids, "steps": len(cycles),
        "rtl": [t for t, _ in got], "integer_model": [t for t, _ in ref],
        "logits_rtl": [b for _, b in got], "logits_integer_model": [b for _, b in ref],
        "cycles": cycles, "predicted_cycles": pred,
        "prediction_error_pct": [round(100.0 * (c - q) / c, 3) for c, q in zip(cycles, pred)],
        "generator_files_replaced": differed,
        "text": tok.decode([t for t, _ in got]) if tok else None}
    log("  RTL %s, integer model %s: %s" % (got, ref, "MATCH" if ok_design else "MISMATCH"))
    if not ok_design:
        return finish(out, rep, t_all, log)

    # ---- 4. the full-depth design, and the board
    full = work
    if k < NL:
        log("4. the full %d-layer design (images and RTL)" % NL)
        imf = qi.IntQwen(cfg, W, 16, True, cal, log=lambda *a: None, exact_io=True)
        imf.ms["lanes"] = lanes
        full = os.path.join(out, "design_full")
        shutil.rmtree(full, ignore_errors=True)
        qwen_full.build_model(imf, ids, 0, full, log=lambda *a: None,
                              want=list(ids) + [0])
        use_signed_off(full, gates, files)
    rep["rtl"] = full
    if package or bridge:
        import board_zybo
        pk = boards.PACKAGES[board]
        if lanes != pk.get("lanes"):
            rep["stages"]["board"] = {"ok": False, "error": "the %s's package is "
                                      "for a %s-lane core, this one has %d"
                                      % (pk["title"], pk.get("lanes"), lanes)}
            return finish(out, rep, t_all, log)
        if not pk.get("ps7"):
            rep["stages"]["board"] = {"ok": False, "error": "%s has no Zynq PS: "
                                      "no package is generated for it" % pk["title"]}
            return finish(out, rep, t_all, log)
        checked = ("every generated block signed off at this shape (simulation "
                   "against its golden model, synthesis, timing, 7-series "
                   "mapping), and the decode step at %d of %d layers choosing "
                   "the integer model's tokens and logits on %s weights "
                   "(spec2rtl.py, report.md)." % (k, NL, "the checkpoint's" if
                                                  weights else "random"))
        if package:
            log("4. the %s package" % pk["title"])
            board_zybo.package(full, board, os.path.join(out, "board"), prompt,
                               16, sd=bool(tok), model=s["name"], checked=checked)
            rep["package"] = os.path.join(out, "board")
        if bridge:
            log("4. the simulated design through the %s's registers, stalling DDR"
                % pk["title"])
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                board_zybo.package(work, board, os.path.join(out, "board_sim"),
                                   prompt, 16, sd=False, sim=True, jitter=True, simulator=SIM,
                                   model=s["name"], checked=checked)
            steps = [l for l in buf.getvalue().splitlines() if l.startswith("STEP")]
            nxt = [int(l.split("next=")[1].split()[0]) for l in steps
                   if int(l.split("pos=")[1].split()[0]) >= len(ids) - 1]
            bus = [int(l.split("clk_cycles=")[1].split()[0]) for l in steps]
            core = [int(l.split("core_cycles=")[1].split()[0]) for l in steps]
            ok_b = nxt == [t for t, _ in ref]
            rep["stages"]["bridge"] = {"ok": ok_b, "tokens": nxt, "bus_cycles": bus,
                                       "core_cycles": core}
            log("  through the registers: %s, %s bus cycles a core cycle"
                % ("MATCH" if ok_b else "MISMATCH",
                   "%.2f" % (sum(bus) / max(1, sum(core)))))
    return finish(out, rep, t_all, log)


def shape_of(s):
    return dict(D=s["d_model"], F=s["d_ff"], H=s["n_head"], KV=s["n_kv_head"],
                hd=s["head_dim"], NL=s["n_layer"], V=s["vocab"], qkn=s["qk_norm"])


def predicted(s, lanes, n_layers, npos, n_prompt, emb, head):
    """The planner's cycle model (cluster.stage_cycles) for the positions
    a simulation ran: context pos + 1 at each, the head where it ran."""
    import cluster
    return sum(cluster.stage_cycles(shape_of(s), lanes, n_layers, p + 1, emb,
                                    head and p >= n_prompt - 1) for p in range(npos))


def run_cluster(s, cfg, W, cal, tok, names, mode, out, rep, t_all, sim_layers,
                gen, agent, package, dv, prompt, seed, log):
    """The heterogeneous scale-out: the planner (cluster.py) splits the
    layers over the boards by each one's speed and memory, each board's
    core at its own width; the blocks are signed off at every width in the
    cluster; the pipeline of stages, each on its own clock and sharing only
    CRC-checked messages (gals.py), is generated around those files and
    simulated against the one-board integer model; the planner's estimate
    of every stage's cycles is checked against that simulation; and with
    package, every board's package, its stage's layers and its place in
    the chain. Returns (report, None), or (None, board) when the plan puts
    every layer on one board."""
    import cluster
    import gals
    try:
        p = cluster.plan(list(names), mode=mode, shape=shape_of(s))
    except ValueError as e:
        rep["stages"]["spec"] = {"ok": False, "problems": [str(e)]}
        return finish(out, rep, t_all, log), None
    st = p["stages"]
    if len(st) == 1:
        log("  the plan puts every layer on the %s" % st[0]["board"])
        return None, st[0]["board"]
    widths = [boards.PACKAGES[x["board"]]["lanes"] for x in st]
    rep.update(plan=p, lanes=widths, board=" + ".join(x["board"] for x in st))
    bad = sorted({q for N in set(widths) for q in problems(s, N)})
    if bad:
        rep["stages"]["spec"] = {"ok": False, "problems": bad}
        log("  cannot build this shape:\n    " + "\n    ".join(bad))
        return finish(out, rep, t_all, log), None
    log("  plan (%s): %s" % (mode, ", then ".join(
        "%s layers %d-%d at %d lanes%s" % (x["board"], x["layers"][0], x["layers"][1] - 1,
                                            N, " +head" if x["head"] else "")
        for x, N in zip(st, widths))))
    if p["cannot_hold"]:
        log("  cannot hold a layer: %s" % ", ".join(p["cannot_hold"]))
    if sim_layers == "all" or (sim_layers is None and (s["d_model"] <= 256
                                                        or SIM == "verilator")):
        split = [list(range(*x["layers"])) for x in st]
    else:
        per = 1 if sim_layers is None else max(1, int(sim_layers))
        split = [list(range(i * per, (i + 1) * per)) for i in range(len(st))]
    k = sum(len(x) for x in split)
    im = qi.IntQwen(dict(cfg, num_hidden_layers=k), W, 16, True, cal,
                    log=lambda *a: None, exact_io=True)
    rep["stages"]["spec"] = {"ok": True, "seq_len": SEQ_LEN}

    # ---- 2. blocks, at every width the cluster has
    gates, rows_all, files = {}, [], None
    for N in sorted(set(widths)):
        log("2. blocks at %d lanes, through the signoff gates (%s agent)" % (N, agent))
        im.ms["lanes"] = N
        g = os.path.join(out, "gates_%d" % N)
        rows, files = sign_off(im, g, agent, log, dv)
        for r in rows:
            r["lanes"] = N
        rows_all += rows
        gates[N] = g
        if not (all(r["converged"] for r in rows) and len(rows) == len(block_plan(im)[0])):
            rep["stages"]["blocks"] = {"ok": False, "rows": rows_all}
            return finish(out, rep, t_all, log), None
    same = [fn for fn in files if fn not in ("b_projn.v", "b_attnn.v") and
            len({open(os.path.join(g, fn)).read() for g in gates.values()}) == 1]
    rep["stages"]["blocks"] = {"ok": True, "rows": rows_all,
                               "shared_across_widths": len(same)}

    # ---- 3. the pipeline, against the one-board integer model
    log("3. the %d-board pipeline at %d of %d layers, against the one-board "
        "integer model" % (len(st), k, s["n_layer"]))
    rnd = random.Random(seed + 2)
    ids = tok.encode(prompt) if tok else [rnd.randrange(s["vocab"]) for _ in range(4)]
    work = os.path.join(out, "cluster")
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    want, srcs = gals.build(im, ids, gen, work, split, lanes=widths, log=lambda *a: None)
    # The signed-off files: each stage's projection and attention at its
    # own width, under that stage's module names; the rest shared.
    first = gates[min(widths)]
    for fn in files:
        if fn not in ("b_projn.v", "b_attnn.v") and os.path.exists(os.path.join(work, fn)):
            shutil.copyfile(os.path.join(first, fn), os.path.join(work, fn))
    for i, N in enumerate(widths):
        pch = chr(ord("a") + i)
        for m in ("projn", "attnn"):
            src = open(os.path.join(gates[N], "b_%s.v" % m)).read().replace(
                "module %s (" % m, "module %s_%s (" % (m, pch), 1)
            with open(os.path.join(work, "b_%s_%s.v" % (m, pch)), "w") as f:
                f.write(src)
    txt = gals.run(work, srcs, timeout=48 * 3600, sim=SIM)
    got = [(int(l.split("tok=")[1].split()[0]), int(l.split("best=")[1].split()[0]))
           for l in txt.splitlines() if l.startswith("TOKEN")]
    link = next((l for l in txt.splitlines() if l.startswith("LINK")), "")
    busy = [int(l.split("busy_cycles=")[1]) for l in txt.splitlines()
            if l.startswith("STAGE")]
    ref = reference(im, want, len(ids))
    ok = got == ref and len(got) == gen and "crc_errors=0 host_bad=0" in link
    if not link:
        rep["stages"]["design"] = {"ok": False, "boards": len(st), "error": txt[-3000:]}
        log("  the pipeline did not run:\n" + txt[-1500:])
        return finish(out, rep, t_all, log), None
    # The planner's cycle model for the same run, stage by stage.
    npos = len(want) - 1
    pred = [predicted(s, N, len(lay), npos, len(ids), i == 0, i == len(split) - 1)
            for i, (lay, N) in enumerate(zip(split, widths))]
    err = [round(100.0 * (b - c) / b, 1) if b else None for b, c in zip(busy, pred)]
    rep["stages"]["design"] = {
        "ok": ok, "layers": k, "of": s["n_layer"], "boards": len(st), "split": split,
        "lanes": widths, "prompt": ids, "steps": npos,
        "rtl": [t for t, _ in got], "integer_model": [t for t, _ in ref],
        "logits_rtl": [b for _, b in got], "logits_integer_model": [b for _, b in ref],
        "link": link, "busy_cycles": busy, "predicted_cycles": pred,
        "prediction_error_pct": err, "seconds": round(time.time() - t0),
        "text": tok.decode([t for t, _ in got]) if tok else None}
    log("  pipeline %s, one-board integer model %s, %s: %s" % (
        got, ref, link, "MATCH" if ok else "MISMATCH"))
    log("  stage cycles %s against the planner's %s (%s%% off)" % (busy, pred, err))
    if not ok:
        return finish(out, rep, t_all, log), None

    # ---- 4. every board's package
    if package:
        import board_zybo
        imf = qi.IntQwen(cfg, W, 16, True, cal, log=lambda *a: None, exact_io=True)
        n = len(st)
        ips = [(192, 168, 1, 10 + i) for i in range(n)]
        outs = []
        for i, (x, N) in enumerate(zip(st, widths)):
            pk = boards.PACKAGES[x["board"]]
            if not pk.get("ps7"):
                log("  %s has no ARM: its stage runs gals.stage_ctrl over the fabric "
                    "UART; no Zynq package" % pk["title"])
                continue
            imf.ms["lanes"] = N
            layers = list(range(*x["layers"]))
            wk = os.path.join(out, "stage%d_build" % i)
            shutil.rmtree(wk, ignore_errors=True)
            qwen_full.build_model(imf, ids, 0, wk, log=lambda *a: None, layers=layers,
                                  stage=True, want=list(ids) + [0],
                                  table=x["emb"] or x["head"])
            use_signed_off(wk, gates[N], files)
            o = os.path.join(out, "stage%d_%s" % (i, x["board"]))
            board_zybo.package(wk, x["board"], o, prompt, 16, sd=bool(tok), stage=dict(
                D=imf.D, index=i, count=n, l0=layers[0], l1=layers[-1] + 1,
                emb=x["emb"], head=x["head"], ip=ips[i], next_ip=ips[(i + 1) % n],
                first_ip=ips[0]), model=s["name"], checked=(
                    "every generated block signed off at this board's %d lanes, and "
                    "the %d-board pipeline this stage belongs to simulated, each "
                    "board on its own clock, choosing the one-board integer "
                    "model's tokens and logits (spec2rtl.py, report.md)." % (N, n)))
            outs.append(o)
            log("  stage %d: %s, layers %d-%d, %d lanes: %s" % (
                i, pk["title"], layers[0], layers[-1], N, o))
        with open(os.path.join(out, "plan.json"), "w") as f:
            json.dump(dict(p, ips=ips, packages=outs), f, indent=1)
        rep["packages"] = outs
    return finish(out, rep, t_all, log), None


def tp_problems(s, widths, part):
    """What this split of the weights over these ranks cannot build."""
    out = []
    grp = s["n_head"] // s["n_kv_head"]
    for r, N in enumerate(widths):
        for what, rows in (("q rows", grp * part["kv"][r] * s["head_dim"]),
                           ("k and v rows", part["kv"][r] * s["head_dim"]),
                           ("share of d_model", part["d"][r]), ("share of d_ff", part["f"][r])):
            if rows % N or rows <= 0:
                out.append("rank %d's %s, %d, is not a positive multiple of its %d lanes"
                           % (r, what, rows, N))
    return out


def _tp_signed_off(work, gates, files, widths):
    """tp.build's sources with the signed-off blocks in: each rank's
    projection and attention head at its own width, the rest shared."""
    first = gates[min(widths)]
    for fn in files:
        if fn not in ("b_projn.v", "b_attnn.v") and os.path.exists(os.path.join(work, fn)):
            shutil.copyfile(os.path.join(first, fn), os.path.join(work, fn))
    for i, N in enumerate(widths):
        pch = chr(ord("a") + i)
        for m in ("projn", "attnn"):
            src = open(os.path.join(gates[N], "b_%s.v" % m)).read().replace(
                "module %s (" % m, "module %s_%s (" % (m, pch), 1)
            with open(os.path.join(work, "b_%s_%s.v" % (m, pch)), "w") as f:
                f.write(src)


def run_tp(s, cfg, W, cal, tok, names, out, rep, t_all, sim_layers, gen, agent,
           dv, prompt, seed, log, package=False, bridge=False, even=False):
    """The weights split over the boards (tp.py): every board on every
    layer with a slice of every matrix, each at its own width and on its own
    clock, gathering each other's slices; the whole group simulated against
    the one-board integer model. bridge: the same ranks again as their
    packages run them, each through its register block and DDR bridge
    with a stalling DDR model of its own, every gather done by the ARM's
    side through the registers. package: every rank's Zynq package."""
    import tp
    import cluster
    widths = [boards.PACKAGES[n]["lanes"] for n in names]
    rep.update(lanes=widths, board=" + ".join(names), split="weights")
    bad = sorted({q for N in set(widths) for q in problems(s, N)})
    plan = None
    if not bad:
        try:
            plan = cluster.tp_plan(names, shape_of(s), even=even)
        except ValueError as e:
            bad = [str(e)]
    if plan:
        bad += tp_problems(s, widths, plan["part"])
    if bad:
        rep["stages"]["spec"] = {"ok": False, "problems": bad}
        log("  cannot split this shape's weights:\n    " + "\n    ".join(bad))
        return finish(out, rep, t_all, log)
    T = len(names)
    part = plan["part"]
    rep["tp_plan"] = plan
    log("  the weights split %d ways, %s: %s" % (T, "evenly" if even else "by speed", ", ".join(
        "%s at %d lanes: %d KV head%s, %d of d_ff, %d of d_model, %s"
        % (n, N, part["kv"][r], "" if part["kv"][r] == 1 else "s", part["f"][r], part["d"][r],
           "head chunks %d-%d" % tuple(part["hk"][r]) if part["hk"][r][1] >= part["hk"][r][0]
           else "no head chunk")
        for r, (n, N) in enumerate(zip(names, widths)))))
    log("  planner: %.1f ms a layer, gathers about %.1f ms more, %.3f s a token"
        % (1e3 * plan["layer_seconds"], 1e3 * plan["gather_seconds"], plan["seconds_per_token"]))
    NL = s["n_layer"]
    if sim_layers == "all" or (sim_layers is None and (s["d_model"] <= 256
                                                        or SIM == "verilator")):
        k = NL
    else:
        k = 1 if sim_layers is None else max(1, min(int(sim_layers), NL))
    im = qi.IntQwen(dict(cfg, num_hidden_layers=k), W, 16, True, cal,
                    log=lambda *a: None, exact_io=True)
    rep["stages"]["spec"] = {"ok": True, "seq_len": SEQ_LEN}
    gates, rows_all, files = {}, [], None
    for N in sorted(set(widths)):
        log("2. blocks at %d lanes, through the signoff gates (%s agent)" % (N, agent))
        im.ms["lanes"] = N
        g = os.path.join(out, "gates_%d" % N)
        rows, files = sign_off(im, g, agent, log, dv)
        for r in rows:
            r["lanes"] = N
        rows_all += rows
        gates[N] = g
        if not (all(r["converged"] for r in rows) and len(rows) == len(block_plan(im)[0])):
            rep["stages"]["blocks"] = {"ok": False, "rows": rows_all}
            return finish(out, rep, t_all, log)
    rep["stages"]["blocks"] = {"ok": True, "rows": rows_all}
    log("3. the %d ranks at %d of %d layers, against the one-board integer model"
        % (T, k, NL))
    rnd = random.Random(seed + 2)
    ids = tok.encode(prompt) if tok else [rnd.randrange(s["vocab"]) for _ in range(4)]
    work = os.path.join(out, "tp")
    shutil.rmtree(work, ignore_errors=True)
    t0 = time.time()
    want, srcs = tp.build(im, ids, gen, work, T, lanes=widths, log=lambda *a: None, part=part)
    _tp_signed_off(work, gates, files, widths)
    txt = tp.run(work, srcs, timeout=48 * 3600, sim=SIM)
    got = [(int(l.split("tok=")[1].split()[0]), int(l.split("best=")[1].split()[0]))
           for l in txt.splitlines() if l.startswith("TOKEN")]
    net = next((l for l in txt.splitlines() if l.startswith("TP ")), "")
    rk = [l for l in txt.splitlines() if l.startswith("RANK")]
    busy = [int(l.split("busy_cycles=")[1].split()[0]) for l in rk]
    comp = [int(l.split("compute_cycles=")[1].split()[0]) for l in rk]
    import cluster
    grp = s["n_head"] // s["n_kv_head"]
    pred = [sum(cluster.tp_rank_cycles(
        shape_of(s), N, T, r, k, p + 1, p >= len(ids) - 1,
        dict(H=grp * part["kv"][r], KV=part["kv"][r], F=part["f"][r], D=part["d"][r]),
        part["hk"][r]) for p in range(len(want) - 1)) for r, N in enumerate(widths)]
    err = [round(100.0 * (b - c) / b, 1) if b else None for b, c in zip(comp, pred)]
    if not net:
        rep["stages"]["design"] = {"ok": False, "boards": T, "error": txt[-3000:]}
        log("  the ranks did not run:\n" + txt[-1500:])
        return finish(out, rep, t_all, log)
    ref = reference(im, want, len(ids))
    ok = got == ref and len(got) == gen and "bad=0" in net
    rep["stages"]["design"] = {
        "ok": ok, "layers": k, "of": NL, "boards": T, "split_weights": True,
        "lanes": widths, "prompt": ids, "steps": len(want) - 1,
        "rtl": [t for t, _ in got], "integer_model": [t for t, _ in ref],
        "logits_rtl": [b for _, b in got], "logits_integer_model": [b for _, b in ref],
        "link": net, "busy_cycles": busy, "compute_cycles": comp, "predicted_cycles": pred,
        "prediction_error_pct": err, "seconds": round(time.time() - t0),
        "text": tok.decode([t for t, _ in got]) if tok else None}
    log("  ranks %s, one-board integer model %s, %s: %s" % (
        got, ref, net, "MATCH" if ok else "MISMATCH"))
    log("  rank compute cycles %s against the cycle model's %s (%s%% off)" % (comp, pred, err))
    if not ok:
        return finish(out, rep, t_all, log)
    no_arm = [n for n in names if not boards.PACKAGES[n].get("ps7")]
    if (package or bridge) and no_arm:
        rep["stages"]["board"] = {"ok": False, "error": "%s has no ARM to run a rank's "
                                  "gathers: no package" % boards.PACKAGES[no_arm[0]]["title"]}
        return finish(out, rep, t_all, log)
    if bridge:
        log("4. the %d ranks through their registers and DDR bridges, stalling DDR" % T)
        wb = os.path.join(out, "tp_boards")
        shutil.rmtree(wb, ignore_errors=True)
        t0 = time.time()
        want_b, srcs_b = tp.build_boards(im, ids, gen, wb, T, lanes=widths, jit=1,
                                         log=lambda *a: None, part=part)
        _tp_signed_off(wb, gates, files, widths)
        txt = tp.run(wb, srcs_b, timeout=48 * 3600, sim=SIM)
        got_b = [(int(l.split("tok=")[1].split()[0]), int(l.split("best=")[1].split()[0]))
                 for l in txt.splitlines() if l.startswith("TOKEN")]
        cyc = [(int(l.split("core_cycles=")[1].split()[0]), int(l.split("clk_cycles=")[1]))
               for l in txt.splitlines() if l.startswith("RANK")]
        net_b = next((l for l in txt.splitlines() if l.startswith("TP ")), "")
        ok_b = got_b == ref and "bad=0" in net_b
        rep["stages"]["bridge"] = {"ok": ok_b, "tokens": [t for t, _ in got_b],
                                   "logits": [b for _, b in got_b], "link": net_b,
                                   "core_cycles": [c for c, _ in cyc],
                                   "bus_cycles": [b for _, b in cyc],
                                   "seconds": round(time.time() - t0),
                                   "error": None if net_b else txt[-2000:]}
        log("  through the registers: %s, %s: %s" % (got_b, net_b or txt[-600:],
                                                     "MATCH" if ok_b else "MISMATCH"))
        if not ok_b:
            return finish(out, rep, t_all, log)
    if package:
        import board_zybo
        log("5. every rank's package")
        imf = qi.IntQwen(cfg, W, 16, True, cal, log=lambda *a: None, exact_io=True)
        ips = [(192, 168, 1, 10 + i) for i in range(T)]
        outs = []
        for i, (n, N) in enumerate(zip(names, widths)):
            imf.ms["lanes"] = N
            wk = os.path.join(out, "rank%d_build" % i)
            shutil.rmtree(wk, ignore_errors=True)
            qwen_full.build_model(imf, ids, 0, wk, log=lambda *a: None,
                                  want=list(ids) + [0], tp=(i, T, part))
            use_signed_off(wk, gates[N], files)
            o = os.path.join(out, "rank%d_%s" % (i, n))
            board_zybo.package(wk, n, o, prompt, 16, sd=bool(tok), tp=dict(
                rank=i, ranks=T, HHD=imf.H * imf.hd, D=imf.D, F=imf.F, ips=ips,
                slices=[(grp * part["kv"][r] * imf.hd, part["d"][r], part["f"][r])
                        for r in range(T)]),
                model=s["name"], checked=(
                    "every generated block signed off at this board's %d lanes, the "
                    "%d ranks this one belongs to simulated each on its own clock%s, "
                    "choosing the one-board integer model's tokens and logits, and "
                    "the ARM program's gathers run over UDP on a host "
                    "(spec2rtl.py, report.md; tests.py)." % (
                        N, T, ", each through its registers and DDR bridge against a "
                        "stalling DDR model" if bridge else "")))
            outs.append(o)
            log("  rank %d: %s, %d lanes: %s" % (i, boards.PACKAGES[n]["title"], N, o))
        with open(os.path.join(out, "ranks.json"), "w") as f:
            json.dump(dict(boards=names, lanes=widths, ips=ips, packages=outs), f, indent=1)
        rep["packages"] = outs
    return finish(out, rep, t_all, log)


def finish(out, rep, t_all, log):
    rep["ok"] = all(v.get("ok") for v in rep["stages"].values()) and \
        {"spec", "blocks", "design"} <= set(rep["stages"])
    rep["seconds"] = round(time.time() - t_all)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(render_report(rep))
    log("%s: %s (report.md in %s)" % ("VERIFIED" if rep["ok"] else "FAILED",
                                      summary(rep), out))
    return rep


def summary(rep):
    st = rep["stages"]
    if not st.get("spec", {}).get("ok"):
        return "the spec cannot be built"
    b = st.get("blocks", {})
    n = sum(r["converged"] for r in b.get("rows", []))
    parts = ["%d blocks signed off" % n]
    d = st.get("design")
    if d and d.get("boards") and "error" in d:
        parts.append("the %d-board pipeline did not run" % d["boards"])
    elif d and d.get("split_weights"):
        parts.append("the weights split %d ways (%s lanes) %s at %d of %d layers" % (
            d["boards"], "/".join(map(str, d["lanes"])),
            "matches the one-board integer model" if d.get("ok") else "DIFFERS",
            d["layers"], d["of"]))
    elif d and d.get("boards"):
        parts.append("the %d-board pipeline (%s lanes) %s at %d of %d layers" % (
            d["boards"], "/".join(map(str, d["lanes"])),
            "matches the one-board integer model" if d.get("ok") else "DIFFERS",
            d["layers"], d["of"]))
    elif d:
        parts.append("the decode step %s at %d of %d layers" % (
            "matches the integer model" if d.get("ok") else "DIFFERS", d.get("layers", 0),
            d.get("of", 0)) if "layers" in d else "the design did not compile")
    if "bridge" in st:
        parts.append("through the registers %s" % ("matches" if st["bridge"]["ok"]
                                                   else "DIFFERS"))
    return ", ".join(parts)


def render_report(rep):
    s, st = rep["spec"], rep["stages"]
    L = ["# spec2rtl: %s" % s["name"], "",
         "**%s**: %s." % ("VERIFIED" if rep["ok"] else "FAILED", summary(rep)), "",
         "## Spec", "",
         "| n_layer | d_model | heads | KV heads | head_dim | d_ff | vocab | "
         "style | lanes | board |", "|---|---|---|---|---|---|---|---|---|---|",
         "| %d | %d | %d | %d | %d | %d | %d | %s | %s | %s |" % (
             s["n_layer"], s["d_model"], s["n_head"], s["n_kv_head"], s["head_dim"],
             s["d_ff"], s["vocab"], "q/k norms" if s["qk_norm"] else "q/k/v biases",
             "/".join(map(str, rep["lanes"])) if isinstance(rep["lanes"], list)
             else rep["lanes"], rep["board"]), "",
         "int8 weights per channel, 16-bit activations, %d-position context."
         % SEQ_LEN, ""]
    if st["spec"].get("problems"):
        L += ["Not buildable:", ""] + ["- " + p for p in st["spec"]["problems"]]
        return "\n".join(L) + "\n"
    if rep.get("plan"):
        p = rep["plan"]
        L += ["## Cluster (%s split)" % p["mode"], "",
              "| stage | board | layers | lanes | embedding | head | planner, s/token |",
              "|---|---|---|---|---|---|---|"]
        for i, (x, N) in enumerate(zip(p["stages"], rep["lanes"])):
            L.append("| %d | %s | %d-%d | %d | %s | %s | %.2f |" % (
                i, x["board"], x["layers"][0], x["layers"][1] - 1, N,
                "yes" if x["emb"] else "", "yes" if x["head"] else "", x["seconds"]))
        L += ["", "Links: %s. One stream: %.2f s a token; every stage busy on its own "
              "stream: %.2f tokens/s." % (", ".join("%s to %s over %s" % (
                  l["src"], l["dst"], l["kind"]) for l in p["links"]),
                  p["seconds_per_token"], p["pipelined_tokens_per_s"])]
        if p.get("cannot_hold"):
            L.append("Cannot hold a layer (no DRAM the design reaches): %s."
                     % ", ".join(p["cannot_hold"]))
        L.append("")
    rows = st.get("blocks", {}).get("rows", [])
    if rows:
        L += ["## Blocks (%s agent)" % rep["agent"], "",
              "Each through simulation against its golden model, synthesis, "
              "timing at its clock and 7-series mapping; fixes are what the "
              "agent changed after reading a tool's failure.", "",
              "| file | block | signed off | by | iterations | fixes | checks | timing "
              "| LUTs | DSPs |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| `%s` | %s | %s | %s | %d | %s | %s | %s | %s | %s |" % (
                r["file"] + (" (%d lanes)" % r["lanes"] if "lanes" in r else ""),
                r["block"], "yes" if r["converged"] else "**no**",
                " then ".join("%s (%d%s)" % (a_[0], a_[1], "" if a_[2] else ", no")
                              for a_ in r.get("attempts", [[r.get("agent", "rules"), r["iterations"], True]])),
                r["iterations"], ", ".join(r["fixes"]) or "none",
                r["checks"], ("%.0f MHz (%s, target %s)" % (
                    r["fmax"], r["timing"], r["target"])) if r["fmax"] else "-",
                r["luts"] if r["luts"] is not None else "-",
                r["dsps"] if r["dsps"] is not None else "-"))
        L.append("")
    d = st.get("design")
    if d and d.get("boards") and "error" in d:
        L += ["## The pipeline", "", "Did not run:", "", "```", d["error"], "```", ""]
    elif d and d.get("split_weights"):
        tpp = rep.get("tp_plan")
        if tpp:
            pt = tpp["part"]
            L += ["## Shares", "",
                  "Each board's share of every layer (`cluster.tp_plan`): whole KV heads "
                  "with their query heads, and d_ff and d_model columns in multiples of "
                  "every rank's lanes, sized so the slowest rank finishes each layer "
                  "soonest; the head's chunks in contiguous runs.", "",
                  "| rank | board | lanes | KV heads | d_ff | d_model | head chunks | "
                  "planner, ms a layer |", "|---|---|---|---|---|---|---|---|"]
            for r, n in enumerate(rep["board"].split(" + ")):
                hk = pt["hk"][r]
                L.append("| %d | %s | %d | %d | %d | %d | %s | %.2f |" % (
                    r, n, tpp["lanes"][r], pt["kv"][r], pt["f"][r], pt["d"][r],
                    "%d-%d" % tuple(hk) if hk[1] >= hk[0] else "none",
                    1e3 * tpp["rank_layer_seconds"][r]))
            L += ["", "A layer, gather by gather the slowest rank: %.2f ms, and about %.2f "
                  "ms of gathers: the PL's mover at its measured cost a word, and 200 us "
                  "of Ethernet latency a gather, an estimate; %.3f s a token at a context "
                  "of 128." % (1e3 * tpp["layer_seconds"], 1e3 * tpp["gather_seconds"],
                               tpp["seconds_per_token"]), ""]
        L += ["## The weights split %d ways" % d["boards"], "",
              "Every board on every layer with a slice of every matrix, at its own "
              "width (%s lanes) and on its own clock, gathering the others' slices "
              "(tp.py); simulated at %d of %d layers on %d positions (prompt %s):" % (
                  "/".join(map(str, d["lanes"])), d["layers"], d["of"], d["steps"],
                  d["prompt"]), "",
              "| | tokens | logits |", "|---|---|---|",
              "| the ranks' RTL | %s | %s |" % (d["rtl"], d["logits_rtl"]),
              "| one-board integer model | %s | %s |" % (d["integer_model"],
                                                         d["logits_integer_model"]), "",
              "Network: `%s`." % d["link"], "",
              "| rank | busy core cycles | computing, not waiting on a gather | "
              "cycle model (cluster.tp_rank_cycles) | off by |", "|---|---|---|---|---|"]
        for i, b in enumerate(d["busy_cycles"]):
            c = d.get("compute_cycles", [None] * (i + 1))[i]
            e = d.get("predicted_cycles", [None] * (i + 1))[i]
            L.append("| %d | {:,} | %s | %s | %s |".format(b) % (
                i, "{:,}".format(c) if c is not None else "-",
                "{:,}".format(e) if e is not None else "-",
                "%s%%" % d["prediction_error_pct"][i] if e is not None else "-"))
        L.append("")
    elif d and d.get("boards"):
        L += ["## The pipeline", "",
              "%d boards, each on its own clock and at its own width (%s lanes), "
              "sharing only CRC-checked messages, simulated at %d of %d layers "
              "(split %s) on %d positions (prompt %s):" % (
                  d["boards"], "/".join(map(str, d["lanes"])), d["layers"], d["of"],
                  d["split"], d["steps"], d["prompt"]), "",
              "| | tokens | logits |", "|---|---|---|",
              "| pipeline RTL | %s | %s |" % (d["rtl"], d["logits_rtl"]),
              "| one-board integer model | %s | %s |" % (d["integer_model"],
                                                         d["logits_integer_model"]), "",
              "Links: `%s`." % d["link"], "",
              "| stage | busy core cycles, simulated | planner's estimate | off by |",
              "|---|---|---|---|"]
        for i, (b, c, e) in enumerate(zip(d["busy_cycles"], d["predicted_cycles"],
                                          d["prediction_error_pct"])):
            L.append("| %d | {:,} | {:,} | %s%% |".format(b, c) % (i, e))
        if d.get("text"):
            L.append("\nText chosen: %r." % d["text"])
        L.append("")
    elif d:
        L += ["## Design", ""]
        if "error" in d:
            L += ["Did not compile:", "", "```", d["error"], "```"]
        else:
            L += ["The decode step, generated around the signed-off files above, "
                  "simulated at %d of %d layers on %d positions (prompt %s):"
                  % (d["layers"], d["of"], d["steps"], d["prompt"]), "",
                  "| | tokens | logits |", "|---|---|---|",
                  "| RTL | %s | %s |" % (d["rtl"], d["logits_rtl"]),
                  "| integer model | %s | %s |" % (d["integer_model"],
                                                   d["logits_integer_model"]), "",
                  "Core cycles a position: %s; the planner's cycle model "
                  "(cluster.py) says %s, %s%% off." % (
                      ", ".join("{:,}".format(c) for c in d["cycles"]),
                      ", ".join("{:,}".format(c) for c in d.get("predicted_cycles", [])),
                      "/".join(str(e) for e in d.get("prediction_error_pct", []))),
                  "The design compiled every signed-off file above; %s."
                  % ("none differed from the generator's own copy" if not
                     d["generator_files_replaced"] else "these differed from the "
                     "generator's own copies, and the signed-off ones were used: "
                     + ", ".join(d["generator_files_replaced"]))]
            if d.get("text"):
                L.append("Text chosen: %r." % d["text"])
            if d["layers"] < d["of"]:
                L.append("The full %d-layer design is in `design_full/`, from the "
                         "same generator and the same blocks; `--sim-layers all` "
                         "simulates it." % d["of"])
        L.append("")
    b = st.get("bridge")
    if b and d and d.get("split_weights"):
        L += ["## Through every rank's registers", "",
              "Each rank as its package runs it: the register block and DDR bridge "
              "around the core, a DDR model of its own that stalls at random, its own "
              "clock, and every gather done by the ARM's side through GADDR, GDATA and "
              "GATHER (tp.build_boards). Tokens %s, logits %s: %s. Network: `%s`." % (
                  b["tokens"], b["logits"], "match" if b["ok"] else "DIFFER", b["link"]), "",
              "| rank | board | core cycles | bus cycles | bus cycles a core cycle |",
              "|---|---|---|---|---|"]
        for i, (n, c, u) in enumerate(zip(rep["board"].split(" + "), b["core_cycles"],
                                          b["bus_cycles"])):
            L.append("| %d | %s | {:,} | {:,} | %.2f |".format(c, u) % (i, n, u / max(1, c)))
        L.append("")
    elif b:
        L += ["## Through the board's registers", "",
              "Tokens %s against a DDR model that stalls at random: %s; %.2f bus "
              "cycles a core cycle." % (b["tokens"], "match" if b["ok"] else "DIFFER",
                                        sum(b["bus_cycles"]) / max(1, sum(b["core_cycles"]))),
              ""]
    if rep.get("packages"):
        L += ["## Packages", ""] + ["- `%s/`" % os.path.basename(o)
                                    for o in rep["packages"]] + [""]
    L += ["%d s in all." % rep["seconds"]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", nargs="?", help="a model shape, model_spec.json's keys")
    ap.add_argument("--weights", help="a checkpoint directory instead (config.json, "
                    "model.safetensors, tokenizer.json)")
    ap.add_argument("--board", default="zybo_z7_20", choices=sorted(boards.PACKAGES))
    ap.add_argument("--boards", nargs="+", choices=sorted(boards.PACKAGES),
                    help="a heterogeneous cluster, in chain order: the planner splits "
                    "the layers over them and the whole pipeline is verified")
    ap.add_argument("--mode", default="balanced", choices=("balanced", "fast", "even"),
                    help="the layer split: balanced (by speed) or fast (fewest, fastest "
                    "boards); the weight split: balanced (shares by speed) or even")
    ap.add_argument("--split", default="layers", choices=("layers", "weights"),
                    help="with --boards: a pipeline of layer ranges (layers), or every "
                    "board on every layer with a slice of every matrix (weights)")
    ap.add_argument("--lanes", type=int, help="the core's width (default: the board's)")
    ap.add_argument("--sim-layers", default=None,
                    help="layers to simulate: a number, or all (default: all for a "
                    "small model, 1 for one at a real model's size)")
    ap.add_argument("--gen", type=int, default=2, help="tokens to generate in simulation")
    ap.add_argument("--sim", default="fast", choices=("fast", "iverilog", "verilator"),
                    help="the simulator: fast is Verilator where installed (every layer of "
                    "a real model in minutes), else Icarus")
    ap.add_argument("--agent", default="rules", choices=("rules", "llm", "swarm"))
    ap.add_argument("--dv", action="store_true", help="mutation-test every testbench")
    ap.add_argument("--package", action="store_true", help="write the board package")
    ap.add_argument("--bridge", action="store_true",
                    help="simulate through the board's registers and DDR bridge")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.spec or a.weights):
        ap.error("give a spec, or --weights")
    spec = json.load(open(a.spec)) if a.spec else None
    rep = run(spec, a.board, a.out, a.weights, a.lanes, a.sim_layers, a.gen,
              a.agent, a.package, a.bridge, a.dv, a.prompt, cluster=a.boards,
              mode=a.mode, split=a.split, sim=a.sim)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
