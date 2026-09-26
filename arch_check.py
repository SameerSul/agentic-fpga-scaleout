"""Does the derivation hold for real LLM architectures, or only for the
one it was written against?

The sweep drives six model variants, but they are variants of one spec.
This takes published configurations, at their real sizes, and asks
whether every block derives and whether the RTL it produces actually
simulates. A derivation that only works for GPT-2 is a demo.

  python3 arch_check.py [--sim] [--only llama3_1_8b]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import specgen
from agent import (RuleBasedAgent, FIX_WIDTH, FIX_CLEAR, FIX_SATURATE,
                   FIX_LUT, FIX_NORM, FIX_EVEN)

WORK = os.path.join(specgen.ROOT, "build_arch")

# name, layers, d_model, d_ff, heads, kv heads, vocab
ARCHS = [
    ("smollm2_135m",   30,   576,  1536,  9,  3,  49152),
    ("gpt2_124m",      12,   768,  3072, 12, 12,  50257),
    ("qwen3_0p6b",     28,  1024,  3072, 16,  8, 151936),
    ("gemma2_2b",      26,  2304,  9216,  8,  4, 256000),
    ("llama3_2_1b",    16,  2048,  8192, 32,  8, 128256),
    ("phi3_mini_3p8b", 32,  3072,  8192, 32, 32,  32064),
    ("qwen3_4b",       36,  2560,  9728, 32,  8, 151936),
    ("mistral_7b",     32,  4096, 14336, 32,  8,  32000),
    ("llama3_1_8b",    32,  4096, 14336, 32,  8, 128256),
    ("gpt2_xl",        48,  1600,  6400, 25, 25,  50257),
]

# Quantizations worth checking: the datapath is derived from these.
QUANTS = [("int4", 4, 8), ("int8", 8, 8), ("w8a16", 8, 16),
          ("fp16-ish", 16, 16)]

BLOCKS = [
    ("chiplet", specgen.derive_chiplet_spec, specgen.render_testbench,
     "render_mac", {FIX_WIDTH, FIX_CLEAR}, ()),
    ("requant", specgen.derive_requant_spec,
     specgen.render_requant_testbench, "render_requant",
     {FIX_SATURATE}, ()),
    ("exp", specgen.derive_exp_spec, specgen.render_exp_testbench,
     "render_exp", {FIX_LUT}, ()),
    ("recip", specgen.derive_recip_spec, specgen.render_recip_testbench,
     "render_recip", {FIX_NORM}, ()),
    ("rsqrt", specgen.derive_rsqrt_spec, specgen.render_rsqrt_testbench,
     "render_rsqrt", {FIX_EVEN}, ()),
]


def model_spec(arch, wb, ab):
    n, L, d, f, h, kv, v = arch
    return {"name": "%s_w%da%d" % (n, wb, ab), "n_layer": L, "d_model": d,
            "d_ff": f, "n_head": h, "n_kv_head": kv, "vocab": v,
            "weight_bits": wb, "activation_bits": ab, "seq_len": 1024,
            "dtype_bytes": 2, "target_tokens_per_s": 10,
            "allreduces_per_token_per_layer": 2, "batch_size": 1}


def simulate(name, spec, tb_src, rtl_src):
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(os.path.join(d, "tb.v"), "w").write(tb_src)
    open(os.path.join(d, "dut.v"), "w").write(rtl_src)
    r = subprocess.run(["iverilog", "-g2005", "-o", "s.out", "tb.v",
                        "dut.v"], cwd=d, capture_output=True, text=True)
    if r.returncode:
        err = [l for l in (r.stdout + r.stderr).splitlines()
               if "error" in l.lower()]
        return "compile", (err[0][:60] if err else "")
    try:
        r = subprocess.run(["vvp", "s.out"], cwd=d, capture_output=True,
                           text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    if "TB_RESULT: PASS" in r.stdout:
        return "pass", ""
    bad = [l for l in r.stdout.splitlines() if "TB_FAIL" in l]
    return "fail", (bad[0][:60] if bad else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true",
                    help="also simulate, not just derive")
    ap.add_argument("--only", default=None)
    ap.add_argument("--quant", default=None)
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    agent = RuleBasedAgent()
    rows, bad = [], []
    print("%-16s %-9s %-5s %-5s %-6s %s"
          % ("architecture", "quant", "dw", "acc", "rsqrt", "blocks"))
    print("-" * 78)
    for arch in ARCHS:
        if a.only and arch[0] != a.only:
            continue
        for qn, wb, ab in QUANTS:
            if a.quant and qn != a.quant:
                continue
            ms = model_spec(arch, wb, ab)
            status, detail = [], ""
            try:
                specs = {}
                for bn, derive, _tb, _r, _f, _e in BLOCKS:
                    specs[bn] = derive(ms)
                c = specs["chiplet"]["parameters"]
                rs = specs["rsqrt"]["parameters"]
            except Exception as e:
                print("%-16s %-9s %s" % (arch[0], qn, "DERIVE FAILED: "
                                         + str(e)[:44]))
                bad.append((arch[0], qn, "derive", str(e)[:60]))
                continue
            for bn, derive, tb_fn, rname, fixes, _e in BLOCKS:
                if not a.sim:
                    status.append(bn + ":-")
                    continue
                sp = specs[bn]
                v, why = simulate("%s_%s_%s" % (arch[0], qn, bn), sp,
                                  tb_fn(sp), getattr(agent, rname)(sp, fixes))
                status.append("%s:%s" % (bn, v[0] if v == "pass" else v))
                if v != "pass":
                    bad.append((arch[0], qn, bn, why))
            print("%-16s %-9s %-5d %-5d %-6d %s"
                  % (arch[0], qn, c["data_width"], c["acc_width"],
                     rs["in_width"], " ".join(status)))
            sys.stdout.flush()
            rows.append({"arch": arch[0], "quant": qn,
                         "data_width": c["data_width"],
                         "acc_width": c["acc_width"],
                         "rsqrt_in": rs["in_width"],
                         "ok": all(s.endswith(":p") or s.endswith(":-")
                                   for s in status)})
    print("\n%d configurations, %d problems" % (len(rows), len(bad)))
    for a_, q, b_, why in bad:
        print("  %-16s %-9s %-8s %s" % (a_, q, b_, why))
    json.dump(rows, open(os.path.join(specgen.ROOT, "arch_check.json"),
                         "w"), indent=2)
    shutil.rmtree(WORK, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
