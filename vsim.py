"""Compile and run a testbench in Icarus or in Verilator.

Icarus is the reference: four-state, so a value never written shows as
x, and every test in tests.py runs on it. Verilator compiles the same
testbenches, delays, forks and all (--timing), into a program some
seventy times faster on the full models: Qwen3-0.6B's 28 layers and the
head, five positions, take two minutes where Icarus takes two and a half
hours. spec2rtl.py uses it when it is installed, and then simulates every
layer of a real model rather than one. It is two-state, so a word never
written reads as zero instead of x; the tokens and logits have to match
the integer model either way, and test_simulators_agree holds the two
to the same tokens, logits and cycles.
"""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile


def spaceless(path, tag=""):
    """path itself, or where to build for it when it has a space in it:
    Verilator's make refuses such a directory. A fixed place under the
    system's temporary directory, named by path, so a rebuild replaces
    the last one instead of adding to it."""
    if " " not in path:
        return path
    h = hashlib.sha1((path + "|" + tag).encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), "fpgai_" + h)


def pick(sim=None):
    """iverilog, verilator, or "fast": Verilator when it is installed."""
    sim = sim or os.environ.get("FPGAI_SIM") or "iverilog"
    if sim == "fast":
        sim = "verilator" if shutil.which("verilator") else "iverilog"
    return sim


def build(work, srcs, sim="iverilog", defines=("SIM",), tag="t"):
    """Compile srcs (the testbench first) in work. Returns the command
    that runs the simulation, or raises RuntimeError with the log."""
    if sim == "verilator":
        top = re.search(r"^\s*module\s+(\w+)", open(os.path.join(work, srcs[0])).read(),
                        re.M).group(1)
        mdir = spaceless(os.path.join(os.path.abspath(work), "vobj_" + tag))
        shutil.rmtree(mdir, ignore_errors=True)
        r = subprocess.run(["verilator", "--binary", "--timing", "-j", "8", "-O2",
                            "-Wno-fatal", "-Wno-lint", "-Wno-style", "--x-assign", "0",
                            "--x-initial", "0", "--top-module", top, "--Mdir", mdir,
                            "-o", "sim"] + ["-D" + d for d in defines] + list(srcs),
                           cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError((r.stdout + r.stderr)[-3000:])
        return [os.path.join(work, mdir, "sim")]
    r = subprocess.run(["iverilog", "-g2005"] + ["-D" + d for d in defines]
                       + ["-o", tag + ".out"] + list(srcs), cwd=work, capture_output=True,
                       text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-3000:])
    return ["vvp", tag + ".out"]


def run(work, srcs, sim="iverilog", timeout=None, defines=("SIM",), tag="t"):
    """Compile and run; the simulation's output, or the compiler's log."""
    try:
        cmd = build(work, srcs, sim, defines, tag)
    except RuntimeError as e:
        return str(e)
    return subprocess.run(cmd, cwd=work, capture_output=True, text=True,
                          timeout=timeout).stdout


def stream(work, srcs, sim="iverilog", defines=("SIM",), tag="t"):
    """Compile and start; a Popen whose stdout is the simulation's."""
    cmd = build(work, srcs, sim, defines, tag)
    return subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, text=True)
