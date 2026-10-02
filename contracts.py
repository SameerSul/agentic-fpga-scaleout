"""Formal handshake contracts: what every block's spec promises on its
ports, proved for every input and every reachable state instead of the
runs a testbench drives.

The testbenches compare values against the golden model on the runs they
drive. On the fifth end-to-end run Sonnet's RMSNorm never gave index 0
and gave index 63 twice, so its count of outputs was right and every
value it did give was too: it passed all 268 checks, and the decode step
built from it came out wrong. This states the protocol as properties and
proves them with ABC: PDR (IC3) for every reachable state, BMC where a
property needs a bound.

Two families, told apart by their ports:

  sequenced   start / busy / <x>_valid / <x>_index: RMSNorm, softmax, the
              attention head, the projection, the matvec sequencer. After
              reset busy and the valid are low; busy is high the cycle
              after a start; an output only inside a run, while busy is
              high or on the cycle it falls; every index below the row's
              length; no index twice in a run; when busy falls every
              index given exactly once; the projection's in order. All of
              that is proved by PDR for every size the spec allows. The
              environment starts a run only while busy is low, holds the
              sizes and the constants for the whole trace, and gives every
              memory read arbitrary data on every cycle. That every run
              ends is checked by BMC, at small sizes, over two whole runs
              back to back, each run's length first measured in
              simulation: a run must end within four times that.

  streaming   valid_in / valid_out: the MACs, the requantizer, exp,
              reciprocal, inverse square root, RoPE, SiLU, the residual
              add. valid_out is 0 after reset, and each valid_in gives
              exactly one valid_out, in order, at one fixed latency: the
              spec's, or for a block whose latency is its own choice, the
              one it shows on a single pulse; after a MAC's clear,
              valid_out is 0. For the blocks that are a function of one
              sample, a second copy fed the same sample on any cycle, and
              arbitrary other samples around it, must give the same
              output: no state leaks from one sample into the next.

Everything is stated on the ports the spec defines, so the same contract
applies to a design an LLM wrote and to the reference one. A failure
comes back with the property, what it means, the cycle, and the handshake
signals cycle by cycle, replayed from the solver's counterexample.

  python3 contracts.py GATES_DIR        every block whose report is there
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys

import formal

ROOT = os.path.dirname(os.path.abspath(__file__))

# Per sequenced top: how many indices a run gives; the sizes the proof
# covers; the small ones, for the bounded check that runs end; whether the
# spec fixes the order. The proof's sizes stop short of the spec's 256:
# PDR proves matvec's contract for every size up to 16 in 4 s and had not
# decided it up to 256 after a minute, nor had interpolation or dprove.
# Up to 16 positions or rows, and three projection groups, the last one
# partial, is every case a counter, a group boundary or a partial group
# can get wrong.
SEQ = {
    "rmsnorm": {"count": ("param", "d_model"), "sizes": {}, "small": {}},
    "softmax": {"count": ("port", "n"), "sizes": {"n": (1, 16)}, "small": {"n": (1, 3)}},
    "attnn": {"count": ("param", "head_dim"), "sizes": {"n": (1, 16)},
              "small": {"n": (1, 2)}},
    "projn": {"count": ("port", "cols"), "sizes": {"cols": (1, 40), "depth": (1, 4)},
              "small": {"cols": (1, 18), "depth": (1, 2)}, "order": True},
    "matvec": {"count": ("port", "cols"), "sizes": {"cols": (1, 16), "depth": (1, 16)},
               "small": {"cols": (1, 3), "depth": (1, 3)}},
}

# Streaming blocks that are a function of one sample; the MACs keep a sum.
STATELESS = ("requant", "expu", "recip", "rsqrt", "rope", "silu", "resadd")
# Inputs a stream holds stable (the residual add's scales); every other
# input of a streaming block may change with each sample.
STREAM_CONFIG = {"resadd": ("scale_a", "scale_b", "shift")}

PDR_TIMEOUT = 300
# The two-copy check asks the solver to show two copies of the datapath
# agree: quick for a table or a shifter, out of reach for some wide
# multipliers. Out of time there is reported, not counted as a failure.
SAME_TIMEOUT = 180
MAX_FRAMES = 3000         # BMC's cycles; bmc3 does 300 of matvec in 4 s

PROPS = {
    "p_reset": "after reset, busy and the valid are 0",
    "p_busy": "busy reads 1 the cycle after a start",
    "p_window": "an output only inside a run, while busy is high or on the cycle it falls",
    "p_range": "every output index is below the row's length",
    "p_once": "no index is given twice in a run",
    "p_order": "the outputs come in index order, 0 first",
    "p_all": "when busy falls, every index has been given exactly once",
    "p_ends": "every run ends",
    "p_clear": "valid_out is 0 the cycle after a clear",
    "p_latency": "each valid_in gives exactly one valid_out, in order, at the fixed latency",
    "p_same": "the output depends only on its own sample: a second copy given the same "
              "sample, and other samples around it, gives the same output",
}


def available():
    return all(shutil.which(t) for t in ("yosys", "yosys-abc", "iverilog", "vvp"))


def _ports(spec):
    return [p for p in spec["ports"] if p["name"] not in ("clk", "rst_n")]


def family(spec):
    names = {p["name"] for p in spec["ports"]}
    if {"start", "busy"} <= names:
        return "sequenced"
    if {"valid_in", "valid_out"} <= names:
        return "streaming"
    return None


def _valid_index(spec):
    outs = {p["name"]: p for p in spec["ports"] if p["dir"] == "output"}
    for v in outs:
        if v.endswith("_valid") and v[:-6] + "_index" in outs:
            return v, v[:-6] + "_index", outs[v[:-6] + "_index"]["width"]
    raise ValueError("no <x>_valid / <x>_index pair")


def _top(spec):
    return spec["top_module"]


def _count_expr(spec, info):
    kind, name = info["count"]
    if kind == "param":
        return str(spec["parameters"][name])
    return name


# -- measuring a run or a latency in simulation ---------------------------

SEQ_TB = """`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
{regs}
{wires}
  {top} dut (.clk(clk), .rst_n(rst_n){conns});
  integer cyc = 0, t0;
  always @(posedge clk) begin
    cyc <= cyc + 1;
{rand}
  end
  initial begin
{init}
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);
    start = 1; t0 = cyc;
    @(negedge clk); start = 0;
    while (busy && cyc < t0 + {limit}) @(negedge clk);
    if (busy) $display("RUN_HUNG");
    else $display("RUN_CYCLES %0d", cyc - t0);
    $finish;
  end
endmodule
"""

STREAM_TB = """`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
{regs}
{wires}
  {top} dut (.clk(clk), .rst_n(rst_n){conns});
  integer cyc = 0, t0 = -1;
  always @(posedge clk) begin
    cyc <= cyc + 1;
{rand}
    if (rst_n && valid_out && t0 >= 0) begin
      $display("LATENCY %0d", cyc - t0);
      $finish;
    end
  end
  initial begin
{init}
    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);
    valid_in = 1; t0 = cyc;
    @(negedge clk); valid_in = 0;
    repeat (200) @(negedge clk);
    $display("LATENCY_NONE");
    $finish;
  end
endmodule
"""


def _tb(spec, fam, info=None):
    regs, wires, conns, rand, init = [], [], [], [], []
    sizes = (info or {}).get("sizes", {})
    for p in _ports(spec):
        w = p.get("width", 1)
        rng = "[%d:0] " % (w - 1) if w > 1 else ""
        n = p["name"]
        conns.append(".%s(%s)" % (n, n))
        if p["dir"] == "output":
            wires.append("  wire %s%s;" % (rng, n))
            continue
        regs.append("  reg %s%s = 0;" % (rng, n))
        if n in ("start", "valid_in", "load_valid", "clear"):
            continue
        if n in sizes:
            init.append("    %s = %d;" % (n, sizes[n][1]))
        elif n.endswith("_data") or fam == "streaming":
            rand.append("    %s <= {$random, $random, $random, $random, $random, $random, "
                        "$random, $random};" % n)
        else:
            init.append("    %s = %d;" % (n, random.Random(n).randrange(1 << min(w, 30))))
    return dict(regs="\n".join(regs), wires="\n".join(wires), top=_top(spec),
                conns="".join(", " + c for c in conns), rand="\n".join(rand),
                init="\n".join(init) or "    ;")


def _sources(work, rtl_path, deps):
    """The design and its supplied blocks under bare names in work: yosys
    scripts split on spaces, and this repo's path has one."""
    names = []
    for i, src in enumerate([rtl_path] + list(deps)):
        dst = "src%d_%s" % (i, os.path.basename(src))
        shutil.copyfile(src, os.path.join(work, dst))
        names.append(dst)
    return names


def _simulate(work, srcs, tb_text, timeout):
    with open(os.path.join(work, "measure_tb.v"), "w") as f:
        f.write(tb_text)
    rc, out = formal._run(["iverilog", "-g2012", "-o", "measure.vvp", "measure_tb.v"] + srcs,
                          work, timeout)
    if rc != 0:
        return None, "compile: " + " ".join(out.strip().splitlines()[-3:])
    rc, out = formal._run(["vvp", "-n", "measure.vvp"], work, timeout)
    return out, None


# -- the contracts, as AIGER-ready wrappers -------------------------------
#
# Free signals are inputs of the wrapper: an AIGER input is free on every
# cycle. A value held for the whole trace is latched from an input on
# cycle 0. t counts the first cycles: reset is cycles 0 and 1.

class _Wrap:
    def __init__(self):
        self.ins, self.body = [], []

    @staticmethod
    def _typ(p):
        w = p.get("width", 1)
        return ("signed " if p.get("signed") else "") + ("[%d:0] " % (w - 1) if w > 1 else "")

    def free(self, p):
        self.ins.append("input %s%s" % (self._typ(p), p["name"]))

    def held(self, p):
        ty, n = self._typ(p), p["name"]
        self.ins.append("input %s%s_in" % (ty, n))
        self.body.append("  reg %s%s_r = 0;\n  wire %s%s = t == 0 ? %s_in : %s_r;\n"
                         "  always @(posedge clk) if (t == 0) %s_r <= %s_in;"
                         % (ty, n, ty, n, n, n, n, n))

    def out(self, p, suffix=""):
        self.body.append("  wire %s%s%s;" % (self._typ(p), p["name"], suffix))

    def header(self, top, kind):
        return ("// GENERATED by contracts.py: %s's %s contract.\n"
                "module contract_top (input clk%s);\n"
                "  reg [1:0] t = 0;\n"
                "  always @(posedge clk) if (t != 2'd3) t <= t + 1;\n"
                "  wire rst_n = t >= 2;\n%s\n"
                % (top, kind, "".join(", " + i for i in self.ins), "\n".join(self.body)))


SEQ_PROPS = """
  // The environment: a start only while idle, a one-cycle pulse; loads
  // only while idle; sizes within range.
  reg busy_q = 0, rst_q = 0, start_q = 0, run = 0;
  always @(*) begin
    assume(!start || (rst_n && !busy && !start_q));
{env}
  end

  // k is one arbitrary index, held for the trace; cnt counts its outputs
  // in the current run, nxt the run's outputs, rt the run's cycles.
  reg [1:0] cnt = 0;
{regs}
  wire accept = rst_n && start;
  wire give = rst_n && {valid};
  wire give_k = give && {index} == k;
  wire fin = run && busy_q && !busy;
  always @(posedge clk) begin
    busy_q <= busy; rst_q <= rst_n; start_q <= accept;
    if (accept) begin
      run <= 1; cnt <= 0;{zero}
    end else begin
      if (give_k && cnt != 2'd3) cnt <= cnt + 1;{count_up}
      if (fin) run <= 0;
    end
  end

  always @(*) if (t >= 1) begin
    if (!rst_q) begin
      p_reset: assert(!busy && !{valid});
    end
    if (start_q) begin
      p_busy: assert(busy);
    end
    if (give) begin
      p_window: assert(run && (busy || busy_q));
      p_range: assert({{1'b0, {index}}} < {count});
    end
    if (give_k && {{1'b0, k}} < {count}) begin
      p_once: assert(cnt == 0);
    end
{order}
    if (fin && {{1'b0, k}} < {count}) begin
      p_all: assert(cnt + give_k == 1);
    end
{ends}
  end
endmodule
"""


def _seq_wrapper(spec, info, sizes, hang=None):
    valid, index, iw = _valid_index(spec)
    w, conns, env = _Wrap(), [], []
    for p in _ports(spec):
        n = p["name"]
        conns.append(".%s(%s)" % (n, n))
        if p["dir"] == "output":
            w.out(p)
        elif n in ("start", "load_valid") or n.endswith("_data"):
            w.free(p)
        else:
            w.held(p)
        if n in sizes:
            lo, hi = sizes[n]
            env.append("    assume(%s >= %d && %s <= %d);" % (n, lo, n, hi))
        if n == "load_valid":
            env.append("    assume(!load_valid || (!busy && !start));")
    w.held({"name": "k", "width": iw})
    text = w.header(_top(spec), "handshake")
    text += "  %s dut (.clk(clk), .rst_n(rst_n)%s);\n" % (
        _top(spec), "".join(", " + c for c in conns))
    order = ("    if (give) begin\n      p_order: assert(%s == nxt);\n    end" % index
             if info.get("order") else "")
    ends = ("    if (run) begin\n      p_ends: assert(rt < %d);\n    end" % hang
            if hang else "")
    # nxt (the run's outputs so far) and rt (its cycles) only when a
    # property reads them: an unread counter still makes PDR go deep.
    regs, zero, up = [], "", ""
    if info.get("order"):
        regs.append("  reg [%d:0] nxt = 0;" % (iw - 1))
        zero += " nxt <= 0;"
        up += "\n      if (give) nxt <= nxt + 1;"
    if hang:
        regs.append("  reg [15:0] rt = 0;")
        zero += " rt <= 0;"
        up += "\n      if (run && rt != 16'hffff) rt <= rt + 1;"
    return text + SEQ_PROPS.format(env="\n".join(env), valid=valid, regs="\n".join(regs),
                                   zero=zero, count_up=up, index=index,
                                   count=_count_expr(spec, info), order=order, ends=ends)


STREAM_PROPS = """
  // h[i]: a sample went in i+1 cycles ago; c[i]: a clear did.
  reg [{hm}:0] h = 0, c = 0;
  always @(posedge clk) begin
    h <= {{h[{hm1}:0], rst_n && valid_in}};
    c <= {{c[{hm1}:0], {clear}}};
  end
{twin}
  always @(*) if (t >= 1) begin
    if (t <= 2) begin
      p_reset: assert(!valid_out);
    end
    if (t == 3 && c[{lat}:0] == 0) begin
      p_latency: assert(valid_out == h[{latm}]);
    end
{clr}{same}
  end
endmodule
"""


def _stream_wrapper(spec, lat, twin):
    w, conns, tconns, outs, shared = _Wrap(), [], [], [], []
    clear = "1'b0"
    for p in _ports(spec):
        n = p["name"]
        conns.append(".%s(%s)" % (n, n))
        if p["dir"] == "output":
            w.out(p)
            outs.append(n)
            if twin:
                w.out(p, "_b")
                tconns.append(".%s(%s_b)" % (n, n))
            continue
        if n == "clear":
            clear = "rst_n && clear"
        if n in STREAM_CONFIG.get(_top(spec), ()):
            w.held(p)
            tconns.append(".%s(%s)" % (n, n))
            continue
        w.free(p)
        if twin:
            w.free(dict(p, name=n + "_b"))
            tconns.append(".%s(%s_b)" % (n, n))
            shared.append(n)
    if twin:
        w.free({"name": "share", "width": 1})
    text = w.header(_top(spec), "streaming")
    text += "  %s dut (.clk(clk), .rst_n(rst_n)%s);\n" % (
        _top(spec), "".join(", " + c for c in conns))
    hm = max(lat + 1, 2)
    twin_text = same = ""
    if twin:
        data = [o for o in outs if o != "valid_out"]
        twin_text = (
            "  %s twin (.clk(clk), .rst_n(rst_n)%s);\n"
            "  // On a cycle with share high the two copies take the same sample;\n"
            "  // sh[i]: that was i+1 cycles ago, and the sample was valid.\n"
            "  reg [%d:0] sh = 0;\n"
            "  always @(posedge clk) sh <= {sh[%d:0], rst_n && share && valid_in};\n"
            "  always @(*) assume(!share || (%s));\n"
            % (_top(spec), "".join(", " + c for c in tconns), hm, hm - 1,
               " && ".join("%s_b == %s" % (n, n) for n in shared)))
        same = ("    if (t == 3 && sh[%d]) begin\n      p_same: assert(%s);\n    end\n"
                % (lat - 1, " && ".join("%s_b == %s" % (o, o) for o in data)))
    clr = ("    if (c[0]) begin\n      p_clear: assert(!valid_out);\n    end\n"
           if clear != "1'b0" else "")
    return text + STREAM_PROPS.format(hm=hm, hm1=hm - 1, clear=clear, twin=twin_text,
                                      lat=lat, latm=lat - 1, clr=clr, same=same)


def _spec_latency(spec):
    for b in spec.get("behavior", []):
        m = re.search(r"[Ll]atency from valid_in to valid_out is (?:exactly )?(\d+) cycles", b)
        if m:
            return int(m.group(1))
    return None


# -- the engine -----------------------------------------------------------

YOSYS_AIG = ("read_verilog -sv -formal contract.v; {reads}; prep -top contract_top; "
             "flatten; async2sync; chformal -assume -early; opt -full; techmap; opt -fast; "
             "memory_map; opt -full; simplemap; dffunmap; abc -g AND; opt_clean; "
             "write_rtlil contract_aig.il; "
             "write_aiger -zinit -map contract.aim -ywmap contract.ywa contract.aig")


def _abc(work, srcs, wrapper, engine, timeout, frames=0):
    """Prove wrapper's assertions with ABC: engine "pdr" for every
    reachable state, "bmc" over frames cycles from reset. Returns a dict
    with status "proved", "bounded", "failed", "timeout" or "error"."""
    with open(os.path.join(work, "contract.v"), "w") as f:
        f.write(wrapper)
    for f in ("cex.aiw", "cex.vcd"):
        if os.path.exists(os.path.join(work, f)):
            os.remove(os.path.join(work, f))
    rc, out = formal._run(["yosys", "-q", "-p", YOSYS_AIG.format(
        reads="; ".join("read_verilog -sv %s" % s for s in srcs))], work, timeout)
    if rc != 0:
        return {"status": "error",
                "detail": " ".join(l for l in out.strip().splitlines()
                                   if not l.startswith("Warning"))[-400:]}
    cmd = ("pdr -T %d" % timeout) if engine == "pdr" else ("bmc3 -F %d -T %d" % (frames, timeout))
    rc, out = formal._run(["yosys-abc", "-c", "read_aiger contract.aig; fold; strash; %s; "
                           "write_cex -a cex.aiw" % cmd], work, timeout + 60)
    if rc == 124 or re.search(r"[Tt]imeout|[Rr]esource limit|time limit", out) and \
            not re.search(r"asserted in frame", out):
        if engine == "bmc":
            m = re.search(r"No output asserted in (\d+) frames", out)
            if m and int(m.group(1)) >= frames:
                return {"status": "bounded", "frames": frames}
        return {"status": "timeout", "engine": engine}
    if "Property proved" in out:
        return {"status": "proved"}
    m = re.search(r"Output (\d+) .*?asserted in frame (\d+)", out)
    if m:
        return _failure(work, int(m.group(1)), int(m.group(2)), engine)
    m = re.search(r"No output asserted in (\d+) frames", out)
    if m:
        return {"status": "bounded", "frames": int(m.group(1))}
    return {"status": "error", "detail": out.strip()[-400:]}


def _failure(work, po, frame, engine):
    prop = None
    try:
        ywa = json.load(open(os.path.join(work, "contract.ywa")))
        asserts = ywa.get("asserts", [])
        if po < len(asserts):
            prop = next((p for p in re.findall(r"p_\w+", json.dumps(asserts[po]))), None)
    except (OSError, ValueError):
        pass
    vcd = os.path.join(work, "cex.vcd")
    formal._run(["yosys", "-q", "-p", "read_rtlil contract_aig.il; sim -clock clk -r cex.aiw "
                 "-map contract.aim -vcd cex.vcd -n %d" % (frame + 2)], work, 300)
    trace = _trace(vcd)
    return {"status": "failed", "engine": engine, "property": prop,
            "means": PROPS.get(prop, "an assertion of the contract"), "cycle": frame,
            "trace": trace, "story": _story(trace)}


def _spans(pairs):
    """[(cycle, index)] as "index a to b on cycles c to d" runs."""
    out, i = [], 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[j][0] + 1 \
                and pairs[j + 1][1] == pairs[j][1] + 1:
            j += 1
        (c0, x0), (c1, x1) = pairs[i], pairs[j]
        out.append("index %s on cycle %d" % (x0, c0) if i == j else
                   "index %s to %s on cycles %d to %d" % (x0, x1, c0, c1))
        i = j + 1
    return out


def _story(trace):
    """A counterexample in a sentence or two: when runs started and ended,
    which indices came out when, and which the property was about."""
    rows = []
    for line in trace:
        m = re.match(r"cycle (\d+): (.*)", line)
        if m:
            rows.append((int(m.group(1)), dict(kv.split("=", 1) for kv in m.group(2).split())))
    if not rows:
        return ""
    valid = next((v for v in ("o_valid", "w_valid", "col_valid") if v in rows[0][1]), None)
    if not valid:
        return ""
    index = valid[:-6] + "_index"
    starts = [c for c, v in rows if v.get("start") == "1" and v.get("rst_n") == "1"]
    falls = [c for (c, v), (_, u) in zip(rows[1:], rows) if v.get("busy") == "0"
             and u.get("busy") == "1" and u.get("rst_n") == "1"]
    outs = [(c, int(v[index])) for c, v in rows if v.get(valid) == "1" and v.get("rst_n") == "1"
            and v.get(index, "x").isdigit()]
    parts = []
    if starts:
        parts.append("start on cycle%s %s" % ("s" if len(starts) > 1 else "",
                                              ", ".join(map(str, starts))))
    if falls:
        parts.append("busy fell on cycle%s %s" % ("s" if len(falls) > 1 else "",
                                                  ", ".join(map(str, falls))))
    if outs:
        parts.append("outputs: " + "; ".join(_spans(outs)))
    seen = [x for _, x in outs]
    twice = sorted({x for x in seen if seen.count(x) > 1})
    if twice:
        parts.append("given more than once: index %s" % ", ".join(map(str, twice)))
    k = rows[-1][1].get("k")
    if k and k.isdigit() and int(k) not in seen:
        parts.append("index %s never given" % k)
    return ". ".join(parts) + "."


def _trace(vcd):
    """The handshake signals of a counterexample, cycle by cycle, as text:
    what an engineer would read off the waveform."""
    if not os.path.exists(vcd):
        return []
    want = ("start", "busy", "o_valid", "o_index", "w_valid", "w_index", "col_valid",
            "col_index", "valid_in", "valid_out", "k", "n", "cols", "depth", "rst_n")
    ids, vals, rows, time = {}, {}, [], None
    scope = []
    for line in open(vcd):
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "$scope":
            scope.append(tok[2])
        elif tok[0] == "$upscope":
            scope.pop()
        elif tok[0] == "$var" and len(scope) == 1 and tok[4] in want:
            ids[tok[3]] = tok[4]
        elif tok[0].startswith("#"):
            if time is not None and vals:
                rows.append((time, dict(vals)))
            time = int(tok[0][1:])
        elif tok[0][0] in "01xz" and tok[0][1:] in ids:
            vals[ids[tok[0][1:]]] = tok[0][0]
        elif tok[0][0] == "b" and len(tok) > 1 and tok[1] in ids:
            v = tok[0][1:]
            vals[ids[tok[1]]] = str(int(v, 2)) if set(v) <= set("01") else v
    if time is not None and vals:
        rows.append((time, dict(vals)))
    order = [w for w in want if w in vals]
    # yosys sim writes each clock edge and the half cycle after it: the
    # period is the first edge's time, and a cycle's row is its first.
    period = next((tm for tm, _ in rows if tm > 0), 1)
    out, seen = [], set()
    for tm, v in rows:
        c = tm // period
        if c in seen:
            continue
        seen.add(c)
        out.append("cycle %d: %s" % (c, " ".join("%s=%s" % (k, v[k]) for k in order if k in v)))
    return out




def check(spec, rtl_path, deps=(), work=None, timeout=PDR_TIMEOUT):
    """Prove spec's contract for the design in rtl_path, with deps the
    supplied blocks it instantiates. Returns a dict: status "proved"
    (every reachable state), "bounded" (every input over the given cycles
    from reset), "failed" (with the property, what it means, the cycle and
    the trace), "timeout", "error" or "n/a"."""
    fam = family(spec)
    if fam is None or (fam == "sequenced" and _top(spec) not in SEQ):
        return {"status": "n/a"}
    work = work or os.path.join(ROOT, "build_contract", _top(spec))
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    srcs = _sources(work, rtl_path, deps)
    if fam == "sequenced":
        info = SEQ[_top(spec)]
        small = dict(info["sizes"], **info["small"])
        out, err = _simulate(work, srcs, SEQ_TB.format(limit=200000, **_tb(
            spec, fam, {"sizes": small})), timeout)
        if err:
            return {"status": "error", "detail": err}
        if "RUN_HUNG" in out:
            return {"status": "failed", "property": "p_ends", "means": PROPS["p_ends"],
                    "detail": "a run at the small sizes did not end in 200000 cycles"}
        m = re.search(r"RUN_CYCLES (\d+)", out)
        if not m:
            return {"status": "error", "detail": "measuring a run: " + out[-300:]}
        run = int(m.group(1))
        # Every safety property, for every size the spec allows.
        res = _abc(work, srcs, _seq_wrapper(spec, info, info["sizes"]), "pdr", timeout)
        if res["status"] == "failed":
            return dict(res, run_cycles=run)
        safety = res["status"]
        if safety in ("timeout", "error"):
            # PDR out of time: the same properties, bounded, at small sizes.
            frames = min(MAX_FRAMES, 2 + 2 * (run + 2) + 8)
            res = _abc(work, srcs, _seq_wrapper(spec, info, small), "bmc", timeout, frames)
            if res["status"] == "failed":
                return dict(res, run_cycles=run)
            safety = "bounded" if res["status"] == "bounded" else safety
        # Every run ends: two runs back to back at the small sizes.
        frames = min(MAX_FRAMES, 2 + 2 * (run + 2) + 8)
        res = _abc(work, srcs, _seq_wrapper(spec, info, small, hang=4 * run + 32), "bmc",
                   timeout, frames)
        if res["status"] == "failed":
            return dict(res, run_cycles=run)
        return {"status": "proved" if safety == "proved" else safety, "safety": safety,
                "ends": res["status"], "frames": frames, "run_cycles": run,
                "sizes": info["sizes"], "small": small}
    lat = _spec_latency(spec)
    if lat is None:
        out, err = _simulate(work, srcs, STREAM_TB.format(**_tb(spec, fam)), timeout)
        if err:
            return {"status": "error", "detail": err}
        m = re.search(r"LATENCY (\d+)", out)
        if not m:
            return {"status": "failed", "property": "p_latency", "means": PROPS["p_latency"],
                    "detail": "one valid_in gave no valid_out in 200 cycles"}
        lat = int(m.group(1))
    res = _abc(work, srcs, _stream_wrapper(spec, lat, False), "pdr", timeout)
    res["latency"] = lat
    if _top(spec) in STATELESS and res["status"] == "proved":
        r2 = _abc(work, srcs, _stream_wrapper(spec, lat, True), "pdr",
                  min(timeout, SAME_TIMEOUT))
        if r2["status"] == "failed":
            return dict(r2, latency=lat)
        res["same"] = r2["status"]
    return res


def describe(res):
    """One line for a report, and for the agent's next prompt."""
    st = res["status"]
    if st == "failed":
        s = "contract FAILED: %s" % res["means"]
        if res.get("cycle") is not None:
            s += ", at cycle %d" % res["cycle"]
        if res.get("detail"):
            s += " (%s)" % res["detail"]
        if res.get("story"):
            s += ". " + res["story"]
        return s
    if "latency" in res and st == "proved":
        same = {"proved": "; one sample, one output: proved",
                "timeout": "; one sample, one output: out of time"}.get(res.get("same"), "")
        return "contract proved for every reachable state, latency %d%s" % (res["latency"], same)
    if st in ("proved", "bounded") and "safety" in res:
        s = ("handshake proved for every reachable state and size" if res["safety"] == "proved"
             else "handshake holds for every input over %d cycles" % res["frames"])
        e = {"bounded": "every run ends (two runs, %d cycles, small sizes)" % res["frames"],
             "timeout": "run ends: out of time"}.get(res["ends"], "run ends: %s" % res["ends"])
        return "%s; %s; a run takes %d cycles" % (s, e, res["run_cycles"])
    return "contract %s%s" % (st, (": " + res["detail"]) if res.get("detail") else "")


def check_gates(gates, log=print, timeout=PDR_TIMEOUT):
    """Every block in a gates directory, by its report's spec."""
    import spec2rtl
    rows = []
    for rep in sorted(f for f in os.listdir(gates) if re.match(r"report_[^.]+\.json$", f)):
        r = json.load(open(os.path.join(gates, rep)))
        spec = r.get("spec")
        tag = rep[7:-5]
        fn = tag + ".v"
        if not spec or not os.path.exists(os.path.join(gates, fn)):
            continue
        src = os.path.join(gates, fn)
        if fn in spec2rtl._RENAMED:
            text = spec2rtl.rename_module(open(src).read(), *spec2rtl._RENAMED[fn])
            src = os.path.join(gates, "contract_%s" % fn)
            with open(src, "w") as f:
                f.write(text)
        deps = supplied(gates, src, skip=fn)
        res = check(spec, src, deps, os.path.join(ROOT, "build_contract", tag), timeout)
        log("  %-16s %s" % (fn, describe(res)))
        rows.append((fn, res))
    return rows


_KEYWORDS = {"module", "begin", "end", "if", "else", "case", "assign", "always",
             "wire", "reg", "input", "output", "integer", "for", "function", "task",
             "localparam", "parameter", "generate", "endgenerate", "initial", "signed"}


def _instances(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return {m.group(1) for m in re.finditer(
        r"^\s*([A-Za-z_]\w*)\s*(?:#\s*\(.*?\)\s*)?[A-Za-z_]\w*\s*\(", text, re.M | re.S)
        if m.group(1) not in _KEYWORDS}


def supplied(gates, src, skip=None, fallback=()):
    """The files in gates that define the modules src instantiates, and
    theirs, transitively: the blocks the spec says are supplied. A module
    gates lacks is looked for in fallback, such as the generated ROM
    tables a saved block set leaves out."""
    defs = {}
    for d in (gates,) + tuple(fallback):
        for f in sorted(os.listdir(d)):
            if f.endswith(".v") and (d != gates or f != skip) and not f.startswith(
                    ("rtl_", "contract_", "resume_", "netlist", "tb_")):
                for m in re.findall(r"^\s*module\s+([A-Za-z_]\w*)",
                                    open(os.path.join(d, f)).read(), re.M):
                    defs.setdefault(m, os.path.join(d, f))
    own = set(re.findall(r"^\s*module\s+([A-Za-z_]\w*)", open(src).read(), re.M))
    want, seen, files = list(_instances(open(src).read()) - own), set(own), []
    while want:
        m = want.pop()
        if m in seen or m not in defs:
            continue
        seen.add(m)
        f = defs[m]
        if f not in files:
            files.append(f)
            want += list(_instances(open(f).read()))
    return files


if __name__ == "__main__":
    for fn, res in check_gates(sys.argv[1]):
        if res["status"] == "failed":
            print("\n".join("    " + l for l in res.get("trace", [])[-12:]))
