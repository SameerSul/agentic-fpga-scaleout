"""Port coverage of a block's testbench: how much of the block's interface
the generated testbench actually exercised, as a percentage of bins.

The testbench is run as the signoff runs it, with the block's ports dumped
to a VCD, and sampled on every rising clock edge after reset. The bins,
all stated on the ports the spec defines:

  toggle      every bit of every data port seen at 0 and at 1
  values      every data input seen at zero, above zero and at its
              largest value, a signed one below zero and at its most
              negative; every data output at zero and above it, a signed
              one below zero
  settings    a setting held for a row or a stream (eps, a scale, a shift,
              a size) seen with two values or more; a shift at 0 and above
              it, since the rounding term differs; a size at 1 and at the
              spec's largest
  indices     every output index of a row (0 .. length-1) seen with its
              valid
  protocol    a streaming block given samples on consecutive cycles and
              with gaps between them; a sequenced block run twice, and
              started on the cycle after a run ended
  saturation  the requantizer's sat seen at both rails

A bin not hit is a case the testbench never drove: the list of them is
what to add to it. This measures the stimulus, not the checking; mutation
testing (dv.py) measures whether the checks would see a fault.

  python3 coverage.py GATES_DIR       every block whose report is there
"""
import json
import os
import re
import shutil
import sys

import formal

ROOT = os.path.dirname(os.path.abspath(__file__))


def _instance(tb_text, top):
    m = re.search(r"^\s*%s\s+(?:#\s*\(.*?\)\s*)?(\w+)\s*\(" % re.escape(top), tb_text, re.M | re.S)
    return m.group(1) if m else None


def _tb_top(tb_text):
    return re.search(r"^\s*module\s+(\w+)", tb_text, re.M).group(1)


def _vcd(path, names):
    """{name: [value at each rising clock edge after reset]} for the scope's
    signals in names, read from a VCD of one scope. Values are ints, or
    None where a bit was x or z."""
    ids, cur, samples = {}, {}, {n: [] for n in names}
    clk_id = rst_id = None
    widths = {}
    block, in_defs = [], True

    def flush(block):
        rising = any(i == clk_id and v == 1 for i, v in block)
        if rising and cur.get(rst_id) == 1:
            for i, n in ids.items():
                if n in samples:
                    samples[n].append(cur.get(i))
        for i, v in block:
            cur[i] = v
    for line in open(path):
        tok = line.split()
        if not tok:
            continue
        if in_defs:
            if tok[0] == "$var":
                w, i, n = int(tok[2]), tok[3], tok[4]
                if n == "clk":
                    clk_id = i
                elif n == "rst_n":
                    rst_id = i
                if n in samples or n in ("clk", "rst_n"):
                    ids[i], widths[i] = n, w
            elif tok[0] == "$enddefinitions":
                in_defs = False
            continue
        t = tok[0]
        if t.startswith("#"):
            flush(block)
            block = []
            continue
        if t[0] in "01xzXZ" and t[1:] in ids:
            block.append((t[1:], int(t[0]) if t[0] in "01" else None))
        elif t[0] in "bB" and len(tok) > 1 and tok[1] in ids:
            v = t[1:]
            block.append((tok[1], int(v, 2) if set(v) <= set("01") else None))
    flush(block)
    return samples


# Held for a row or a stream, not a sample: covered by the values a run
# uses, not bit by bit (a 7-bit shift is never 127).
SETTINGS = ("eps", "scale", "scale_o", "scale_a", "scale_b", "shift", "shift_o", "shift_s",
            "n", "depth", "cols")
SIZE_MAX = {"n": "capacity", "depth": ("max_dim", "max_depth"), "cols": ("max_dim",)}


def _size_max(spec, n):
    keys = SIZE_MAX.get(n, ())
    keys = (keys,) if isinstance(keys, str) else keys
    for k in keys:
        if k in spec["parameters"]:
            return spec["parameters"][k]
    m = re.search(r"1 <= %s <= (\d+)" % n, " ".join(spec.get("behavior", [])))
    return int(m.group(1)) if m else None


def bins(spec, samples, count=None):
    """(hit, total, missed) over the spec's ports, from the samples."""
    hit, missed = 0, []
    total = 0
    ports = [p for p in spec["ports"] if p["name"] not in ("clk", "rst_n")]

    def bin_(name, ok):
        nonlocal hit, total
        total += 1
        if ok:
            hit += 1
        else:
            missed.append(name)
    for p in ports:
        n, w, sg = p["name"], p.get("width", 1), p.get("signed", False)
        vals = [v for v in samples.get(n, []) if v is not None]
        if n.endswith("_addr"):
            continue                # follows the sizes, which are binned
        if n in SETTINGS:
            bin_("%s with two values or more" % n, len(set(vals)) > 1)
            if n.startswith("shift"):
                bin_("%s at 0" % n, 0 in vals)
                bin_("%s above 0" % n, any(v > 0 for v in vals))
            top = _size_max(spec, n)
            if top:
                bin_("%s at 1" % n, 1 in vals)
                bin_("%s at its largest, %d" % (n, top), top in vals)
            continue
        if n.endswith("_index"):
            continue                # binned by value below, or by the sizes
        for b in range(w):
            ones = any((v >> b) & 1 for v in vals)
            zeros = any(not (v >> b) & 1 for v in vals)
            bin_("%s[%d]=1" % (n, b), ones)
            bin_("%s[%d]=0" % (n, b), zeros)
        if w > 1 and not n.endswith("_index"):
            # The spec's domain: exp's x is non-positive, the reciprocal's
            # x non-zero; a bin outside it is a case no caller makes.
            desc = p.get("desc", "")
            nonpos, nonzero = "non-positive" in desc, "non-zero" in desc
            sv = [v - (1 << w) if sg and v >> (w - 1) else v for v in vals]
            if not nonzero:
                bin_("%s zero" % n, 0 in sv)
            if not nonpos:
                bin_("%s above zero" % n, any(v > 0 for v in sv))
            if sg:
                bin_("%s below zero" % n, any(v < 0 for v in sv))
            if p["dir"] == "input":
                if not nonpos:
                    bin_("%s largest" % n, ((1 << (w - 1)) - 1 if sg else (1 << w) - 1) in sv)
                if sg:
                    bin_("%s most negative" % n, -(1 << (w - 1)) in sv)
    names = {p["name"] for p in ports}
    valid = next((v for v in names if v.endswith("_valid") and v[:-6] + "_index" in names), None)
    if valid and count:
        index = valid[:-6] + "_index"
        seen = {i for v, i in zip(samples.get(valid, []), samples.get(index, []))
                if v == 1 and i is not None}
        for i in range(count):
            bin_("%s %d given" % (index, i), i in seen)
    if "valid_in" in names:
        v = [x == 1 for x in samples.get("valid_in", [])]
        bin_("valid_in on consecutive cycles", any(a and b for a, b in zip(v, v[1:])))
        bin_("valid_in with a gap between samples",
             any(a and not b and c for a, b, c in zip(v, v[1:], v[2:])))
    if "start" in names and "busy" in names:
        st = [x == 1 for x in samples.get("start", [])]
        bz = [x == 1 for x in samples.get("busy", [])]
        bin_("more than one run", sum(st) > 1)
        bin_("a start on the cycle after a run ended",
             any(b0 and not b1 and s for b0, b1, s in zip(bz, bz[1:], st[1:])))
    if "sat" in names and "q_out" in names:
        q = [x for x in samples.get("q_out", []) if x is not None]
        bin_("q_out at the top rail", 32767 in q)
        bin_("q_out at the bottom rail", 32768 in q)
    return hit, total, missed


def _count(spec):
    import contracts
    info = contracts.SEQ.get(spec["top_module"])
    if not info:
        return None
    kind, name = info["count"]
    if kind == "param":
        return spec["parameters"][name]
    return None                     # a run-time length: every index the runs use


def measure(spec, tb_path, rtl_path, deps=(), work=None, timeout=600):
    """Run tb_path on rtl_path with the DUT's ports dumped and count the
    bins. Returns {"hit", "total", "percent", "missed"} or {"error"}."""
    work = work or os.path.join(ROOT, "build_coverage", spec["top_module"])
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    tb = open(tb_path).read()
    top = spec["top_module"]
    inst = _instance(tb, top)
    if not inst:
        return {"error": "the testbench has no %s instance" % top}
    with open(os.path.join(work, "cov_dump.v"), "w") as f:
        f.write("module cov_dump;\n  initial begin\n    $dumpfile(\"cov.vcd\");\n"
                "    $dumpvars(1, %s.%s);\n  end\nendmodule\n" % (_tb_top(tb), inst))
    srcs = []
    for i, s in enumerate([tb_path, rtl_path] + list(deps)):
        d = "s%d_%s" % (i, os.path.basename(s))
        shutil.copyfile(s, os.path.join(work, d))
        srcs.append(d)
    rc, out = formal._run(["iverilog", "-g2005", "-o", "cov.vvp", "cov_dump.v"] + srcs,
                          work, timeout)
    if rc != 0:
        return {"error": "compile: " + " ".join(out.strip().splitlines()[-3:])}
    rc, out = formal._run(["vvp", "-n", "cov.vvp"], work, timeout)
    if "TB_RESULT: PASS" not in out:
        return {"error": "the testbench did not pass"}
    names = [p["name"] for p in spec["ports"]]
    samples = _vcd(os.path.join(work, "cov.vcd"), names)
    hit, total, missed = bins(spec, samples, _count(spec))
    return {"hit": hit, "total": total, "percent": round(100.0 * hit / max(1, total), 1),
            "missed": missed}


def testbench_sources(gates, tb, rtl, top):
    """What the testbench and the design instance, from gates: the
    supplied blocks, an integration testbench's other blocks, the ROMs;
    never a second definition of the block under test."""
    import contracts
    out = []
    for f in contracts.supplied(gates, tb) + contracts.supplied(gates, rtl):
        mods = re.findall(r"^\s*module\s+(\w+)", open(f).read(), re.M)
        if top not in mods and f not in out:
            out.append(f)
    return out


def summary(missed, n=8):
    """The bins not hit, grouped: a port's toggles as one entry."""
    groups = {}
    for m in missed:
        k = re.sub(r"\[\d+\]=[01]", "[bits]", m)
        groups[k] = groups.get(k, 0) + 1
    items = ["%s (%d)" % (k, c) if c > 1 else k for k, c in groups.items()]
    return ", ".join(items[:n]) + (", ..." if len(items) > n else "")


if __name__ == "__main__":
    import spec2rtl
    g = sys.argv[1]
    for rep in sorted(f for f in os.listdir(g) if re.match(r"report_[^.]+\.json$", f)):
        tag = rep[7:-5]
        spec = json.load(open(os.path.join(g, rep)))["spec"]
        rtl = os.path.join(g, "rtl_%s.v" % tag)
        import contracts
        tb = os.path.join(g, "tb_%s.v" % tag)
        res = measure(spec, tb, rtl, testbench_sources(g, tb, rtl, spec["top_module"]))
        print("%-16s %s" % (tag, res.get("error") or "%5.1f%% of %d bins; missed: %s" % (
            res["percent"], res["total"], summary(res["missed"]))))
