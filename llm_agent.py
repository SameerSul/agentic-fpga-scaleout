"""LLM design agent: the drop-in replacement for RuleBasedAgent.

Same interface, propose(spec, feedback_history) -> (verilog_source, notes),
but the Verilog comes from a real language model. The prompt serializes the
spec, the agent's own previous attempt, and the parsed tool feedback (failing
test, expected vs got, compile and synthesis errors); the orchestrator's
verify-and-feed-back loop is unchanged, so a wrong first cut converges the
same way the rule-based seeded bugs do, except the fixes are the model's.

Three backends, all Python 3 stdlib (no SDK dependency), picked by the
CHIPLET_LLM environment variable or autodetected in this order:

    anthropic    Anthropic Messages API via urllib, needs ANTHROPIC_API_KEY
    ollama       local Ollama server at localhost:11434 (lightweight local
                 models, e.g. qwen2.5-coder)
    claude-cli   the Claude Code CLI in print mode (claude -p), which reuses
                 an existing login with zero extra setup

CHIPLET_LLM accepts an optional model after a colon, e.g.
CHIPLET_LLM=ollama:qwen2.5-coder:7b or CHIPLET_LLM=claude-cli:haiku.
Select the agent with CHIPLET_AGENT=llm or: python3 chiplet_flow.py --agent llm
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULTS = {"anthropic": "claude-haiku-4-5", "ollama": "qwen2.5-coder:7b",
            "claude-cli": "haiku"}
MAX_FEEDBACK = 4      # most recent failure records included in the prompt
MAX_ERR_LINES = 8     # tool error lines per record
MAX_MISMATCHES = 3    # testbench mismatches per record

RULES = """Write synthesizable Verilog for the block specified below.

Hard rules:
- Output ONLY the Verilog source. No explanation, no markdown fences.
- One module named exactly '{top}'. Port names, directions, and widths
  exactly as listed in the spec. Define no other module and use no
  includes; where the spec names supplied modules, instantiate them.
- Verilog-2005 only: no SystemVerilog (no logic, always_ff, always_comb,
  typedef, enum, interfaces, packed structs).
- Fully synchronous: state changes only on posedge clk. rst_n and clear are
  synchronous as described in the spec.
- The behavior section of the spec is normative, including latency and the
  priority of clear."""


def _ollama_up():
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def pick_backend(choice=None):
    """Return (backend, model). Explicit choice ('name' or 'name:model')
    wins; otherwise autodetect API key, then local Ollama, then Claude CLI."""
    choice = choice or os.environ.get("CHIPLET_LLM")
    if choice:
        backend, _, model = choice.partition(":")
        if backend not in DEFAULTS:
            raise RuntimeError("unknown CHIPLET_LLM backend %r, expected one "
                               "of %s" % (backend, sorted(DEFAULTS)))
        return backend, model or DEFAULTS[backend]
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", DEFAULTS["anthropic"]
    if _ollama_up():
        return "ollama", DEFAULTS["ollama"]
    if shutil.which("claude"):
        return "claude-cli", DEFAULTS["claude-cli"]
    raise RuntimeError(
        "no LLM backend available: set ANTHROPIC_API_KEY, run an Ollama "
        "server, or install the Claude CLI (or set CHIPLET_LLM explicitly)")


_BAD_LITERAL = re.compile(r"\d+\s*'\s*[sS]?[dDhHbBoO]\s*[-(]")
_EXPR_SELECT = re.compile(r"[)}]\s*\[[^\]]*\]")
_ERR_LOC = re.compile(r"^(?P<path>[^\s:]+(?: [^\s:]+)*\.s?v):(?P<line>\d+):\s*(?P<msg>.*)$")


def annotate_errors(errors, rtl):
    """Quote the source line each compile error points at.

    iverilog reports 'file.v:141: syntax error' and nothing else. An
    engineer opens the file at line 141; the model sees its draft as plain
    text and has to count to it, which it does badly. Measured on the
    requantizer: told only a line number, it restructured the code around
    an illegal literal (48'sd-128) and kept that literal in every draft.
    Absolute paths are cut to the file name, since they cost tokens and
    carry nothing. Lines in the testbench or the supplied dependency and
    table files are not the writer's code, so they are left unquoted.
    """
    lines = rtl.splitlines() if rtl else []
    out = []
    for e in errors:
        m = _ERR_LOC.match(e.strip())
        if not m:
            out.append(_explain(e, lines))
            continue
        name = os.path.basename(m.group("path"))
        n = int(m.group("line"))
        text = _explain("%s line %d: %s" % (name, n, m.group("msg")), lines)
        ours = not (name.startswith("tb_") or name.endswith("_dep.v")
                    or name.endswith("_rom.v"))
        if ours and 1 <= n <= len(lines):
            src = lines[n - 1]
            text += "  | source: " + src.strip()
            # A sign inside a sized literal. Traced on the requantizer, the
            # model wrote 48'sd-128, then 8'sd(-128), then 47'sd-128, each
            # time with the line quoted back to it: it is a misconception
            # about the syntax, so the feedback states the rule.
            if _BAD_LITERAL.search(src):
                text += ("  | meaning: a sized literal cannot hold a sign "
                         "or parentheses after the base. Put the minus in "
                         "front of the whole literal: -8'sd128, not "
                         "8'sd-128 or 8'sd(-128).")
            # A select taken of an expression, not a name. Traced on
            # softmax, Haiku wrote (s_buf[i] - mx)[12:0] in every one of
            # eight drafts, each quoted back to it as a syntax error.
            if _EXPR_SELECT.search(src):
                text += ("  | meaning: Verilog-2005 cannot take a bit or part "
                         "select of an expression, a concatenation or a "
                         "function result: (a - b)[12:0] is illegal. Assign "
                         "the expression to a wire of its full width and "
                         "select bits from that wire, or assign it straight "
                         "to a narrower reg, which keeps the low bits.")
        out.append(text)
    return out


def _explain(err, lines):
    """Restate a yosys message whose wording names an internal cell rather
    than the construct in the design. This is the tool's verdict in the
    design's terms, not a guess at a cause: the cell type in the message
    is an asynchronously reset flop, and nothing else produces it.
    Traced on softmax, the only report was '$_DFFE_PN0P_ cannot be
    legalized', after a draft that passed simulation with an asynchronous
    reset the testbench cannot see."""
    m = re.search(r"Unable to bind [\w/ ]*`dut\.(\w+)", err)
    if m:
        err += ("  | meaning: the testbench checks an internal memory named "
                "%s that the spec requires. Declare it with exactly that "
                "name and layout." % m.group(1))
    if "Signing cast requires SystemVerilog" in err:
        err += ("  | meaning: signed'(x) is a SystemVerilog cast. In "
                "Verilog-2005 write $signed(x), or $unsigned(x).")
    if "is not allowed in a constant expression" in err:
        err += ("  | meaning: a part-select x[msb:lsb] needs constant bounds. "
                "For a slice at a variable offset use the indexed form "
                "x[base +: width], whose width is constant: x[i*16 +: 16], "
                "not x[i*16+15:i*16].")
    if "declaration in unnamed block requires SystemVerilog" in err:
        err += ("  | meaning: Verilog-2005 allows a reg, wire or integer to "
                "be declared only at module level or in a named block. Move "
                "the declaration out of the begin ... end, to the top of the "
                "module.")
    if "async set or reset are not supported" in err:
        where = ["line %d: %s" % (i + 1, l.strip())
                 for i, l in enumerate(lines)
                 if "always" in l and "rst" in l and "edge" in l
                 and l.count("edge") > 1][:3]
        err += ("  | meaning: a flop has an asynchronous reset, which this "
                "flow does not support. The reset must be synchronous: "
                "always @(posedge clk) with if (!rst_n) inside the block."
                + ("  | at " + "; ".join(where) if where else ""))
    return err


# A declaration runs to the next port keyword as well as to ';', so an
# ANSI port list on one line does not fold every port into the first.
_DECL = re.compile(r"\b(input|output|inout|wire|reg)\b"
                   r"((?:(?!\b(?:input|output|inout)\b)[^;])*)")
_MUL = re.compile(r"(\$signed\s*\(\s*)?\b([A-Za-z_]\w*)\b\s*\*\s*"
                  r"(\$signed\s*\(\s*)?\b([A-Za-z_]\w*)\b")
# A product of two whole names with a whole name added to it or subtracted
# from it, either side: acc + a * b, acc + (a * b), a * b - acc.
_ADD_MUL = re.compile(
    r"\b([A-Za-z_]\w*)\s*[-+]\s*\(?\s*([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\b(?!\s*[\[(])"
    r"|\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\s*\)?\s*[-+]\s*([A-Za-z_]\w*)\b(?!\s*[\[(])")


def _signedness(rtl):
    """Map each declared name to whether it was declared signed."""
    sig = {}
    for line in rtl.splitlines():
        line = line.split("//")[0]
        for m in _DECL.finditer(line):
            body = m.group(2)
            signed = bool(re.search(r"\bsigned\b", body))
            body = re.sub(r"\[[^\]]*\]", " ", body)
            body = re.sub(r"\b(wire|reg|signed|unsigned|integer)\b", " ",
                          body).split("=")[0]
            for name in re.findall(r"[A-Za-z_]\w*", body):
                sig.setdefault(name, signed)
    return sig


_ASSIGN = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*(<=|=)(?!=)\s*([^;]*);")
_ENDS = re.compile(r"from (?:register|port) ([A-Za-z_]\w*)\S* "
                   r"to (?:register|port) ([A-Za-z_]\w*)")


def timing_path(rtl, start, end, depth=10):
    """The lines of rtl a failing path runs through, from start to end: the
    endpoint's assignment back through wires and blocking temporaries to
    one that reads start. [] when no such chain is found.

    Traced on the attention head: two models each got a draft through all
    236 checks and synthesis, then missed timing on "from register j_issue
    to register out_acc" and spent their last drafts elsewhere. The path
    was an address subtract, a 256-entry buffer read, a 17 by 16 multiply
    and a 32-bit add, all in one cycle, on four lines of their own code."""
    assigns = {}
    for n, line in enumerate(rtl.splitlines(), 1):
        code, pos = line.split("//")[0], 0
        while True:
            m = _ASSIGN.search(code, pos)
            if not m:
                break
            before = code[:m.start()]
            if before.count("(") > before.count(")"):
                pos = m.end(1)  # a comparison inside if ( ... ), not a write
                continue
            pos = m.end()
            names = set(re.findall(r"(?<!['$\w])([A-Za-z_]\w*)", m.group(3)))
            assigns.setdefault(m.group(1), []).append(
                (n, code.strip(), m.group(2), names))
    reg = {k for k, v in assigns.items() if any(a[2] == "<=" for a in v)}
    # An endpoint inside a module instance (requant_inst/_21444_ in the
    # report) is reached through the signals wired into that instance:
    # traced on Haiku's RMSNorm, whose -7.12 ns path ran into requant.
    if end not in assigns:
        lines = rtl.splitlines()
        for n, line in enumerate(lines, 1):
            if not re.search(r"\b[A-Za-z_]\w*\s+(?:#\s*\(.*\)\s*)?%s\s*\("
                             % re.escape(end), line.split("//")[0]):
                continue
            for k in range(n - 1, min(n + 40, len(lines))):
                code = lines[k].split("//")[0]
                for m in re.finditer(r"\.(\w+)\s*\(\s*([^()]*?)\s*\)", code):
                    names = set(re.findall(r"(?<!['$\w])([A-Za-z_]\w*)", m.group(2)))
                    assigns.setdefault(end, []).append(
                        (k + 1, "%s, into %s" % (m.group(0), end), "=", names))
                if ");" in code:
                    break
            break
    frontier, seen = [(end, [])], {end}
    for _ in range(depth):
        nxt = []
        for name, path in frontier:
            for n, text, op, names in assigns.get(name, []):
                here = [(n, text)] + path
                if start in names:
                    return sorted(set(here))
                for r in names - seen:
                    if r in assigns and r not in reg:
                        seen.add(r)
                        nxt.append((r, here))
        frontier = nxt
    return []


def _decl_names(code, kind):
    return {m.group(2) for m in re.finditer(
        r"\b%s\b([^;,)]*?)\b([A-Za-z_]\w*)\s*(?=[,;)])" % kind, code)}


def handshake_findings(rtl):
    """An output driven straight from a module instance's output while the
    valid beside it is a register set from that instance's valid, so the
    two reach the ports a clock edge apart.

    Traced on RMSNorm: Haiku's and Sonnet's drafts both wrote assign
    o_data = requant's q_out and set o_valid and o_index a cycle after
    requant's valid_out, so every output was its neighbour's, and the
    testbench, which cannot tell that from reading x a cycle early, said
    only that. Registering o_data beside o_valid passed all 268 checks."""
    code = "\n".join(l.split("//")[0] for l in rtl.splitlines())
    ins, outs = _decl_names(code, "input"), _decl_names(code, "output")
    conn = set(re.findall(r"\.\w+\s*\(\s*([A-Za-z_]\w*)\s*\)", code))
    driven = set(re.findall(r"\bassign\s+([A-Za-z_]\w*)", code))
    driven |= set(re.findall(
        r"\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*(?:<=|=)(?!=)", code))
    inst = conn - driven - ins          # driven only by an instance
    lines = code.splitlines()
    data = []
    for n, l in enumerate(lines, 1):
        m = re.match(r"\s*assign\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)"
                     r"\s*(?:\[[^\]]*\])?\s*;", l)
        if m and m.group(1) in outs and m.group(2) in inst:
            data.append((n, m.group(1), m.group(2)))
    for n, l in enumerate(lines, 1):
        m = re.search(r"\bif\s*\(\s*([A-Za-z_]\w*)\s*\)", l)
        if not (data and m and m.group(1) in inst):
            continue
        for k in range(n - 1, min(n + 4, len(lines))):
            v = re.search(r"\b([A-Za-z_]\w*)\s*<=", lines[k])
            if v and v.group(1) in outs and v.group(1) not in [d[1] for d in data]:
                dn, dp, dw = data[0]
                vp, vw = v.group(1), m.group(1)
                return [
                    "line %d: %s is driven straight from %s, an output of a "
                    "module instance, but %s (line %d) is a register set when "
                    "%s is high, so it reaches the port one clock edge after "
                    "the value it describes. Unless %s holds its value for "
                    "that extra cycle, %s has already moved on to the next "
                    "result by then. Drive them the same way: register %s "
                    "beside %s, or drive %s from %s directly."
                    % (dn, dp, dw, vp, k + 1, vw, dw, dp, dp, vp, vp, vw)]
    return []


def code_findings(rtl, history):
    """Facts about the failing draft, found by inspecting it and the
    testbench output. Not diagnoses: each is true of the code whether or
    not it is the cause, and the prompt labels them that way.

    Both come from traces. The requantizer multiplied a signed
    accumulator by an unsigned scale for five drafts running, reading
    'acc -128 in, 127 saturated out' each time without connecting it to
    Verilog making the product unsigned. Softmax emitted x, which always
    means a value that was never written or reset.
    """
    out = []
    if not rtl:
        return out
    sig = _signedness(rtl)
    for n, line in enumerate(rtl.splitlines(), 1):
        code = line.split("//")[0]
        for m in _MUL.finditer(code):
            a, b = m.group(2), m.group(4)
            if m.group(1) or m.group(3) or a not in sig or b not in sig:
                continue
            if sig[a] != sig[b]:
                s_, u_ = (a, b) if sig[a] else (b, a)
                out.append(
                    "line %d: %s * %s multiplies a signed operand (%s) by an "
                    "unsigned one (%s). Verilog then evaluates the whole "
                    "expression as unsigned, so a negative %s multiplies as "
                    "a large positive number. For a signed value times a "
                    "non-negative unsigned one, write %s * $signed({1'b0, %s})."
                    % (n, a, b, s_, u_, s_, s_, u_))
        # Traced on Haiku's RMSNorm: ssq <= ssq + (x_data * x_data), with
        # x_data signed and ssq unsigned, for eight drafts. The unsigned
        # sum makes the product unsigned too, so each negative x squared
        # as (x + 2**16)**2 and the sum came out 19 times too large.
        for m in _ADD_MUL.finditer(code):
            acc, a, b = m.group(1, 2, 3) if m.group(1) else m.group(6, 4, 5)
            if sig.get(a) and sig.get(b) and sig.get(acc) is False \
                    and not re.search(r"\$signed\s*\(\s*$", code[:m.start()]):
                out.append(
                    "line %d: %s * %s multiplies two signed values, but %s, "
                    "added on the same line, is unsigned, and Verilog makes "
                    "an expression unsigned when any operand is. %s and %s "
                    "are then zero-extended before the multiply, so a "
                    "negative value multiplies as a large positive one. "
                    "Write $signed({1'b0, %s}) for the unsigned operand, or "
                    "compute the product into its own signed wire first."
                    % (n, a, b, acc, a, b, acc))
        # Traced on Haiku's RMSNorm over a head: {8'b0, x * x}, where each
        # part of a concatenation keeps its own width, so a 16-bit x times
        # itself kept 16 bits and the sum of squares came out tiny.
        for m in re.finditer(r"\{([^{}]*)\}", code):
            pm = re.search(r"([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)", m.group(1))
            if pm and "," in m.group(1):
                out.append(
                    "line %d: %s is inside a concatenation, where each part "
                    "is sized by itself: a product there is only as wide as "
                    "its wider operand, so its high bits are lost. Compute "
                    "the product into a wire as wide as the result first, "
                    "then use that wire." % (n, pm.group(0)))
                break
        for m in re.finditer(r"(\$signed\s*\(\s*)?(\{[^{}]*\})", code):
            if m.group(1):
                continue
            # Only a whole signed signal inside the braces loses a sign. A
            # part-select is unsigned by the language already, and
            # {1'b0, x[22:0]} is a deliberate zero-extension: the reference
            # requantizer does exactly that, and flagging it would teach the
            # agent to distrust a correct construct.
            whole = {v for v in re.findall(r"([A-Za-z_]\w*)(?!\s*\[)",
                                           m.group(2))}
            # {x[msb], x} and {{n{x[msb]}}, x} are sign extension, the
            # standard idiom for widening a signed value by hand, and they
            # produce the right bits. Sonnet's first requantizer draft
            # passed all 209 checks using one; flagging it would send a
            # model after a bug that is not there.
            ext = re.match(r"\{\s*(?:\d+\s*\{\s*)?([A-Za-z_]\w*)\s*\[[^\]]*\]",
                           m.group(2))
            if ext and ext.group(1) in whole:
                continue
            rest = code[:m.start()] + code[m.end():]
            arith = re.search(r"[-+*]", rest.split("=", 1)[-1]) or \
                re.search(r"[-+*]\s*$", code[:m.start()])
            if arith and any(sig.get(v) for v in whole):
                out.append(
                    "line %d: %s is a concatenation, and in Verilog a "
                    "concatenation is always unsigned, even when its parts "
                    "are signed. Used in arithmetic next to a signed value it "
                    "makes the whole expression unsigned, so negative values "
                    "turn into large positive ones. To shift a signed value, "
                    "write x <<< n, or wrap it as $signed(%s)."
                    % (n, m.group(2), m.group(2)))
                break
    out += handshake_findings(rtl)
    last = history[-1] if history else {}
    ends = last.get("stage") == "timing" and _ENDS.search(
        last.get("critical_path") or "")
    lines = ends and timing_path(rtl, ends.group(1), ends.group(2))
    if lines:
        out.append(
            "the failing timing path, from %s to %s, runs through these lines "
            "of your code, all in one clock cycle: %s. Register a value part "
            "way along it and use it a cycle later, adjusting the control so "
            "the result still lines up with the rest."
            % (ends.group(1), ends.group(2),
               "; ".join("line %d: %s" % (n, t.rstrip(";")) for n, t in lines)))
    for mm in last.get("mismatches") or []:
        if any(k.startswith("got") and re.fullmatch(r"[xXzZ]+", str(v))
               for k, v in mm.items()):
            out.append("the testbench read x (unknown) from the design: that "
                       "value came from a register or memory that was never "
                       "written or reset, or from an address never loaded.")
            break
    return out


def condense_feedback(history, rtl=None):
    """Bound the prompt: the most recent failure records, each trimmed to the
    fields an engineer would actually read. rtl is the draft the most
    recent record was produced from; its failing lines get quoted."""
    out = []
    recent = history[-MAX_FEEDBACK:]
    for i, fb in enumerate(recent):
        rec = {"iteration": fb.get("iteration"), "stage": fb.get("stage"),
               "status": fb.get("status")}
        if fb.get("phase"):
            rec["phase"] = fb["phase"]
        if fb.get("errors"):
            # Errors before warnings. iverilog prints a width warning ahead
            # of the syntax error that actually failed the build, and with a
            # bounded number of lines the warning was taking the real
            # error's slot: traced on the attention head.
            errs = sorted(fb["errors"], key=lambda e: "warning" in e.lower())
            errs = errs[:MAX_ERR_LINES]
            # Only the latest record matches the draft we are holding.
            rec["tool_errors"] = annotate_errors(
                errs, rtl if i == len(recent) - 1 else None)
        if fb.get("mismatches"):
            rec["testbench_mismatches"] = fb["mismatches"][:MAX_MISMATCHES]
        for k in ("worst_slack_ns", "clock_period_ns", "critical_path"):
            if fb.get(k) is not None:
                rec[k] = fb[k]
        out.append(rec)
    return out


def build_prompt(spec, history, last_rtl, note=None):
    parts = [RULES.format(top=spec["top_module"]),
             "", "SPEC:", json.dumps(spec, indent=2)]
    if last_rtl:
        parts += ["", "YOUR PREVIOUS ATTEMPT (it failed, revise it):",
                  last_rtl]
    if note:
        parts += ["", "NOTE: " + note]
    if history:
        parts += ["", "TOOL FEEDBACK, most recent last (make these pass):",
                  json.dumps(condense_feedback(history, last_rtl), indent=2)]
        found = code_findings(last_rtl, history)
        if found:
            parts += ["", "FACTS ABOUT YOUR PREVIOUS ATTEMPT, found by "
                          "inspecting it (true of the code; check whether "
                          "they explain the failure):"] + ["- " + f for f in found]
    return "\n".join(parts)


def is_complete_module(rtl, top):
    return bool(re.search(r"\bmodule\s+%s\b" % re.escape(top), rtl)) \
        and "endmodule" in rtl


def ask_for_module(ask, prompt, top, retries=2):
    """Call the model and insist on a whole module.

    A reply that is a fragment, as if proposing an edit, used to be
    compiled as the design and spend a full iteration of every gate on a
    formatting slip; traced on the requantizer, one draft was a single
    line. It is re-asked on the spot instead, which costs one call and no
    tool time. Returns (rtl, calls made).
    """
    rtl = extract_verilog(ask(prompt))
    calls = 1
    while not is_complete_module(rtl, top) and calls <= retries:
        rtl = extract_verilog(ask(
            prompt + "\n\nYour previous reply was not a complete module. "
            "Reply with the entire Verilog source, from 'module %s' to "
            "'endmodule', and nothing else." % top))
        calls += 1
    return rtl, calls


def extract_verilog(text):
    """Robustly pull the module out of a model response: prefer a fenced
    block, then trim to the module..endmodule span."""
    m = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = re.search(r"\bmodule\b", text)
    end = text.rfind("endmodule")
    if start and end != -1:
        text = text[start.start():end + len("endmodule")]
    return text.strip() + "\n"


def call_anthropic(prompt, model):
    body = {"model": model, "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        ANTHROPIC_URL, data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    return "".join(b.get("text", "") for b in resp["content"])


def call_ollama(prompt, model):
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2}}
    req = urllib.request.Request(
        OLLAMA_URL + "/api/generate", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["response"]


# A hung or rate-limited call is a property of the transport, not of the
# design being written, so it gets retried rather than ending the block.
# This is not hypothetical: the exponential was recorded as a convergence
# failure once when what actually happened was one call sitting at zero
# CPU until it hit the timeout, which aborted the whole run.
#
# A call that is still thinking is not hung, though. On the attention head
# Sonnet at low effort streamed thinking from the second second on, 13 kB
# of it in three minutes and not a character of answer, and a flat 25
# minute timeout killed it three times running, 75 minutes a block spent
# asking the same question again. So the CLI's events are read as they
# come: a call that goes silent has hung and is retried, and a call still
# streaming at the cap is a model that has not answered in the time it is
# given, which another attempt at the same prompt will not change.
CLI_TIMEOUT_S = 600
# Measured: one Sonnet softmax call at low effort took 852 s and returned a
# complete module. The larger models get room for that; haiku keeps 600.
CLI_TIMEOUT_LARGE_S = 1500
CLI_SILENCE_S = 180
# Silent calls are retried over about twenty minutes before a model's turn
# at a block ends: traced, three silent calls inside ten minutes, with five
# and ten seconds between them, handed softmax on from Sonnet after four
# drafts in the end-to-end run.
CLI_ATTEMPTS = 6


class StillThinking(RuntimeError):
    """The model was still producing events when its time ran out."""


def cli_timeout(model):
    """The cap on one call. A hung call is caught by silence now, so the
    cap bounds only how long a model may take over an answer, and every
    model gets the same. Haiku's used to be 600 s, from when a flat
    timeout was the only way to catch a hang; in run 6 it cut off
    Haiku's softmax while it was still streaming, a draft after one had
    passed all 132 checks and failed only on its reset."""
    return CLI_TIMEOUT_LARGE_S


def cli_command(model, thinking=True):
    """The CLI invocation for one writer call.

    --tools "": the model writes text and the flow runs the tools. Left
    enabled, print mode is an agent that may go off and run its own.
    --effort: larger models think at length by default, and on these specs
    Sonnet spent over ten minutes in extended thinking without emitting a
    character (778 thinking deltas in 150 s, measured), so every call hit
    the timeout. Low effort returned a complete requantizer in about four
    minutes, and that first draft passed all 209 checks. Haiku keeps its
    default so its results stay comparable with every earlier run.
    CHIPLET_LLM_EFFORT overrides either way.
    The output is the CLI's event stream, so a thinking model can be told
    from a hung one; the answer is its final result event.
    """
    cmd = ["claude", "-p", "--model", model, "--tools", ""]
    effort = os.environ.get("CHIPLET_LLM_EFFORT") or \
        (None if "haiku" in model else "low")
    if effort:
        cmd += ["--effort", effort]
    if not thinking:
        # Measured on the attention head: with thinking off Sonnet streams
        # its answer from the third second, a whole module in 105 s, where
        # with it on an hour of thinking gave nothing.
        cmd += ["--settings", json.dumps({"alwaysThinkingEnabled": False})]
    return cmd + ["--output-format", "stream-json", "--verbose",
                  "--include-partial-messages"]


class CliResult:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def run_cli(cmd, prompt, cap, silence=None, cwd=None):
    """One CLI call. Returns a CliResult whose stdout is the answer.
    Raises subprocess.TimeoutExpired when no event arrives for silence
    seconds (hung), and StillThinking when events are still arriving at
    cap seconds (not hung, but no answer in the time given)."""
    import select
    silence = silence or CLI_SILENCE_S
    err = tempfile.TemporaryFile(mode="w+")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=err, text=True, cwd=cwd)
    try:
        p.stdin.write(prompt)
        p.stdin.close()
        t0 = last = time.time()
        result, text, is_error = None, [], False
        thought = 0
        while True:
            now = time.time()
            if now - t0 > cap:
                # A whole module already written is the answer, whatever
                # follows it. Traced: with thinking off, Sonnet and Opus
                # still streamed for the whole 1500 s on the head norm,
                # and nothing said whether it was text or thinking.
                said = "".join(text)
                if re.search(r"\bmodule\b[\s\S]*\bendmodule\b", said):
                    return CliResult(0, said, "")
                raise StillThinking(
                    "still thinking after %ds, no answer (%d chars of "
                    "thinking, %d of text)" % (cap, thought, len(said)))
            if now - last > silence:
                raise subprocess.TimeoutExpired(cmd, silence)
            ready, _, _ = select.select([p.stdout], [], [], 5)
            if not ready:
                continue
            line = p.stdout.readline()
            if not line:
                break
            last = time.time()
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "stream_event":
                d = (ev.get("event") or {}).get("delta") or {}
                if d.get("type") == "text_delta":
                    text.append(d.get("text", ""))
                elif d.get("type") == "thinking_delta":
                    thought += len(d.get("thinking", ""))
            elif ev.get("type") == "result":
                result = ev.get("result")
                is_error = bool(ev.get("is_error"))
        rc = p.wait(timeout=30)
        err.seek(0)
        out = result if result is not None else "".join(text)
        if is_error and rc == 0:
            rc = 1
        return CliResult(rc, out or "", err.read())
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        err.close()


def call_claude_cli(prompt, model, attempts=CLI_ATTEMPTS, thinking=True):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            # A neutral directory, so the CLI does not load this repo's
            # project context into what should be a self-contained prompt.
            r = run_cli(cli_command(model, thinking), prompt, cli_timeout(model),
                        cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            last = "timed out: no output for %ds" % CLI_SILENCE_S
        except StillThinking as e:
            # The same prompt again would think the same way. The agent may
            # ask again with thinking off; otherwise this model's attempt
            # at the block is over, and the next agent takes it.
            raise StillThinking("claude CLI failed after %d attempt(s): %s"
                                % (attempt, e))
        else:
            if r.returncode == 0:
                return r.stdout
            # The CLI reports auth and API errors on stdout, not stderr.
            last = (r.stderr.strip() or r.stdout.strip())[:500]
            # An auth or argument fault will fail the same way every time;
            # only transport faults are worth another attempt.
            if not _is_retryable(last):
                break
        if attempt < attempts:
            time.sleep(min(300, 30 * 2 ** (attempt - 1)))
    raise RuntimeError("claude CLI failed after %d attempt(s): %s"
                       % (attempt, last))


def _is_retryable(detail):
    d = (detail or "").lower()
    if any(k in d for k in ("not logged in", "unauthor", "invalid api key",
                            "authentication", "unknown option",
                            "no such model")):
        return False
    return any(k in d for k in ("timed out", "timeout", "rate limit",
                                "429", "overloaded", "503", "502", "500",
                                "connection", "network", "temporarily"))


CALLERS = {"anthropic": call_anthropic, "ollama": call_ollama,
           "claude-cli": call_claude_cli}


class LLMAgent:
    """propose() keeps the agent's own last attempt so revisions are edits of
    real code, not regenerations from scratch."""

    def __init__(self, choice=None):
        self.backend, self.model = pick_backend(choice)
        self.last_rtl = None
        self.calls = 0
        self.thinking = True
        self.drafts = []       # this agent's drafts, in the order proposed
        self.seed = None       # (rtl, failures) handed on by the last agent

    def _ask(self, p):
        if self.backend != "claude-cli":
            return CALLERS[self.backend](p, self.model)
        try:
            return call_claude_cli(p, self.model, thinking=self.thinking)
        except StillThinking:
            # Still thinking at the cap: ask the same thing with thinking
            # off, and keep it off for the rest of this block, so no later
            # draft spends the cap again before it answers.
            if not self.thinking:
                raise
            self.thinking = False
            return call_claude_cli(p, self.model, thinking=False)

    def _best_draft(self, history):
        """The draft to go back to when the most recent one failed
        simulation or synthesis after an earlier one had passed both: the
        latest such draft of this agent's, else the one it was handed.
        Returns (rtl, that draft's failure records, what to call it).

        Traced on the attention head: Sonnet's fourth draft passed all 236
        checks and synthesis and missed timing by 1.18 ns, and every draft
        after it, each an edit of the one before, broke the simulation."""
        if not self.drafts:
            return None
        fails = {}
        for h in history:
            fails.setdefault(h.get("iteration"), set()).add(h.get("stage"))
        last = len(self.drafts)
        if not fails.get(last, set()) & {"sim", "synth"}:
            return None
        for it in range(last - 1, 0, -1):
            st = fails.get(it, set())
            if st and not st & {"sim", "synth"}:
                return (self.drafts[it - 1],
                        [h for h in history if h.get("iteration") == it],
                        "Your draft %d" % it)
        if self.seed and self.seed[2]:
            return self.seed[0], self.seed[1], "The draft you were handed"
        return None

    def best_of(self, iterations):
        """What to hand the next agent in the chain, from the flow's
        per-iteration records: (rtl, its failure records, whether it passed
        simulation and synthesis). The draft that passed both with the least
        negative slack, else the last draft; None with no drafts.

        Traced on the attention head: Opus was handed Sonnet's last draft,
        which broke the simulation, with none of its feedback, and spent
        three drafts getting back to where Sonnet's fourth had been."""
        best = None
        for rec in iterations:
            i = rec.get("iteration") or 0
            if not 1 <= i <= len(self.drafts):
                continue
            if (rec.get("sim") or {}).get("status") != "pass" or \
                    (rec.get("synth") or {}).get("status") != "pass":
                continue
            slack = (rec.get("timing") or {}).get("worst_slack_ns")
            key = slack if slack is not None else float("-inf")
            if best is None or key >= best[0]:
                best = (key, i, rec)
        passed = best is not None
        if not passed:
            done = [r for r in iterations
                    if 1 <= (r.get("iteration") or 0) <= len(self.drafts)]
            if not done:
                return None
            rec = done[-1]
            i = rec["iteration"]
        else:
            _, i, rec = best
        fails = [dict(r, iteration=0) for r in
                 (rec.get(k) for k in ("sim", "synth", "timing", "fpga"))
                 if r and r.get("status") == "fail"]
        return self.drafts[i - 1], fails, passed

    def propose(self, spec, feedback_history):
        base, hist, note = self.last_rtl, feedback_history, None
        best = self._best_draft(feedback_history)
        if best:
            base, hist, who = best
            note = ("%s, shown above, passed every simulation check and "
                    "synthesis, and failed only what the feedback below "
                    "says. The drafts after it broke the simulation, so you "
                    "are revising it again: change only what that failure "
                    "needs, and keep its behaviour exactly as it is." % who)
        elif not self.drafts and self.seed:
            base, hist = self.seed[0], self.seed[1]
            note = ("The draft shown above was written by another model. It "
                    "passed every simulation check and synthesis, and failed "
                    "only what the feedback below says. Start from it: change "
                    "only what that failure needs, and keep its behaviour "
                    "exactly as it is." if self.seed[2] else
                    "The draft shown above was written by another model, and "
                    "the feedback below is how it failed. Revise it.")
        prompt = build_prompt(spec, hist, base, note)
        rtl, n = ask_for_module(self._ask, prompt, spec["top_module"])
        self.drafts.append(rtl)
        self.last_rtl = rtl
        self.calls += n
        return rtl, ["llm:%s@%s#%d" % (self.model, self.backend, self.calls)]


if __name__ == "__main__":
    b, m = pick_backend()
    print("backend: %s, model: %s" % (b, m))
