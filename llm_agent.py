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
  exactly as listed in the spec. No other modules, no includes.
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


def condense_feedback(history):
    """Bound the prompt: the most recent failure records, each trimmed to the
    fields an engineer would actually read."""
    out = []
    for fb in history[-MAX_FEEDBACK:]:
        rec = {"iteration": fb.get("iteration"), "stage": fb.get("stage"),
               "status": fb.get("status")}
        if fb.get("phase"):
            rec["phase"] = fb["phase"]
        if fb.get("errors"):
            rec["tool_errors"] = fb["errors"][:MAX_ERR_LINES]
        if fb.get("mismatches"):
            rec["testbench_mismatches"] = fb["mismatches"][:MAX_MISMATCHES]
        out.append(rec)
    return out


def build_prompt(spec, history, last_rtl):
    parts = [RULES.format(top=spec["top_module"]),
             "", "SPEC:", json.dumps(spec, indent=2)]
    if last_rtl:
        parts += ["", "YOUR PREVIOUS ATTEMPT (it failed, revise it):",
                  last_rtl]
    if history:
        parts += ["", "TOOL FEEDBACK, most recent last (make these pass):",
                  json.dumps(condense_feedback(history), indent=2)]
    return "\n".join(parts)


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
CLI_TIMEOUT_S = 600
CLI_ATTEMPTS = 3


def call_claude_cli(prompt, model, attempts=CLI_ATTEMPTS):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            r = subprocess.run(["claude", "-p", "--model", model],
                               input=prompt, capture_output=True, text=True,
                               timeout=CLI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            last = "timed out after %ds" % CLI_TIMEOUT_S
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
            time.sleep(min(30, 5 * 2 ** (attempt - 1)))
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

    def propose(self, spec, feedback_history):
        prompt = build_prompt(spec, feedback_history, self.last_rtl)
        raw = CALLERS[self.backend](prompt, self.model)
        rtl = extract_verilog(raw)
        self.last_rtl = rtl
        self.calls += 1
        return rtl, ["llm:%s@%s#%d" % (self.model, self.backend, self.calls)]


if __name__ == "__main__":
    b, m = pick_backend()
    print("backend: %s, model: %s" % (b, m))
