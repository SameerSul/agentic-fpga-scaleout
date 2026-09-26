"""Multi-role agent swarm, behind the same propose() the orchestrator already uses.

One agent writing RTL is not a swarm. Three roles that check each other is:

    debugger  reads the parsed tool failures and says what is actually wrong,
              so the writer gets a diagnosis instead of a wall of log output
    writer    produces the Verilog from the spec, the diagnosis and its own
              previous attempt
    reviewer  reads the candidate against the spec before any tool runs and
              either accepts it or lists concrete defects, which sends the
              writer back for one more pass

The roles escalate rather than all running every time. A first attempt is a
lone writer, which costs exactly what a single agent costs; the debugger and
reviewer engage only after the tools have rejected something, which is the
only point at which either has anything real to work from. That ordering
came from measurement, not taste: with all three roles always on, the
reviewer accepted every first draft the tools went on to pass, so it doubled
the cost of the runs that were already fine.

The orchestrator is unchanged: it still calls propose(spec, feedback_history)
and still gates everything on the real tools. The swarm only decides what to
hand over; simulation, synthesis, timing and FPGA mapping remain the judge.

Every role degrades independently. If the reviewer errors or returns nonsense
the candidate goes through anyway, because a broken reviewer must never be
able to block a design the tools would have accepted."""
import re

from llm_agent import (pick_backend, CALLERS, extract_verilog,
                       condense_feedback, RULES)
import json

MAX_REVIEW_ROUNDS = 1      # internal revisions before handing to the tools
# The gates in the order the flow runs them. How far an attempt got is
# the only ranking signal available, and it is a real one: a design that
# fails timing is strictly further along than one that fails to compile.
STAGE_RANK = {"sim": 0, "synth": 1, "timing": 2, "fpga": 3}
# Failed iterations before the reviewer is switched on. It is off by
# default because it never changed an outcome on a first draft, but a
# writer that has failed repeatedly is the case it was built for.
REVIEW_AFTER = 2
REVIEW_ACCEPT = "ACCEPT"
# The reviewer is off by default. Across every bench run recorded in
# RESULTS.md it accepted the draft every single time, including on runs the
# tools then rejected, so it has not once changed an outcome and it costs a
# call on each retry. The role is kept because it is cheap to re-enable on a
# block hard enough to need it, but it is not on the measured path until it
# earns its place: SwarmAgent(use_reviewer=True).
USE_REVIEWER = False


def _fmt_feedback(history):
    return json.dumps(condense_feedback(history), indent=2)


def debugger_prompt(spec, history, last_rtl):
    return "\n".join([
        "You are a hardware debug engineer. Verilog failed its tools.",
        "Say what is wrong and where, in at most four lines. Name the signal",
        "or construct at fault. Do not write any Verilog and do not restate",
        "the log. If the cause is genuinely unclear, say so and name the one",
        "thing you would check first.",
        "", "SPEC:", json.dumps(spec, indent=2),
        "", "THE RTL THAT FAILED:", last_rtl or "(none)",
        "", "WHAT THE TOOLS REPORTED:", _fmt_feedback(history),
    ])


def writer_prompt(spec, history, last_rtl, diagnosis, objections,
                  tried=()):
    """Build the writer's prompt with the tool output as the authority.

    The ordering here is load bearing. An earlier version put the debugger
    above the tool feedback and labelled it "fix this", while demoting the
    tool output to "RAW TOOL FEEDBACK". That tells the writer to trust a
    hypothesis over ground truth, and when the hypothesis was wrong the
    writer chased it for every remaining iteration: measured over five runs
    on the signed spec, the swarm converged 3/5 where a single agent seeing
    only the tool output converged 5/5. The tools are the only authority.
    The diagnosis is a hint, it comes last, and it is labelled fallible.
    """
    parts = [RULES.format(top=spec["top_module"]),
             "", "SPEC:", json.dumps(spec, indent=2)]
    if last_rtl:
        parts += ["", "YOUR PREVIOUS ATTEMPT (it failed, revise it):",
                  last_rtl]
    if history:
        parts += ["", "TOOL FEEDBACK, most recent last (make these pass). "
                      "This is ground truth:", _fmt_feedback(history)]
    if tried and history:
        # Proposing a design the tools already rejected wastes a whole
        # iteration of simulation, synthesis, timing and mapping, and it
        # is the most common way a retry loop stalls.
        parts += ["", "You have already proposed %d design%s that the "
                      "tools rejected. Do not propose any of them again; "
                      "change the part the feedback above names."
                      % (len(tried), "" if len(tried) == 1 else "s")]
    if objections:
        parts += ["", "A REVIEWER RAISED THESE OBJECTIONS TO YOUR LAST "
                      "DRAFT (address them where they agree with the tool "
                      "feedback above):", objections]
    if diagnosis:
        parts += ["", "ONE ENGINEER'S HYPOTHESIS ABOUT THE CAUSE. It is a "
                      "hint, not a finding, and it may be wrong. Where it "
                      "conflicts with the tool feedback above, believe the "
                      "tools and ignore this:", diagnosis]
    return "\n".join(parts)


def reviewer_prompt(spec, rtl):
    return "\n".join([
        "You are reviewing Verilog against its specification, before any tool",
        "runs. Report only defects that would make a tool fail or make the",
        "design disagree with the spec: wrong port names, widths or",
        "directions; truncated arithmetic; a missing reset or clear path;",
        "latches; SystemVerilog syntax; behaviour contradicting the spec.",
        "",
        "Do not comment on style, naming, formatting or efficiency.",
        "If you find no such defect, reply with exactly: " + REVIEW_ACCEPT,
        "Otherwise list each defect on its own line, at most four lines.",
        "", "SPEC:", json.dumps(spec, indent=2),
        "", "RTL UNDER REVIEW:", rtl,
    ])


def parse_review(text):
    """Return (accepted, objections). A reviewer that returns nothing usable
    is treated as acceptance: a broken reviewer must not block a design the
    tools would have passed."""
    if not text:
        return True, ""
    t = text.strip()
    if not t:
        return True, ""
    head = t.splitlines()[0].strip().upper().strip(".!*` ")
    if head.startswith(REVIEW_ACCEPT) or REVIEW_ACCEPT in t.upper()[:40]:
        return True, ""
    # Drop any preamble and keep the concrete lines.
    lines = [l.strip(" -*\t") for l in t.splitlines() if l.strip()]
    lines = [l for l in lines if len(l) > 8][:4]
    if not lines:
        return True, ""
    return False, "\n".join(lines)


class SwarmAgent:
    """Writer, reviewer and debugger over one backend. Drop-in for LLMAgent."""

    def __init__(self, choice=None, review_rounds=MAX_REVIEW_ROUNDS,
                 use_reviewer=USE_REVIEWER, use_debugger=True,
                 escalate=True):
        self.backend, self.model = pick_backend(choice)
        self.review_rounds = review_rounds
        self.use_reviewer = use_reviewer
        self.use_debugger = use_debugger
        self.escalate = escalate
        self.last_rtl = None
        # The furthest any attempt has got, and the RTL that got there.
        # Handing the writer its most recent attempt compounds a
        # regression: if iteration three is worse than iteration two,
        # every later iteration starts from the worse one.
        self.best_rtl = None
        self.best_rank = -1
        self.tried = []
        self.calls = {"writer": 0, "reviewer": 0, "debugger": 0}
        self.log = []

    def _ask(self, role, prompt):
        self.calls[role] += 1
        return CALLERS[self.backend](prompt, self.model)

    def _rank(self, feedback_history):
        """How far the most recent attempt got, by the last stage the
        tools complained about."""
        if not feedback_history:
            return -1
        last_iter = max(f.get("iteration", 0) for f in feedback_history)
        stages = [STAGE_RANK.get(f.get("stage"), -1)
                  for f in feedback_history
                  if f.get("iteration", 0) == last_iter]
        return max(stages) if stages else -1

    def propose(self, spec, feedback_history):
        notes = []
        # Keep whichever attempt reached the furthest gate.
        rank = self._rank(feedback_history)
        if self.last_rtl is not None and rank > self.best_rank:
            self.best_rank, self.best_rtl = rank, self.last_rtl
        fails = len({f.get("iteration") for f in feedback_history})
        # Escalation. On the first attempt there is no tool feedback, so the
        # debugger has nothing to read and the reviewer is guessing at what
        # the tools will say. Measured over five runs each, the reviewer
        # accepted every first draft the tools then passed, so it doubled
        # the call count and bought nothing. The swarm therefore opens as a
        # single writer, exactly as cheap as one agent, and engages the
        # other roles only once the tools have actually rejected something.
        engaged = bool(feedback_history) or not self.escalate
        # After repeated failures the reviewer is worth its call: a
        # writer that has been wrong several times in a row is exactly
        # the case that reading the draft before simulating it helps.
        reviewing = (self.use_reviewer or fails >= REVIEW_AFTER) and engaged

        diagnosis = ""
        if feedback_history and self.use_debugger:
            try:
                diagnosis = self._ask(
                    "debugger",
                    debugger_prompt(spec, feedback_history, self.last_rtl)
                ).strip()
                notes.append("debugger")
                self.log.append(("debugger", diagnosis[:300]))
            except Exception as e:                      # a role may fail alone
                self.log.append(("debugger_error", str(e)[:200]))

        # prev is what the writer is shown as "your previous attempt". It
        # starts as the last iteration's RTL, but once the reviewer rejects a
        # draft it becomes that draft: objections are worthless to a writer
        # that cannot see the code they refer to.
        # Start from the best attempt, not the latest, and say which it
        # is so the writer is not told a design failed when it was the
        # furthest one to get through.
        base = self.best_rtl if (self.best_rank > rank and self.best_rtl)  \
            else self.last_rtl
        objections, rtl, prev = "", None, base
        for attempt in range(self.review_rounds + 1):
            rtl = extract_verilog(self._ask(
                "writer",
                writer_prompt(spec, feedback_history, prev,
                              diagnosis, objections, self.tried)))
            notes.append("writer" if attempt == 0 else "writer:revised")
            if not reviewing or attempt == self.review_rounds:
                break
            try:
                ok, objections = parse_review(self._ask(
                    "reviewer", reviewer_prompt(spec, rtl)))
            except Exception as e:
                self.log.append(("reviewer_error", str(e)[:200]))
                break
            self.log.append(("reviewer", "accept" if ok else objections[:300]))
            if ok:
                notes.append("reviewer:accept")
                break
            notes.append("reviewer:reject")
            prev = rtl

        if rtl and rtl not in self.tried:
            self.tried.append(rtl)
        self.last_rtl = rtl
        n = sum(self.calls.values())
        return rtl, ["swarm:%s@%s" % (self.model, self.backend),
                     "+".join(notes), "calls=%d" % n]


if __name__ == "__main__":
    a = SwarmAgent()
    print("backend %s, model %s" % (a.backend, a.model))
