# The agent swarm

## Why more than one agent

The flow already worked with a single LLM writing Verilog: propose, then let
iverilog, yosys, OpenSTA and the Xilinx mapper judge the result, feed the
parsed failures back, repeat. That converges. The problem is that it converges
*sometimes*, and a capstone deliverable that works when you demo it and not
when the marker runs it is not a deliverable.

Three things go wrong with one agent:

1. It is handed raw tool output and has to both diagnose and rewrite in a
   single step. Diagnosis and synthesis are different jobs and the model does
   both worse when they are fused.
2. Nothing looks at the RTL before the tools do. A missing reset or a
   truncated accumulator costs a full iteration of simulation and synthesis to
   discover, when reading the code would have caught it in seconds.
3. Its only memory of what went wrong is a JSON blob of the last two failures.

The swarm splits the job into three roles that each do one thing.

## The roles

| Role | Sees | Produces | Never does |
|---|---|---|---|
| **debugger** | spec, failed RTL, parsed tool output | at most four lines naming the faulty signal or construct | write Verilog |
| **writer** | spec, its previous attempt, the diagnosis, any reviewer objections | the candidate RTL | run tools |
| **reviewer** | spec, the candidate RTL | `ACCEPT`, or at most four concrete defects | comment on style |

The debugger only runs when there is tool feedback to read, so the first
proposal of a run costs one call, not three. The reviewer runs before any tool
does, which is the point: catching a width mismatch by reading costs one call,
catching it by simulating costs a full iteration.

## What did not change

The orchestrator. `SwarmAgent.propose(spec, feedback_history)` has the same
signature as `LLMAgent.propose` and `RuleBasedAgent.propose`, so
`chiplet_flow.py` is agnostic:

```
python3 chiplet_flow.py --agent rules     # deterministic, free, the baseline
python3 chiplet_flow.py --agent llm       # one model, one role
python3 chiplet_flow.py --agent swarm     # three roles
```

The tools remain the only authority. The swarm decides what to hand over; it
does not decide whether the design is correct. A reviewer that accepts broken
RTL loses an iteration and nothing else, because simulation still runs.

## Failure isolation

Each role is wrapped so it can die alone. If the reviewer throws, the
candidate goes to the tools unreviewed. If the debugger throws, the writer
works from the raw feedback as the single-agent path always did. This is
deliberate and it is tested: **a broken reviewer must never be able to block a
design the tools would have accepted.** An advisory role that can veto is
worse than no advisory role.

`parse_review` follows the same rule. Empty output, whitespace, a one-word
reply, anything it cannot turn into concrete objections is read as acceptance.
The reviewer has to earn a rejection by producing something the writer can act
on.

## Measuring consistency

`bench.py` runs the same job N times with a fresh agent each time and reports
what actually happened:

```
python3 bench.py --agent swarm --runs 5
python3 bench.py --agent llm   --runs 5     # the control
python3 bench.py --agent rules --runs 2     # free smoke test of the harness
```

It reports the convergence rate, the iteration count, the model calls broken
down by role, how many runs passed every gate on the first attempt, and which
gates forced a retry. It writes to its own job files so the committed profiles
are never disturbed.

The metric that matters is not the mean. It is the convergence rate and the
first-pass-clean count, because the failure mode being designed against is
variance, not average quality.

## Measuring the DV, not just the RTL

Convergence means the RTL passed the testbench. That is only worth something
if the testbench could have failed, and the testbench is generated too, so
nobody has checked it either. `dv.py` checks it by mutation: take RTL that
already passes, inject a known defect, and see whether the testbench notices.

```
python3 dv.py --rtl build/mac.v --tb tb_mac.v
python3 dv.py --rtl build/crc.v --tb tb_crc.v
```

Eight operators, each a defect class that actually shows up in generated
RTL: a truncated accumulator, a subtraction where an addition belongs, a
dead reset branch, a clear that never fires, an ungated accumulate, a block
clocked on the wrong edge, an off-by-one product, and non-blocking
assignments turned blocking.

Results are split into buckets that mean different things:

- **killed** is the testbench working.
- **compile** means iverilog rejected it. Real, but the compiler caught it,
  not the DV, so it is excluded from the score.
- **equivalent** means yosys proved the mutant behaves identically to the
  original, so no testbench could ever kill it. Also excluded. A survivor is
  only called a hole after that proof is attempted and fails, because
  counting an unkillable mutant against the testbench would understate it.
- **SURVIVED** is a defect the flow would sign off on.

This found two real holes in the endpoint testbench, which scored 4/7:

| Survivor | Why it survived |
|---|---|
| `reset_dead` | every test opened with `start_frame`, whose `clear` reinitialises the same register reset does, so nothing ever depended on reset working |
| `clock_edge_flipped` | `crc_out` was only checked at end of frame, where wrong-edge clocking is just a half-cycle shift and observes identically |

Two tests in `specgen.py` close them. `reset_init` drives a word straight
after reset with no intervening `clear`, so the state has to come from reset
alone. `edge_discipline` checks the outputs one time unit after the falling
edge, where they must not have moved yet, and again after the rising edge,
where they must have. Both blocks now kill every behaviour-changing mutant.

| Block | Score | Excluded |
|---|---|---|
| chiplet (MAC) | 7/7 | 1 operator not applicable |
| endpoint (CRC32) | 6/6 | 1 not applicable, 1 proven equivalent |

The third endpoint survivor, `nonblocking_to_blocking`, is not a hole. The
CRC has one always block and `state` and `crc_out` both read `nxt` rather
than each other, so assignment order cannot matter. Yosys proves it: 65 of
65 equivalence cells, equivalence successfully proven. It is killed in the
MAC, where the pipeline does have cross-stage ordering.

`tests.py` runs the whole mutation sweep as a regression guard, so the DV
cannot quietly weaken later.

Results: see RESULTS.md.
