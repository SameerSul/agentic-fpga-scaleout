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

## Escalation, and why it was not the first design

The roles do not all run every time. A first attempt is a lone writer. The
debugger and the reviewer engage only after the tools have rejected
something, which is the first moment either has anything real to work from:
the debugger has no failure to read, and the reviewer is otherwise guessing
at what the tools are about to say.

This was not the original design and it is not a matter of taste. With all
three roles always on, measured over five runs, the reviewer accepted every
single first draft that the tools then passed. It doubled the cost of every
run that was already fine and changed no outcome. Escalation makes the swarm
cost exactly what one agent costs on work that was never going to fail,
which is most work, and spend the extra calls only where a single agent
would have been stuck.

The reviewer still earns its place, but on the retry path: catching a width
mismatch by reading costs one call, catching it by simulating costs a whole
iteration of sim, synthesis, timing and mapping.

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

## Does it work for specs nobody tuned it for

A flow that only works on the spec it was written against is a demo.
`sweep.py` drives many different specs end to end and checks every stage,
not just the last one: derivation, RTL, simulation, synthesis, timing
closure, FPGA mapping, the profile fields the sizing model consumes, and a
full mutation sweep of the testbench generated at that width.

```
python3 sweep.py                 (everything)
python3 sweep.py --only chiplet
python3 sweep.py --skip-dv       (gates only, much faster)
```

It runs on the rule-based agent, so it is free, deterministic and safe to
put in CI. Six model specs, from int4 weights to a 16-bit datapath to a
model small enough that the guard term collapses, and four link rates from
1G to 100G. Fifteen cases, all clean, every gate, and the generated DV kills
every behaviour-changing mutant at every width.

## The endpoint could not reach 25G, and why that was an RTL bug

The first endpoint sweep failed at 25G and 100G, at both datapath options.
That looked like a physics limit until the required combinational delay was
plotted against the datapath width:

| bytes/cycle | delay needed | clock budget | slack |
|---|---|---|---|
| 8 | 6.42 ns | 6.40 ns | -0.02 |
| 16 | 11.70 ns | 12.80 ns | +1.10 |
| 32 | 21.78 ns | 10.24 ns | -11.54 |
| 64 | 42.18 ns | 5.12 ns | -37.06 |
| 128 | 83.90 ns | 10.24 ns | -73.66 |

Delay is **linear** in the width, which is the signature of a serial ripple.
The generated RTL wrote the CRC as a byte loop containing a bit loop, so 32
bytes per cycle is 256 shift-XOR steps in one combinational path.

CRC32 does not need that. Its next-state function is linear over GF(2), so
the next state is the XOR of a fixed set of current-state and input bits:

    state' = A . state  XOR  B . data

`specgen.crc_matrix` derives A and B from the same polynomial the golden
vectors use, by pushing basis vectors through the step function. It asserts
`step(0, 0) == 0` first, because a constant term would mean the function is
not linear and the whole construction invalid. Written this way each output
bit is one XOR reduction and the depth is logarithmic in the width:

| bytes/cycle | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| max fan-in | 14 | 52 | 89 | 157 | 288 | 554 |
| XOR depth | 4 | 6 | 7 | 8 | 9 | 10 |

So the endpoint now has an architecture axis, not just a width axis. Each
rate offers every width as a ripple first, because it is much smaller, and
as a flat XOR tree second. The search escalates only when the cheap form
misses its clock, which is the call a human designer makes at that point.

| rate | before | after |
|---|---|---|
| 1G | 1 B/cyc ripple, fmax 476 MHz | unchanged |
| 10G | 16 B/cyc ripple, fmax 85 MHz | unchanged |
| 25G | **failed both options** | 16 B/cyc flat, fmax 385 MHz |
| 100G | **failed both options** | 64 B/cyc flat, fmax 325 MHz |

Both now close on the *narrow* datapath, which is the better answer anyway:
widening was only ever a way to buy clock period, and it costs bandwidth per
pin. The 100G endpoint costs 34,682 cells against 774 for 1G, which is the
honest price of the flat form and the reason the ripple is still tried first.

Results: see RESULTS.md.
