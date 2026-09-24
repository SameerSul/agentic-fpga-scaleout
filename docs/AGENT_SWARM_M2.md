# Milestone: spec to bitstream, autonomously

Owner: Sameer. Window: the eight working days after the September 30 presentation.

## What this milestone is

Hand the system a block specification. An LLM writes the RTL. A second agent writes the
verification. A third reviews and debugs when a tool rejects the work. Four existing gates plus
a new fifth gate sign it off. A loadable bitstream comes out the other end and runs on a real
board.

That is **spec to completion for one block**. It is deliberately not a complete accelerator.

## What this milestone is not

A full inference accelerator needs a systolic array, softmax, layernorm, requantization, an
on-chip buffer with a DMA controller, and top level integration. That is an October to December
arc, not a two week one. Promising it inside this window is how the capstone goes wrong.

## Where we start from

Already working:

- Four signoff gates: simulation (Icarus), synthesis (yosys), timing (OpenSTA), FPGA technology
  mapping (yosys synth_xilinx)
- A full three iteration convergence runs in **2.7 seconds**
- A real LLM backend behind the same `propose()` interface, with Anthropic API, Ollama and
  Claude CLI paths
- Specs derived rather than written: the chiplet from the model, the fabric endpoint from the
  link rate, including an architecture level retry when timing does not close
- 66 automated tests

Missing, and this milestone closes all four:

1. **No place and route and no bitstream.** No XDC constraints, no top level wrapper, no Vivado
   script. Nothing has ever been loaded onto a board.
2. **LLM iteration is unproven.** Haiku one-shotted both existing blocks, so no language model
   has ever actually read a tool failure and fixed its own RTL. Only the rule based agent has.
3. **One agent, not a swarm.** A single `propose(spec, feedback)` function.
4. **Testbenches are templated**, written by hand in `specgen.py` rather than by an agent.

## The constraint that shapes the design

The current loop costs 2.7 seconds. A Vivado place and route on a Zynq costs 10 to 30 minutes.
That is roughly a **500x slowdown per iteration**, so an agent cannot iterate against place and
route the way it iterates against simulation.

The loop therefore becomes two tier:

- **Fast tier**, unchanged: simulate, synthesize, time, map. Agents iterate here freely.
- **Slow tier**, new: place and route runs **once**, only after every fast gate passes. A routing
  failure re-enters the fast loop carrying post route timing as feedback, and the agent gets at
  most two attempts before a human looks at it.

This is how real teams work and the orchestrator is already structured for it, so gate five is an
extension rather than a rewrite.

## Hard dependency

**Vivado cannot run on Sameer's machine.** It is Windows and Linux only and he is on an ARM Mac.
Every bitstream has to be produced on Vaibhav's machine, a lab machine, or a Linux box.

This is a coordination dependency rather than a coding one, which makes it the single most likely
thing to slip. It has to be settled in the first two days or the milestone does not land.

Vivado's free tier covers Artix-7 and Zynq-7000, so both the Basys 3 and the Zybo Z7 are licensed
at no cost.

## Plan

| Days | Work | Depends on | Confidence |
|---|---|---|---|
| 1 to 2 | Vivado access settled. XDC constraints and a top level wrapper for the target board | Vaibhav, board confirmed | depends on Vaibhav |
| 2 to 3 | Place and route as gate five. Parse utilization and post route timing into the profile | above | high |
| 2 to 3 | Prove LLM iteration on a block chosen to fail on the first attempt | nothing | high |
| 2 to 3 | Second and third agent roles: RTL writer, verification writer, reviewer and debugger | nothing | high |
| 2 | An agent writes the testbench instead of the template filling it in | role split above | medium |

The three middle rows need no hardware, so they proceed in parallel with the Vivado work and are
not blocked if it slips.

## Done means

- [ ] A spec goes in and a `.bit` file comes out with no human in the loop
- [ ] The bitstream loads on a real board and its self test passes
- [ ] At least one block is fixed by an LLM reading a tool failure, with the transcript captured
- [ ] At least three distinct agent roles participate in producing a signed off block
- [ ] The testbench for that block was written by an agent, not a template
- [ ] Post route timing and utilization appear in the profile and feed the sizing layer
- [ ] The whole thing is reproducible from a single command and covered by tests

## Risks

| Risk | Mitigation |
|---|---|
| Vivado access slips past day two | Start it day one. The three software rows are not blocked by it |
| Place and route runtime makes iteration impractical | Two tier loop. P&R runs once, capped at two agent retries |
| The LLM one-shots the harder block too and iteration stays unproven | Pick a block with a known subtle failure, for example a CRC needing a pipelined or matrix form to close timing at 25 Gbps |
| Scope drifts toward the full accelerator | The non goals above are explicit. Anything beyond one block is October |
