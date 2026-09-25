# What is verified, and what is not

Last run 2026-09-25. Every number here came from a command in this repo, and
every command is named so it can be re-run.

## Short answer

The flow reliably turns a model spec into signed-off RTL for the two blocks
it generates, and the generated hardware provably computes the arithmetic
the model needs. It does not yet host a local LLM the way Architect Labs
does, because the blocks it generates are a MAC unit and a fabric endpoint,
not a whole inference engine, and nothing has been put on a board. The gap
is listed at the bottom rather than glossed over.

## Verified

### The full suite

```
python3 tests.py            # 120 tests, all passing
```

### Spec to RTL, across the spec space

`python3 sweep.py` drives fifteen cases through every stage: derivation,
RTL, simulation, synthesis, timing closure, FPGA mapping, the profile
fields the sizing model consumes, and a mutation sweep of the testbench
generated at that width. All fifteen clean.

| case | cyc/unit | fmax | cells | DV |
|---|---|---|---|---|
| mac gpt2_124m, int8, acc28 | 1.00 | 177 MHz | 1408 | 8/8 |
| mac int4 weights, acc24 | 1.00 | 187 MHz | 1383 | 8/8 |
| mac int16, acc46 | 1.00 | 104 MHz | 4570 | 6/6 |
| mac tiny model, acc24 | 1.00 | 187 MHz | 1383 | 8/8 |
| mac qwen3_0p6b, acc28 | 1.00 | 177 MHz | 1408 | 8/8 |
| mac w8/a16, acc37 | 1.00 | 115 MHz | 4400 | 6/6 |
| crc 1G, 1 B/cyc | 1.00 | 476 MHz | 774 | 6/6 |
| crc 10G, 16 B/cyc ripple | 0.06 | 85 MHz | 7333 | 6/6 |
| crc 25G, 16 B/cyc flat | 0.06 | 385 MHz | 10527 | 5/5 |
| crc 100G, 64 B/cyc flat | 0.02 | 325 MHz | 34682 | 5/5 |

Timing is OpenSTA, and the sweep rejects any case whose number came from
the gate-depth proxy instead, because a proxy is an estimate and not
closure.

### The generated hardware computes the model's arithmetic

```
python3 inference.py
```

Every other check in the repo verifies the RTL against a testbench written
from the same spec, which is circular with respect to the only question
that matters for hosting a model. `inference.py` breaks the circle: real
quantized transformer dot products are pushed through the actual generated
Verilog in iverilog and compared against a bit-accurate model with signed
operands and a wrapping accumulator.

- 20 real dot products, reduction depths 64 and 256, agree bit for bit.
- The validated model then runs a full-size block, d_model 768 and d_ff
  3072: zero accumulator overflows.
- The accumulator rule is confirmed rather than asserted: 28 bits derived,
  27 needed for the worst case over 3072 terms.

### Agent consistency

Five runs each, fresh agent per run, same job and same gates
(`python3 bench.py --agent <kind> --runs 5`). Two spec generations are
reported separately because the signed spec is the harder problem and the
earlier numbers were measured before it existed.

On the current signed spec:

| agent | converged | median calls | cell spread | first pass clean |
|---|---|---|---|---|
| solo LLM | 5/5 | 1 | 3% | 3/5 |
| swarm, before the prompt fix | **3/5** | 1 | 0% | 3/3 |
| swarm, after the prompt fix | 5/5 | 1 | 2% | 4/5 |

On the earlier unsigned spec, which is where escalation was measured:

| agent | converged | median calls | cell spread |
|---|---|---|---|
| rules (deterministic, free) | 2/2 | n/a | 0% |
| solo LLM | 5/5 | 1 | 0% |
| swarm, all roles always on | 5/5 | 2 | 5% |
| swarm, escalating | 5/5 | 1 | 1% |

Three measurements changed the design, and each is worth more than the
final number:

1. **The always-on swarm did not pay for itself.** The reviewer accepted
   every first draft the tools then passed, so it doubled the cost of runs
   that were already fine. Moving the debugger and reviewer behind a tool
   failure took the median from two calls to one and the cell spread from
   5% to 1%.
2. **The swarm was worse than one agent on the hard path.** 3/5 against
   5/5, with two runs burning sixteen calls and never passing simulation.
   The cause was prompt ordering: the diagnosis sat above the tool output
   labelled "fix this" while the tool output was demoted to "RAW", so a
   wrong hypothesis cost every remaining iteration. With the tools restored
   as the authority the same configuration scores 5/5, and the one hard run
   recovered in five iterations using the debugger, which is the case the
   swarm exists for.
3. **The reviewer has never changed an outcome.** It replied ACCEPT to
   every draft it was ever shown, including drafts the tools then rejected.
   It is off by default.

An earlier unsigned-spec swarm arm scored 4/5. That build re-prompted the
writer with the previous iteration's RTL rather than the draft the reviewer
had objected to, so it was asked to fix code it could not see. Fixed, the
same configuration scores 5/5. It is recorded so the 4/5 is not quoted as a
property of the design.

The honest summary is that on these two blocks a single agent is already
enough, and the swarm's value is not demonstrated so much as its cost is
now neutral. It converges as often, at the same median cost, and has a
recovery path a single agent does not. That is insurance, not a win, and it
should be described that way until a block hard enough to need it shows up.

### Defects the flow used to sign off on, and now cannot

Mutation testing (`python3 dv.py`) injects a known defect into RTL that
already passes and checks the testbench notices. Survivors are put to yosys
for an equivalence proof before being called holes, so an unkillable mutant
is never counted against the testbench.

| defect | was | now |
|---|---|---|
| unsigned datapath | passed every gate | fails `signed_operands` |
| dead reset branch | passed the endpoint DV | fails `reset_init` |
| wrong clock edge | passed the endpoint DV | fails `edge_discipline` |
| 25G/100G endpoint | failed timing at every option | closes at 385 and 325 MHz |
| signed netlist | silently fell back to the timing proxy | reads in OpenSTA |

## Not verified, and not claimed

These are the distance between this repo and a local LLM host.

1. **Nothing has run on a board.** There is no bitstream. Vivado does not
   run on an ARM Mac, so place and route, real fmax and real resource use
   are all unmeasured. Every timing number here is from OpenSTA against a
   generic standard cell library, not a device.
2. **The generated blocks are not an inference engine.** The flow generates
   a multiply-accumulate unit and a CRC32 fabric endpoint. It does not
   generate the matrix engine, the attention datapath, softmax, the
   normalisation, the requantisation between layers, the weight streaming
   controller or the memory subsystem. `inference.py` models those in
   Python and puts only the dot products through real hardware.
3. **No real weights.** The quantized layer uses the model's shapes,
   reduction depths and quantization scheme, with random weights. No
   Qwen3-0.6B checkpoint is loaded and no tokenizer exists, so the repo has
   never produced a token.
4. **Throughput is predicted, not measured.** tokens/s comes from a sizing
   model checked against a fabric simulation, agreeing within 15% and at
   ratio 1.00 on the current configuration. Both are models. Neither is a
   board.

Items 1 and 3 are the ones that would let this claim what Architect Labs
demonstrated. Item 2 is the largest amount of remaining design work.
