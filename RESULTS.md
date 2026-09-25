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
(`python3 bench.py --agent <kind> --runs 5`).

| agent | converged | iterations | model calls | cell spread |
|---|---|---|---|---|
| rules | 2/2 | 3 | n/a | 0% |
| solo LLM | 5/5 | 1 | 1 | 0% |
| swarm, all roles always on | 5/5 | 1 median | 2 median | 5% |

The always-on swarm is not free and did not pay for itself: the reviewer
accepted every first draft the tools then passed. That measurement is why
the roles now escalate, a lone writer first and the other roles only after
the tools reject something.

An earlier swarm arm scored 4/5. That run used a build in which the writer
was re-prompted with the previous iteration's RTL rather than the draft the
reviewer had objected to, so it was asked to fix code it could not see.
With that fixed the same configuration scores 5/5, and the 4/5 is recorded
here only so the number is not quoted as a property of the design.

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
