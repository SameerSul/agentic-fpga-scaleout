# What is verified, and what is not

Last run 2026-09-25. Every number here came from a command in this repo, and
every command is named so it can be re-run.

## Short answer

The flow turns a model spec into signed-off RTL for three generated blocks,
and a trained language model now decodes through those blocks' exact
arithmetic and emits text. It still does not host a local LLM the way
Architect Labs does: nothing has been put on a board, and the generated
blocks are the arithmetic of an inference engine rather than the whole of
one. The remaining gap is listed at the bottom rather than glossed over.

## Verified

### The full suite

```
python3 tests.py            # 201 tests, or 199 without OpenSTA
```

### Spec to RTL, across the spec space

`python3 sweep.py` drives sixty three cases through every stage: derivation,
RTL, simulation, synthesis, timing closure, FPGA mapping, the profile
fields the sizing model consumes, and a mutation sweep of the testbench
generated at that width. All sixty three clean.

| case | cyc/unit | fmax | cells | DV |
|---|---|---|---|---|
| mac gpt2_124m, int8, acc28 | 1.00 | 177 MHz | 1408 | 8/8 |
| exp Q4.8 to Q0.15 | 4.00 | 172 MHz | 2347 | 4/4 |
| recip 26b to 17b | 4.00 | 223 MHz | 1562 | 4/4 |
| rsqrt 28b to 17b | 4.00 | 174 MHz | 1351 | 4/4 |
| matvec sequencer | 68.0 | 169 MHz | 1397 | 4/4 |
| wmem tile + loader | 1024 | 333 MHz | 61712 | 5/5 |
| softmax sequencer | 64.0 | 106 MHz | 1546 | 6/6 |
| mlp layer | 78.0 | 102 MHz | 9275 | 3/3 |
| requant acc28 to 8 | 7.00 | 102 MHz | 9274 | 5/5 |
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

## The model actually runs

```
python3 train_tiny.py      # one off: trains and writes tiny_llm.json
python3 generate.py
```

A character-level transformer, trained from scratch in this repo with a
hand-written reverse-mode autodiff (`autodiff.py`, gradients checked
against finite differences to 1e-11), then decoded through the generated
hardware's arithmetic. Every dot product goes through the bit-accurate MAC
model and every requantization through the bit-accurate requantizer model,
and both are checked against their RTL in iverilog on vectors taken from
that very decode.

    prompt:    'the agent'
    generated: ' writes the rtools decidecid'
    float ref: ' writes the rtoools decideci'

    MAC:       12/12 dot products bit exact
    requant:   12/12 requantizations bit exact
    exp:       12/12 exponentials bit exact
    recip:     12/12 reciprocals bit exact

Two things are worth reading off that.

The text is what the hardware would emit, not what a float model emits.
Nothing in the decode's arithmetic is unverified against RTL.

And int8 costs this model nothing measurable, once the pipeline is
correct:

| measure | agreement with float |
|---|---|
| teacher-forced next token | **100%** (48/48) |
| free-running decode | **100%** |

The checkpoint carries RMSNorm, which is what the target model family
normalises with. Adding it took training loss from 0.23 to 0.11 and the
free-running agreement to 100%, and it is what puts the inverse square
root on the decode path: before it, that block was verified on its own
and called by nothing, which is a weaker claim than it looks.

That is a correction, not a result. This file previously reported that
int8 cost the model 75% of its characters, and drew a lesson from it about
validating quantization at the target model size. The lesson was invented
to explain a bug. Activations carry a scale, and the residual adds were
summing two int8 vectors whose scales differed, which is not a rounding
loss but arithmetic in mismatched units. With scales tracked through the
network the quantized decode reproduces the float model exactly.

Two things are worth keeping from how that was found, since the wrong
number survived several rounds of reporting.

The first measurement was also the wrong measurement. Free-running
agreement counts every character after one divergence as a disagreement,
so it mostly measures how fast greedy decoding amplifies a single flip.
Teacher-forced agreement, both models given the same context, is the one
that is about arithmetic. The free-running figure at 28 tokens is still
68%, entirely from one late divergence where the hardware in fact produced
the cleaner text.

And a plausible explanation is not evidence. "A 16-dimensional model has
no redundancy to spare" was a comfortable story that fit the number, so it
went unchallenged. The test of it, training a wider model, was confounded
by undertraining (float accuracy 52%, top-2 margin 1.07, against 94% and
4.37 for the small one) and pointed the wrong way, which is what finally
forced a look at the pipeline itself. `generate.py` now prints the float
model's own accuracy and logit margin next to the agreement, so an
undertrained checkpoint cannot be mistaken for a quantization result
again.

## A real bitstream exists

```
python3 bitstream.py --block mac        # also requant, crc
```

yosys synth_ice40, then nextpnr-ice40 for place and route, then icepack.
This is real place and route against a real device timing model, not a
generic cell library.

| block | LUTs of 7680 | target | post-route fmax | bitstream verified |
|---|---|---|---|---|
| MAC chiplet | 214 (2%) | 100 MHz | **166.14 MHz** | 613 checks |
| CRC32 endpoint, 1 B/cyc | 90 (1%) | 125 MHz | **236.63 MHz** | 9 checks |
| exponential | 350 (4%) | 100 MHz | **101.73 MHz** | 153 checks |
| reciprocal | 184 (2%) | 100 MHz | **299.67 MHz** | 206 checks |
| inverse square root | 155 (2%) | 100 MHz | **256.81 MHz** | 212 checks |
| matvec sequencer | 231 (3%) | 100 MHz | **103.89 MHz** | 64 checks |
| weight memory + loader | 67 (1%) + 2 BRAM | 100 MHz | **218.53 MHz** | 20 checks |
| softmax sequencer | see note | 100 MHz | **106 MHz** (library) | 121 checks |
| requantizer | 1924 (25%) | 100 MHz | 97.88 MHz | 173 checks |

**The bitstream is verified, not just produced.** icepack emits the actual
configuration bits; `icebox_vlog` turns those bits back into logic, and
the original self-checking testbench runs against them. Everything
upstream can be right and the packed result still be wrong, and until now
nothing had looked at the artifact that would actually be loaded onto a
device. All three pass.

That check immediately earned its place by catching a fault in itself.
The endpoint bitstream was first reported as failing, producing
0xD202EF8D where 0xECBB4B55 was expected. 0xD202EF8D is CRC32 of a single
zero byte: the harness had built the 1 Gbps endpoint, one byte per cycle,
and compared it against the committed `tb_crc.v`, which is generated for
the 10 Gbps endpoint at sixteen. The design had computed exactly the right
answer for its own configuration. The testbench is now generated for the
spec being built rather than taken from a committed file.

The device is a Lattice iCE40 HX8K. That is **not** the part this project
is aimed at: the boards are Xilinx, Vivado does not run on an ARM Mac, and
nextpnr has no mainline Xilinx target, so the iCE40 is simply the only
family with an open place and route flow installable here. What this
proves is that the flow reaches a bitstream and that the generated RTL
survives real place and route. It does not prove anything about the
Artix-7 or Zynq parts the team owns.

Two things the device taught that the generic library could not.

**The library is optimistic, and not uniformly.** It gave the requantizer
102 MHz where silicon gives 97.9. The gate in the flow is therefore
slightly generous, which is why `tests.py` now checks the two stay within
2x of each other rather than trusting either alone.

**More pipelining is not always faster.** Splitting the 46-bit adds as
well, so every wide add is two stages, improved the generic library from
102 to 109 MHz and made the real device *worse*, 97.9 to 93.5. On a real
fabric this block is routing bound, not logic bound: more registers means
more routing pressure. Precomputing the saturation flags to shorten the
last stage was worse still, 97.9 to 58.8, because it duplicates the barrel
shifter three times. Both were tried, measured, and reverted, and the
reasons are in the source so they are not tried again.

The requantizer missing 100 MHz by 2% on an iCE40 is not treated as a
design failure here. It is a 25%-utilised block on a small LUT4 fabric
with no carry chains, and it runs once per dot product, so once per
reduction depth, which is at least 64 MACs and usually thousands. It is
recorded rather than engineered around, because the device it misses on is
not a device this project uses.

## The first block that sequences another

`matvec` walks a weight matrix column by column, drives the MAC unit,
waits out that unit's pipeline and flags each finished column. It is the
first generated block whose correctness depends on another generated
block's latency: the drain count comes from the MAC's own
pipeline_stages, so a deeper MAC changes this block rather than
silently desynchronising from it, and a test pins that.

Its testbench instantiates the real MAC rather than a model of one, so
the flow gained a notion of an integration job: a job may name extra
sources, which are compiled into both the simulation and every mutant.

It drives a synchronous memory, because block RAM registers its read.
An asynchronous model would hide the bug this block is most prone to,
so the second seeded bug is driving valid in the same cycle as the
address, which multiplies whatever the memory held before. The two bugs
fail at different columns, and that is enough to tell them apart: a
wrong column zero means the data was not there yet, a correct column
zero with a wrong one after it means the accumulator was never cleared.
The agent uses exactly that inference and converges in three
iterations, fixing one bug, uncovering the second and fixing that. It
is the only multi-bug convergence in the repo.

Real place and route then caught something the generic library hid.
Writing the column base as col*depth puts a 12 by 12 multiplier in the
control path. The library reported 134 MHz and silicon came back at
69.6 against a 100 MHz target. The base only ever advances by depth, so
a running accumulate replaces it: 103.89 MHz, and the device LUT count
went from 559 to 231.

Two further things the sweep found, both in the testbench rather than
the design. The matrix was 8 by 4, which leaves the top half of the derived
address register always zero, so a mutation that halved it survived
every vector. Enlarging the matrix fixed it for the 24-bit case and not
for the 28-bit one, because a fixed size cannot track a derived width:
the matrix is now sized from the address width, and the memory is filled
by a formula so the file stays readable at four figures of entries.

## Blocks that contain other blocks

Three of the nine generated blocks are composite. The sequencer drives
the MAC, the weight memory is checked feeding both, and softmax
instantiates the exponential and the reciprocal outright. That last one
made the flow itself learn something: synthesis, FPGA mapping and the
bitstream stage all read only the top file, so a block with a hierarchy
below it failed to elaborate. All three stages now take the extra
sources, and yosys prunes whatever the top does not reach.

Every one of those composites is checked against the real blocks it
uses rather than a model of them, and all four bitstreams are verified
from their packed bits: MAC 613 checks, softmax 121, sequencer and MAC
64, memory and sequencer and MAC 20.

Softmax took four timing bugs to get right and every one was the same
family: a transition that pre-issued an address and then reset the
counter, so element zero was read twice; a valid flag one cycle short
of the two-cycle read latency, so every pass compared stale data on its
first element; a registered valid gated on the data-valid flag, which
asserted it a cycle late so the exponential consumed the next element;
and the multiply, variable shift and saturate in one stage, a tenth of
a nanosecond over. Each of them passes a one-element row and fails on
two, which is why the testbench sweeps rows of 1, 2, 5, 16 and 64.

And one thing only a generated testbench can do. The normalising
product is shifted right about seventeen bits, so an off-by-one in it
changes a weight about once in a hundred thousand random rows, and that
mutation survived every vector. The generator now searches for a row
that lands on the carry boundary and embeds it, which took under a
second.

## Which blocks an LLM can write, measured

The agent interface takes either a deterministic rule-based agent or a
real LLM. Both face identical gates. Running it at every block, for
Qwen2.5-0.5B:

| block | LLM result | iterations |
|---|---|---|
| MAC chiplet | converged | 1 |
| matmul sequencer | converged | 1 |
| weight memory | converged | 15 |
| exponential | converged | 6 |
| reciprocal | converged, 4 of 4 runs | 1, 2, 1, 2 |
| inverse square root | converged, 4 of 4 runs | 3, 1, 2, 2 |
| requantizer, softmax, MLP layer | failed | 23 to 26 calls |

The three table-driven units were first recorded here as blocks an LLM
cannot write, with the explanation that exact fixed-point arithmetic is
a poor fit for a model. That explanation was wrong, and the measurement
behind it was measuring the specification rather than the agent.

The testbench holds these units to bit-exact agreement with a Python
model, but the spec described the arithmetic in prose that left the
deciding details to guesswork. It never said the log2(e) constant is
scaled by 2**16, or that the integer part is an arithmetic shift that
floors toward minus infinity. Worse, the reciprocal indexes its table
with the mantissa bits below the implicit leading one, while the inverse
square root keeps the leading one in the index because its normalised
range spans a factor of four. Both specs said only "the table entry the
normalised mantissa selects". No engineer could have guessed which, and
every draft failed on the first vector.

Rewriting the three behaviour sections at the level of a datasheet,
with the exact operations and bit ranges, changed the result from zero
to every run converging. A test now walks the arithmetic the prose
describes and checks it against the golden model on all six model
variants, so the two cannot drift apart again.

The exponential run is also the clearest instance of the loop working as
intended. Its first draft multiplied a signed input by a bare decimal
literal, which Verilog treats as unsigned, so the whole product was
unsigned and the shift logical. It diverged at the very first vector,
x = -1, and the model corrected that and the errors after it from tool
feedback alone over six iterations.

One recorded failure was not a failure of the model at all. An earlier
exponential run, and the inverse square root row above as first
recorded, ended on a tool error: a single CLI call sat at zero CPU
until the ten minute timeout and aborted the block. Transport faults are
now retried with backoff, and authentication faults still fail at once,
since retrying them cannot give a different answer.

The requantizer, softmax and MLP rows are from before the specification
fix and have not been rerun. Given what the table units turned out to
be, the first thing to check there is whether their specs are
implementable as written, before concluding anything about the agent.

The deterministic agent is still what generated the committed RTL, and
it is still the right default for anyone who wants hardware reproducibly
and offline. The LLM path is no longer only a demonstration of the loop.

## Not verified, and not claimed

These are the distance between this repo and a local LLM host.

1. **Nothing has run on hardware.** The bitstreams are placed, routed,
   packed and functionally verified, but for an iCE40 HX8K, and nobody
   here owns that board. No device has been configured and clocked, so
   every timing number is from a vendor model rather than from silicon in
   operation. The Xilinx parts the team owns are still unreachable: Vivado
   does not run on an ARM Mac and nextpnr has no mainline Xilinx target.
   This is the only remaining item that work on this machine cannot close;
   it needs a board (a ~50 CAD iCE40 runs these bitstreams unmodified) or
   an x86 machine with Vivado for the Zynq and Artix parts.
2. **The generated blocks are the arithmetic, not the whole engine.** The
   flow generates the multiply-accumulate unit, the requantizer between
   matmuls, the exponential and the reciprocal that softmax needs, the
   inverse square root that RMSNorm needs, and the CRC32 fabric
   endpoint, the weight-streaming sequencer that drives the MAC through
   a matrix, the weight tile and its loader, softmax as a single
   sequenced block, and an MLP layer that drives two matmuls in order
   and routes the activations between them. Softmax is hardware apart from accumulating the
   sum, and so is the transcendental part of RMSNorm. It does not
   generate attention itself, though its parts now exist. The weight
   tile holds 1024 entries and the activation bank 64, so a real matrix
   needs tiling logic that does not exist yet. In `generate.py` those run on the host, and the
   output says so each run.
3. **The model is small and its weights are its own.** Qwen3-0.6B is not
   loaded; there is no numeric stack here to load it with and no network
   dependency wanted in a capstone repo. The committed checkpoint is a
   real trained transformer with a real tokenizer, but it is 16
   dimensional and trained on two sentences.
4. **Throughput is predicted, not measured.** tokens/s comes from a sizing
   model checked against a fabric simulation, agreeing within 15% and at
   ratio 1.00 on the current configuration. Both are models. Neither is a
   board.

Item 1 is the one that would let this claim what Architect Labs
demonstrated, and it is blocked on tooling rather than on design. Items 2
and 3 are ordinary remaining work.
