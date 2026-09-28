# What is verified, and what is not

Last run 2026-09-27. Every number here came from a command in this repo, and
every command is named so it can be re-run.

## Short answer

The flow turns a model spec into signed-off RTL for eighteen generated
blocks, which cover every operation in a Qwen layer: attention, RMSNorm,
SiLU, the gated MLP, the residual add and the rotary position embedding
among them. An earlier version of this file claimed that before the
rotary embedding existed. One
projection runs at the model's full size, 896 by 4864, bit-exact on every
output. A generated decoder sequences those blocks into a whole decode
step, and the trained checkpoint runs on it in RTL, embedding to argmax,
with every token of its output chosen by the hardware. A second
checkpoint trained with Qwen's structure, RoPE, two query heads over one
KV head, the gated SiLU MLP, two layers and a final norm, decodes in RTL
the same way, bit-exact on every logit. An LLM has written
and signed off ten of the nineteen blocks through the same gates, the rotary embedding the newest of them, and the multiply-accumulate unit's accumulator is formally proved
never to overflow, for any input sequence, on the int8 targets. A trained
language model decodes through the blocks' exact arithmetic and emits
text. It still does not host a local LLM the way Architect Labs does:
nothing in this repo has been put on a board, and the decoders run
models of 16 and 32 dimensions trained here, not Qwen's 896-dimensional
weights.
The remaining gap is listed at the bottom rather than glossed over.

## Verified

### The full suite

```
python3 tests.py            # 339 tests, or 337 without OpenSTA
```

### Spec to RTL, across the spec space

`python3 sweep.py` drives 105 cases through every stage: derivation,
RTL, simulation, synthesis, timing closure, FPGA mapping, the profile
fields the sizing model consumes, and a mutation sweep of the testbench
generated at that width. All 105 are clean: sixteen block kinds over six
model variants, 96 cases, and nine endpoint datapath options across four
link rates, including the options the endpoint's search tries and
rejects on timing before it settles. The multi-lane
projection was added after that run and passed the same gates on all six
variants separately; the two decoders are sized for their checkpoints and
are gated by the flow and `tests.py` instead. Cycles are measured by each
testbench. The rows below are Qwen2.5-0.5B, the base variant.

| case | cycles | fmax | cells | DV killed |
|---|---|---|---|---|
| mac, int8, acc29 | 1.00 /MAC | 169 MHz | 1473 | 8/8 |
| requant acc29 to 8 | 10.0 /activation | 115 MHz | 10445 | 5/5 |
| exp Q4.8 to Q0.15 | 4.00 /score | 164 MHz | 2370 | 4/4 |
| recip 26b to 17b | 4.00 /row | 206 MHz | 1636 | 4/4 |
| rsqrt 30b to 17b | 4.00 /row | 170 MHz | 1340 | 4/4 |
| matvec sequencer | 73.0 /column | 187 MHz | 1294 | 4/4 |
| wmem tile + loader | 2132 /tile | 333 MHz | 61712 | 5/5, 1 unproven |
| softmax, 21-bit scores | 49.0 /row | 120 MHz | 40262 | 6/6 |
| mlp layer | 158 /layer | 119 MHz | 23124 | 4/4 |
| proj, full size | 4726 /projection | 115 MHz | 13989 | 3/3 |
| rope | 1.03 /pair | 129 MHz | 21224 | 7/7 |
| resadd | 1.05 /element | 136 MHz | 4894 | 5/5 |
| gmlp, Qwen's MLP | 269 /layer | 110 MHz | 48418 | 5/5 |
| silu | 1.06 /element | 120 MHz | 11935 | 4/4 |
| rmsnorm, d896 | 1818 /norm | 115 MHz | 19904 | 3/3 |
| attn, hd64 | 1803 /head | 115 MHz | 132047 | 4/4 |
| crc 1G, 1 B/cyc | 1.00 /byte | 476 MHz | 774 | 6/6 |
| crc 10G, 16 B/cyc | 0.06 /byte | 85 MHz | 7333 | 6/6 |
| crc 25G, 16 B/cyc | 0.06 /byte | 385 MHz | 10527 | 5/5 |
| crc 100G, 64 B/cyc | 0.02 /byte | 325 MHz | 34682 | 5/5 |

At every link rate but 1G the endpoint's first datapath misses timing and
the flow advances to the next standard width and clock, which is the
architecture search, not a failure. The one DV case marked unproven is a
survivor the equivalence prover could not decide in its time budget.

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

## The model decodes in RTL

```
python3 decoder.py         # the integer reference, against float
python3 tests.py           # includes the RTL decode below
```

`generate.py` runs the checkpoint through bit-accurate models of the
blocks, but the host sequences it and picks each activation's scale at
run time in floating point. A decoder on an FPGA can do neither, so
`decoder.py` removes both. Every activation scale is fixed ahead of time
by calibration over the corpus, every stage is one of the generated
blocks' own golden models called with those constants, and nothing passes
between two stages but an int8 code. That integer-only model agrees with
the float checkpoint on all 93 teacher-forced corpus positions and
greedy-decodes the same text.

Then the hardware. `decoder.v` is one decode step, embedding lookup to
argmax, as a sequencer over one instance each of the generated
projection, RMSNorm, attention head and residual add, with every block
beneath them (matvec, the MAC, the requantizer, softmax, the exponential,
the reciprocal, the inverse square root) in the loop. One projection
instance runs all seven matrices; one RMSNorm both norms; one residual
add the embedding and both residuals. Weights, embeddings and gains sit
behind one registered parameter port, as they would in DDR; activations
and the KV cache are on chip. The testbench is the host: it feeds the
prompt, then feeds back whichever token the hardware chose, and checks
all 16 logits of every step against the integer reference.

```
  the agent writes the rtl
  the tools decide. the ag
TB_PROFILE tokens=46 span_cycles=174294 latency_cycles=4229
TB_PASS checks=847
```

Every character after each prompt is the hardware's argmax, fed back as
its next input, and all 736 logits are bit-exact. It closes the flow's
gates like every other block: the rules agent's first cut, which leaves
out the MLP's ReLU, fails on the first logit, and the second converges.
111 MHz from OpenSTA, 3790 cycles per token, so about 29,000 tokens/s for
this model, and 14117 LUTs, 22 DSPs and 2 BRAMs on yosys's UltraScale+
mapping. On the 7-series mapping it is 13247 LUTs and 22 DSPs, 64% and
24% of the XC7A35T on the team's Basys 3, so the decoder fits that
board. Mutation testing kills every applicable operator. The one that
first survived took the last maximum on a tie instead of the first: the
model's logits never tie, so the testbench now ends with a step whose head
weights are zeroed, and all sixteen logits tie at zero.

Two derivation rules only showed up at this size, and neither changes any
model the sweep runs. The attention head's weighted sum shares the
datapath's requantizer and needs 24 bits, more than a 16-wide model's
21-bit accumulator, so the accumulator now has that floor. And the
softmax's capacity was fixed at 256 whatever the context, past what the
head's matvec could count at this size; it is now bounded by the context.

What this is not: the model is the 16-dimensional checkpoint, one layer
and one head, in one 24-position window, with learned positions and a
ReLU MLP. The next section runs Qwen's structure.

## Qwen's structure decodes in RTL

```
python3 train_qwen.py      # one off: trains and writes tiny_qwen.json
python3 qwen_decoder.py    # the integer reference, against float
```

`tiny_qwen.json` is trained in this repo with Qwen's structure rather
than the simpler one above: no position vectors but RoPE on q and k, two
query heads sharing one key/value head, the gated SiLU MLP, two layers,
a final RMSNorm before the head. 19,616 parameters on the same two
sentences; a fused dot-product node in the autodiff, checked against
finite differences, made training it take minutes rather than hours.

The integer reference fixes every scale by calibration, as before, and
agrees with the float checkpoint on all 93 teacher-forced positions.
`qwen_decoder.v` runs a decode step over one instance each of the
projection, RMSNorm, attention head, residual add, rotary unit, SiLU unit
and a requantizer, looping over both layers and both query heads, with
the KV cache laid out per layer and per KV head so the two query heads
read the same keys. The gate is requantized to a power-of-two scale so
its int8 code shifts straight into SiLU's input, and SiLU(gate) * up is
requantized in one stream. Fed its own tokens back:

```
  the agent writes the rtl
  the tools decide. the ag
TB_PROFILE tokens=46 span_cycles=1139558 latency_cycles=26533
TB_PASS checks=847
```

All 736 logits are bit-exact. It closes every gate: 110 MHz and 24773
cycles per token. On the 7-series mapping it is 8376 LUTs, 164 more as
LUT RAM, 34 DSPs and 3 block RAMs, about 41% and 38% of the XC7A35T on
the team's Basys 3, so it fits that board. The first version did not: one
shared activation array with two writes a cycle, from the rotary unit,
and a key cache written two elements at a time could not be inferred as
memory, so all of it became flip-flops and multiplexers, 36289 LUTs.
Each activation now has its own small memory with one write port, and
q, k and the key cache are split into the two halves of the rotary pairs,
so every memory takes one write a cycle and maps to LUT RAM or block
RAM. `fpga.py` now counts LUT RAM as the LUTs it occupies; it had been
listing those cells as unmapped. The rules
agent's first cut stores the keys without rotating them, which is the
identity at position 0 and fails two steps later. Mutation testing kills
every applicable operator.

What is still not Qwen: 32 dimensions rather than 896, 2 layers rather
than 24, and a checkpoint trained here rather than Qwen's weights.

## Ready for the Basys 3

```
python3 board.py           # writes board_basys3/ and simulates it
```

`board_basys3/` holds everything Vivado needs to put the Qwen-shaped
model on the team's Basys 3: the decoder and its blocks, a top level on
the board's USB-UART, the weights as a block RAM image, the XC7A35T's pin
constraints and a batch build script. The top level takes a prompt as
UART characters, maps them to tokens through the checkpoint's vocabulary,
runs a decode step per position, and sends back each generated character
until the context is full. Simulated whole, UART frames in and out, it
answers `the agent` with ` writes the rtl `, every character checked
against the integer reference. Yosys maps it to 8632 LUTs and 172 more as
LUT RAM, 34 DSPs, 12 block RAMs and one MMCM on 7-series, about 42% of
the part; the MMCM runs the core at 50 MHz, half what the generic library
closes, since a real Artix-7 route is slower than that library. The one
step not taken here is running Vivado, which needs an x86 host.

## Attention scores wider than the exponential

The Qwen-shaped checkpoint first agreed with float on 48 of 93 positions,
and the reason was a limit in the attention head, not the model. Its
scores reached 149 and 205 in real units. The head clamped every score to
+-8 before the softmax subtracted the row maximum, because the
exponential's input spans +-16 and a score minus the maximum had to fit
it; once scores pass 8 they all clamp to the same value and the weights
flatten. Real models' heads routinely produce scores past 8, so this was
a bug for Qwen, not for this checkpoint.

The softmax now reads scores eight bits wider than the exponential and
clamps only each score minus the maximum, at the exponential's floor.
That clamp is exact: below -16 the exponential is zero in its 15-bit
output anyway. The head also forms scores from q.k shifted left by four
before its right shift, so shift_s can scale scores up as well as down;
the Qwen-shaped model needed about three times more than a pure right
shift allowed without saturating q. Registering the clamped difference
before the exponential kept timing; the softmax closes at 121 MHz and
the head at 115 MHz. The 16-dimensional checkpoint's scores are
bit-identical, since a left shift by four followed by a right shift by
four more is the same right shift.

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

| block | LLM result | drafts | model |
|---|---|---|---|
| MAC chiplet | converged | 1 | Haiku |
| matmul sequencer | converged | 1 | Haiku |
| weight memory | converged | 15 | Haiku |
| exponential | converged | 6 | Haiku |
| reciprocal | converged, 4 of 4 runs | 1, 2, 1, 2 | Haiku |
| inverse square root | converged, 4 of 4 runs | 3, 1, 2, 2 | Haiku |
| requantizer | converged | 5 | Sonnet |
| softmax | converged | 8 | Haiku |
| MLP layer | converged | 3 | Haiku |
| attention head | not converged | 8 | Haiku |
| RMSNorm | not yet attempted | | |
| SiLU | not yet attempted | | |
| gated MLP | not yet attempted | | |
| residual add | not yet attempted | | |
| full-size projection | not yet attempted | | |
| multi-lane projection | not yet attempted | | |
| decoder | not yet attempted | | |
| rotary embedding | converged | 5 | Haiku |
| Qwen-shaped decoder | not yet attempted | | |

The softmax row predates the wider score input added for the Qwen-shaped
checkpoint; the block Haiku signed off read 13-bit scores, not 21-bit.

Every block the flow generated before the attention head has been
written and signed off by an LLM through all four gates. The attention
head, with eight generated blocks beneath it, has not yet: Haiku's drafts
compile and simulate once the spec lists each sub-block's exact ports,
and the testbench's direct check of the score buffer names where they go
wrong, but eight drafts did not close it. Sonnet, which closed the
requantizer, did not return a single draft: on this spec each call spent
more than the 25 minute limit in extended thinking, at low effort, and
the run was stopped after two. Haiku is the smallest current Claude model,
run through the Claude Code CLI; the requantizer is the one block it did
not close, reaching timing with a correct datapath and running out of
drafts there, and Sonnet closed it. Transcripts of the three hardest runs,
every draft with the tool feedback it received, are in `transcripts/`.

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

### What it took for the last three

The same diagnosis as the table units, three times over, and then a
series of defects in what the tools told the agent. Each was found by
tracing a failed run draft by draft and asking what an engineer would
have needed to see.

The specs were not implementable as written. The requantizer described
five pipeline stages and demanded a latency of nine, and never said what
a shift of zero does. Softmax never gave its normalising formula. The MLP
layer never gave its weight layout or said the rectifier follows the
requantizer, and declared a 26-bit port against a 13-bit testbench wire.
Softmax also said a score arrives "on the cycle after s_addr = a", which
is true, but three different agents drove s_addr from a register and
read one edge early, giving every score the same weight. The spec now
counts the two edges.

The feedback was the rest:

- Compile errors named a line number and nothing else. The model sees
  its draft without line numbers, so it rewrote around an illegal
  literal and kept it in every draft. Errors now quote the source line,
  and a sign inside a sized literal is explained.
- A yosys message about legalising `$_DFFE_PN0P_` is how an asynchronous
  reset shows up. It is now restated in the design's terms.
- A reply that was a fragment rather than a module used to be compiled
  as the design. It is re-asked instead, without spending an iteration.
- A draft that never lowered busy hung the simulator, and the timeout
  raised out of the flow and ended the run. It is now a failed
  simulation the agent can read.
- A timing failure reported only a slack. It now names the path, from
  the register whose net keeps its RTL name to the endpoint.
- The requantizer multiplied a signed accumulator by an unsigned scale,
  and when that was fixed, lost the sign again in a concatenation, which
  Verilog makes unsigned. Both are now reported as facts about the code,
  checked against all 48 reference designs with no false positives.
- The MLP testbench now checks the hidden layer directly and reports it
  before the outputs, since a draft sat for six iterations on "output 0
  wrong" when layer one was at fault.
- Softmax reported `expected_w=x` for a weight emitted past the end of a
  row, and a right value under the wrong index read as an arithmetic
  error. Both are now named for what they are.

Two findings about the models themselves. Sonnet was not failing: it
spent every call in extended thinking past the ten minute timeout, 778
thinking deltas in 150 s without a character of output. At low effort
its first requantizer draft passed all 209 checks. And a larger model is
not simply better here: it fell into the same read-latency trap as
Haiku, because the trap was in the spec.

The deterministic agent is still what generated the committed RTL, and
it is still the right default for anyone who wants hardware reproducibly
and offline. The LLM path is no longer only a demonstration of the loop.

## An attention head

The last block a transformer layer was missing. One head, one decode
step: the query arrives as head_dim int8 values, and keys and values come
from a KV cache through registered read ports. Scores are q.k over each
cached position, rounded and shifted into the softmax's score format and
clamped to half its range, so that a score minus the row maximum always
fits the exponential's input. The softmax block turns them into weights,
and the output is the weight-averaged value row, requantized to int8.

Every part is an existing generated block: matvec and the MAC for the
scores, softmax for the weights, the requantizer for the output. The new
logic is the routing between them and a 16-by-8 multiply-accumulate for
the weighted sum, which the 8-bit MAC cannot do.

Against real attention on the same integers the fixed-point head is
within 0.08 on values up to 60. The rules agent converges in two
iterations on every model variant, 115 MHz on Qwen2.5-0.5B, 3265 LUTs
and 12 DSPs on yosys's UltraScale+ mapping with its sub-blocks, and
mutation testing kills every mutant on all six. Its weighted-sum
accumulator is sized from its own bound, the weights summing to under
2.0, rather than from the MAC: at 16-bit operands the MAC's 46 bits made
an add that could not close timing, and the bound needs 32. Its seeded first cut uses each value on the edge before
its registered read arrives, which is the mistake every model made on
softmax's score read, and the testbench catches it. The testbench checks
the scores and the softmax weights directly, before the outputs, since
both are upstream of them.

## RMSNorm

Qwen normalises before attention and before the MLP. The inverse square
root was generated for this from the start; this is the sequencer that
uses it. Pass one sums the squares of the row, starting from epsilon.
One inverse square root gives a mantissa and a shift, and pass two forms
t = (x * g * m) >> (e + k) per element and requantizes it, with the
mean's square root and the quantization scales folded into the output
scale. Against float RMSNorm it is within 0.04% of the row's largest
output.

The extra shift k is derived rather than chosen. Since x squared cannot
exceed the sum of squares, |x| is at most its square root, and the
halved exponent keeps 2**e above half of that, so |t| stays under
2**(data_width + 17 - k). k is what brings that inside the
requantizer's input: 0 for the int8 targets, 2 for the tiny model. The
testbench drives a full-scale spike to put t at that bound, and an
all-zero row where epsilon is the whole answer. It checks the sum of
squares and the rsqrt result directly, before the outputs, and the
seeded first cut that forgets epsilon fails on exactly that check.

At 16-bit operands the product is 50 bits wide and missed timing by
1.54 ns as one multiply; splitting both factors into four partial
products closes it on every variant.

## SiLU

The nonlinearity in Qwen's gated MLP, x * sigmoid(x), built from two
units that already exist. With a = |x| and e = exp(-a), which the
exponential handles because its argument is non-positive, sigmoid(x) is
1/(1+e) for x >= 0 and e/(1+e) for x < 0, and the reciprocal supplies
the division. It streams one element per cycle at 120 MHz. Against float
SiLU on all 8192 inputs of its Q4.8 format the worst error is 0.018,
under a seventh of an int8 step at typical activation scales, and it
comes from the reciprocal table amplified by x near 5. The seeded first
cut takes the positive-x numerator for every x, which is right for half
the range, and the testbench catches it at x = -1.

## Qwen's gated MLP

down(SiLU(gate(x)) * up(x)): three projections, not two. One matmul
sequencer, the SiLU unit and one requantizer run all of it, over one
weight memory holding the three matrices back to back, with the
requantizer's scale chosen by phase. The gate is rounded and clamped
into SiLU's input format and streamed through it in order; the up
projection is requantized; their product is requantized again; and the
down projection runs over that. The testbench checks the SiLU, up and
product buffers directly, in that order, before the outputs, and the
seeded first cut, whose up projection reads the gate's matrix, fails at
the up buffer. 110 MHz on Qwen2.5-0.5B, 6/6 variants clean.

Mutation testing found one hole on the way. An off-by-one in the
SiLU-times-up product survived on one variant and could not be decided
on the rest: the product is shifted right about twenty bits and six
random hidden units almost never land on a rounding boundary. The
generator now searches the seed until one does, and the mutant dies on
every variant.

## Rotary position embedding

Qwen does not add a position vector; it rotates q and k. Each pair
(x[i], x[i + d/2]) of a head turns by pos * theta_i, with theta_i =
1e6**(-2i/64) on Qwen2.5, so the fastest pair wraps every six tokens and
the slowest barely moves across the context. The unit streams one pair
per cycle. The angle is kept in turns as a 24-bit fraction, so wrapping
past a full turn is free, from a generated table of the 32 frequencies;
it indexes a generated 4096-entry sine table, cosine a quarter turn on,
and four products give y1 = x1 cos - x2 sin and y2 = x2 cos + x1 sin,
rounded and saturated. Against float RoPE the worst error on any
unsaturated output is 0.61 of an int8 step, 0.5 of which is the final
rounding. 129 MHz, 4 cycles, 1874 LUTs and 5 DSPs, and all six model
variants converge; at 16-bit data the 16 by 24 product missed timing by
0.78 ns and is split into two partial products.

The seeded first cut rotates by minus the angle, the transpose of the
rotation, which is identical to it at position zero; the testbench walks
the fastest pair through forty positions and catches it. Mutation testing
found one hole first: a phase one count off moves the table index only
when pos * F_i sits one count below a rounding boundary, about one
position in 4096, so random pairs never land there. The testbench now
searches the grid for such points, and the mutant dies on every variant.

## Residual add

The add that closes each half of a layer, attention's output onto its
input and the MLP's onto that. The operands carry different scales, so
it is the requantizer's arithmetic over two terms: y = sat(round((a *
scale_a + b * scale_b) >> shift)), one pair per cycle at 136 MHz. The
testbench plants exact rounding ties, negative ones among them, so the
direction a tie rounds is checked and not only that it rounds, and the
seeded first cut that truncates is caught.

Mutation testing found a reset hole here that the deeper SiLU pipeline
had been hiding. Both streaming testbenches checked valid_out once,
after three idle cycles; a three-stage pipeline has flushed by then with
no reset at all. They now check it from the first cycle of reset.

## A projection at full size

Every composite layer block holds its activations in a 64-entry bank, so
until now no block had run a matrix at the size the model actually has.
A Qwen2.5-0.5B layer's up projection is 896 by 4864. Growing the banks is
not the answer: a real accelerator keeps activations in SRAM outside the
compute and streams them in, as the weights already are here. The
projection block does that. It holds no activation buffer; it drives
matvec, the MAC and the requantizer over registered activation and weight
memory ports sized from the model's largest dimension, and labels each
output with its column index, carried through the requantizer's pipeline
alongside the value.

`tests.py` runs the full 896 by 4864 projection, 4,358,144
multiply-accumulates, and checks all 4864 outputs against the Python
golden: bit-exact, in 34 s of simulation. The testbench computes both
operands from a hash of the address, so it needs no per-element
initializer and a full-size case costs no more to write than a small
one. 115 MHz on Qwen2.5-0.5B, 1831 LUTs and 9 DSPs.

The seeded first cut labels each result with matvec's current column
index, which by the time the requantizer finishes belongs to the next
column. On a long reduction the next column has not started yet, so the
label happens to be right; the testbench includes a reduction shorter
than the requantizer's pipeline, and the bug fails there.

Mutation testing found a hole in the first version of that testbench.
On the int4 variant the widest vector in the design is the 26-bit weight
address, not the accumulator, and halving it survived: no case reached
address 8192, so the address's top bits were never exercised outside the
full-size test. The flow's testbench now ends with a case sized to reach
past half the address width on every variant.

## FPGA counts were double

Every composite block's LUT and DSP count was about twice its real size.
yosys prints one table per module, alphabetically, and then the totals
for the whole hierarchy; the parser took everything after the top
module's own table, which for any block that sorts early is most of the
submodules and then the totals again. The attention head read 6364 LUTs
and 24 DSPs and maps to 3182 and 12; the decoder read 25655 and maps to
13905. The parser now reads only the final statistics pass and, when
there is one, its hierarchy totals. A first version of that fix doubled
the MAC instead, because synth_xilinx prints a hierarchy section of its
own before the script's last pass; `tests.py` now checks both shapes.
Single-module blocks were always counted correctly.

## Enough lanes to use the DDR

The projection above does one multiply-accumulate a cycle, and batch-1
decode reads every weight once per token, so at Qwen2.5-0.5B's 494
million weights that is under a quarter of a token a second at 115 MHz.
The sizing model's 4 tokens/s on one board assumes the board is bound by
DDR bandwidth instead, and nothing in the RTL could get there.

The multi-lane projection derives its width from the board: the smallest
power of two of lanes whose bytes per cycle, at the fabric clock, cover
the Zybo's 2 GB/s of sustained DDR bandwidth, which is 32. Each lane is
an instance of the generated MAC and takes its own byte of a 256-bit
weight word; a group of 32 columns reads each activation once, then the
32 sums move to a shadow bank and drain through one requantizer while
the next group accumulates. At the full 896 by 4864 up projection it is
bit-exact on every output in 136,844 cycles, 0.5% over one cycle per row
per group and 32 times fewer than the single lane. 115 MHz, 4516 LUTs
and 40 DSPs. The rules agent's first cut wires each lane to the mirrored
byte of the word and is caught.

Carried to a whole Qwen2.5-0.5B token, the weight matmuls take about
15.5 million cycles, 7.4 tokens/s at 115 MHz, so compute is no longer
what binds: the DDR's 4 tokens/s is. The attention head is still one
lane, and over a full 1024-token context it would take about 44 million
cycles a token, so it is now the block to widen.

## Cycle counts are measured, not assumed

The sequencing blocks' profiles used to report cycle counts the
testbench computed from a formula, not ones it counted. They were close
for the streaming blocks and far off for the layer blocks: the attention
head reported 128 cycles per head and a 16-cycle latency, and measures
1802 and 2954. Every sequencing testbench now counts clock edges from
start to the last output and to the first, and the profile carries what
it counted:

| block | unit | was | measured | latency was | measured |
|---|---|---|---|---|---|
| attention head | head | 128 | 1802 | 16 | 2954 |
| RMSNorm | norm | 1792 | 1818 | 16 | 923 |
| gated MLP | layer | 126 | 269 | 16 | 225 |
| MLP layer | layer | 78 | 158 | 16 | 113 |
| softmax | row | 64 | 52.3 | 8 | 143 |
| matmul sequencer | column | 68 | 73 | 68 | 74 |
| weight tile | tile | 1024 | 2132 | 64 | 69 |
| SiLU | element | 1.03 | 1.05 | | |
| residual add | element | 1.03 | 1.06 | | |

The matmul sequencer's five extra cycles per column are real overhead
the formula hid. The weight tile's unit is now the whole tile, loading
it and computing over it, where before it was the load alone. The
tokens/s predictions do not move: the sizing model reads only the MAC's
and the link endpoint's profiles, and both were already measured. The four
pipelined units (exp, recip, rsqrt, requant) still print their
testbench's pacing. Their latency is exact, since each check requires
valid_out high on precisely that cycle, but their cycles per input is the
testbench feeding one input at a time and waiting, not the unit's
throughput.

## Formal proof: the accumulator cannot overflow

The accumulator width comes from a closed-form rule, and until now every
check of it was a testbench driving some vectors. `formal.py` proves it
for all of them. A wrapper keeps the exact sum of the products the MAC is
given, in a register two bits wider than the accumulator, and counts the
accumulations since the last clear. It assumes at most the reduction
depth of them per clear and operands within their quantized ranges, and
proves by k-induction with yosys-smtbmc and z3 that the sum always fits
and that `acc` always equals it.

```
python3 formal.py
```

| variant | depth | acc bits | result |
|---|---|---|---|
| Qwen2.5-0.5B, int8 | 4864 | 29 | proved |
| Qwen3-0.6B, int8 | 3072 | 28 | proved |
| int4 weights | 4864 | 25 | proved |
| tiny model | 256 | 24 | proved |
| 16-bit operands (two variants) | 11008, 8192 | 46, 37 | not proved: timeout |

It is stated on the spec's ports, so it applies to whoever wrote the RTL:
the MAC Haiku wrote in one draft is proved by the same run. And it can
fail: it rejects a truncated accumulator, a width two bits under the
rule, and six of the other seven mutation operators that apply.

It found three things.

- **The rule is one bit conservative unless the depth is a power of
  two.** It rounds log2(depth) up. At depth 4864 the worst case is
  4864 x 2^14, which fits 28 signed bits, and the proof holds at 28 and
  fails at 27. At depth 8192 the rule is exact: 29 holds, 28 fails. Safe
  either way; the rule is left as it is.
- **The rule depends on an operand range the spec never stated.** With
  int4 weights the accumulator is sized for 4-bit weights, but the weight
  port is 8 bits wide, and the proof produced a counterexample with a
  full-range value on it. The MAC spec now states the range wherever a
  quantization is narrower than its port.
- **It cannot see a flipped clock edge.** The formal model advances every
  flop once per step whichever edge it is written on. The testbench's
  edge-discipline check covers that one.

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
   an x86 machine with Vivado for the Zynq and Artix parts. For the
   team's Basys 3 that step is now one command: `board_basys3/` is a
   complete Vivado build of the Qwen-shaped decoder behind the board's
   USB-UART, simulated end to end, and `build.tcl` is the part not run.
2. **The generated blocks are the arithmetic, not the whole engine.** The
   flow generates the multiply-accumulate unit, the requantizer between
   matmuls, the exponential and the reciprocal that softmax needs, the
   inverse square root that RMSNorm needs, and the CRC32 fabric
   endpoint, the weight-streaming sequencer that drives the MAC through
   a matrix, the weight tile and its loader, softmax as a single
   sequenced block, an MLP layer that drives two matmuls in order
   and routes the activations between them, and one attention head over
   a KV cache, RMSNorm, and a projection at the model's full size over
   external memories. Softmax is hardware apart from
   accumulating the sum. Every per-layer operation is a generated block,
   the residual add and the rotary position embedding included; the
   checkpoint here uses learned positions, so the decoder has no use for
   the rotary unit, but Qwen does. The generated decoder runs a
   whole decode step of the 16-dimensional checkpoint in RTL. At Qwen's
   size nothing yet sequences a layer: many heads over shared KV heads,
   24 layers and DDR-streamed weights are still on the host. Qwen's MLP is the gated block,
   down(SiLU(gate(x)) * up(x)); the older MLP block, two matmuls with a
   ReLU, is not Qwen's and remains as the simpler case. The composite
   layer blocks still hold 64-entry activation banks; only the
   projection block streams activations from memory at full size, and
   the attention and MLP blocks have not been moved onto it. In `generate.py` those run on the host, and the
   output says so each run.
3. **The model is small and its weights are its own.** Qwen3-0.6B is not
   loaded; there is no numeric stack here to load it with and no network
   dependency wanted in a capstone repo. The committed checkpoints are
   real trained transformers with a real tokenizer, one 16 dimensional
   and one with Qwen's structure at 32 dimensions and 2 layers, both
   trained on two sentences.
4. **Throughput is predicted, not measured.** tokens/s comes from a sizing
   model checked against a fabric simulation, agreeing within 15% and at
   ratio 1.00 on the current configuration. Both are models. Neither is a
   board.

5. **Formal coverage is one property of one block.** The accumulator
   proof holds for the int8 targets; with 16-bit operands the MAC splits
   its multiplier, the proof becomes multiplier equivalence, and z3 runs
   out of time. Those variants rest on the closed-form bound, the
   testbenches and mutation testing. No other block has a formal proof.

Item 1 is the one that would let this claim what Architect Labs
demonstrated, and it is blocked on tooling rather than on design. Items 2
and 3 are ordinary remaining work.
