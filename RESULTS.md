# What is verified, and what is not

Last run 2026-09-29. Every number here came from a command in this repo, and
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
and signed off ten of the nineteen blocks through the same gates, the rotary embedding the newest of them, and a
decode step whose fifteen blocks the agents wrote, ten of them the models' own, matches the integer model; the
multiply-accumulate unit's accumulator is formally proved
never to overflow, for any input sequence, on the int8 targets. A trained
language model decodes through the blocks' exact arithmetic and emits
text, and a generated RTL sequencer decodes the real Qwen2.5-0.5B, all
24 layers on its own weights, completing "The capital of France is" with
" Paris" in simulation, through the board package's own registers
against a DDR model that stalls at random. Qwen3-0.6B, the model
Architect Labs hosted, decodes the same way: all 28 layers and the head
in generated RTL choose " Paris", the integer model's token. The package is generated for any Zynq board in
`boards.py`, today the Zybo Z7-20 and the ZC706, and without Vivado the
open 7-series flow places and routes the whole design on both parts at
50 MHz, with bitstreams that round-trip frame for frame. The Zybo takes the
ZC706's 32-lane core as well, 24 of its score multipliers in LUTs so the
open router can finish, and is then about a third faster a token. The ZC706's core
is twice the Zybo's width, 32 lanes, and the work to get there found two
faults in the DDR bridge that simulation had passed: the weight image's
byte order, which would have reversed every word's lanes on the board,
and a streamer that could hand the core another line's weights after a
jump back. Both are fixed and tested. Several boards
can share one model as GALS stages on their own clocks, simulated exact
to one board's tokens over fabric UARTs, with the Zynq's UDP protocol
run on a host. One command, `spec2rtl.py`, now takes a model's shape or
checkpoint and a list of boards, any mix, and returns verified RTL: every
generated block signed off at that shape's parameters and every width in
the cluster, and the whole design, one board, a pipeline of up to eight
mixed boards, or the weights split over several, simulated against the
one-board integer model to the logit, then every board's package. Split
by weights, each board's share is sized by its speed, each rank also runs
through its own registers and DDR bridge
with the ARM's side serving every gather, and the ARM's UDP program for
those gathers has run on a host, asking again for datagrams lost or
corrupt. The ARM programs themselves, compiled unchanged, now run on
their own packages' RTL under Verilator: the Zybo's and the ZC706's print
"The capital of France is Paris. Paris is the capital of France. Paris is
the capital of France.", and the ZC706's Qwen3 package "The capital of
France is Paris. The capital of the United States is Washington, D.C.",
all 16 tokens of each and their logits the integer model's; so does that
Qwen3 split by weights over a ZC706 and a Zybo, both ranks' programs on
their own RTL, gathering over UDP. Running the gates at the widths the
design really uses found the attention head's scores could overflow at
16-bit activations; its score lanes now have their own, wide enough, MAC.
The links recover from bit errors by CRC and resend, and the planner's
cycle model is within 0.02% of simulation on the real models. It still does not host a local LLM the way Architect Labs
does, for one reason: nothing has been loaded onto a board, and that is
now the whole of the remaining step (`HANDOFF.md`).
The remaining gap is listed at the bottom rather than glossed over.

## Against Redwood

Redwood, Architect Labs' result as the proposal cites it, and this repo
as of this file:

| | Redwood | this repo |
|---|---|---|
| what the agent was given | an architecture two human architects specified | the model's shape or checkpoint and a list of boards: the widths, lanes, split and shares are derived |
| what it wrote | RTL, a UVM environment, formal properties, firmware and drivers | every block through five gates, the sequencer, the DDR bridge, the register block, the gather mover and the ARM programs, for each board |
| model | Qwen3-0.6B | Qwen3-0.6B and Qwen2.5-0.5B |
| hardware | one AMD Versal VPK180, \$17,995 | any mix of Zynq boards, the Zybo Z7-20 (\$300) and the ZC706 today; other FPGAs over the fabric UART |
| across chips | none | the layers split or the weights split over 1 to 8 boards, each board's share sized by its speed, bit-exact to one board |
| verification | UVM, over 95% functional coverage | each block bit-exact to its golden model with its testbench mutation-tested; the whole decode at every layer of both real models; each package's own ARM program on its own RTL, 16 tokens with every logit, alone, split by layers and split by weights |
| on a board | yes, 12.1 tokens/s | not yet: the bitstreams route and round-trip in the open flow, and `HANDOFF.md` is the bring-up |
| tokens/s per dollar | 12.1 / \$17,995 | estimated: one 32-lane Zybo 1.47 / \$300, about 7 times; two split by weights 2.55 / \$600, about 6 times |

The one row that is Redwood's alone is the board. Until one runs, the
throughput here is the planner's, which the co-simulation's own step
times put within 10%.

## Verified

### The full suite

```
python3 tests.py            # 417 tests, or 415 without OpenSTA
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

## The real Qwen2.5-0.5B on the blocks' arithmetic

```
python3 fetch_qwen.py      # once: Qwen2.5-0.5B from Hugging Face, ~1 GB, git-ignored
python3 qwen_real.py       # the checkpoint in float, pure Python
python3 qwen_int.py --act-bits 16 --per-channel
```

The checkpoints above are trained here. This is Qwen's own: its 494
million weights, its byte-level BPE tokenizer, loaded and run in pure
Python, about 11 seconds a token in float. `qwen_real.py` completes "The
capital of France is" with " Paris. It is the".

`qwen_int.py` runs the same checkpoint the way the hardware does. The
weights are int8 with a scale per output channel, the activations a fixed
width with every scale fixed by calibration over a paragraph of text, and
each stage is a generated block's golden model derived at Qwen's own
size: RMSNorm over 896, heads of 64 with the wide score path, RoPE with
theta 1e6, SiLU, the residual add, the requantizer. The q, k and v biases
are added to the accumulator, in its own units, before requantization.

| activations | teacher-forced agreement with float | greedy continuation |
|---|---|---|
| int8 | 0 of the first 5 positions | unrelated tokens |
| int16 | **14 of 16** | "The capital of France is Paris. Paris is" |

int8 activations fail for a reason every int8 LLM deployment meets:
Qwen's residual stream carries a few massive activations, about 700 in
layer 2 where typical values are near 1, and one static scale per tensor
leaves a handful of levels for everything else. Every stage of layer 0
was already 10 to 20% off. At 16 bits the stages are within about 2%, and
the model decodes. The w8/a16 datapath is not new: it is the sweep's
asym_w8_a16 variant, which every block derives and signs off.

The one stage still well off at 16 bits is SiLU times up, 9% in layer 1,
from SiLU's fixed Q4.8 output.

That run is the arithmetic in Python; no RTL simulation runs a
494-million-weight token. What does run in RTL is the real matrices. The
multi-lane projection gained a per-column mode, in which each output
column's bias, scale and shift come from a small memory read two edges
ahead of its sum, the bias added in the accumulator's own units, since a
per-channel weight scale gives every column its own requantizer
constants. `qwen_cosim.py` feeds it Qwen's own layer-0 matrices at full
size, with the int16 activation the integer model computes for a real
token:

```
python3 qwen_cosim.py
projection: 16 lanes of 16-bit, 37-bit accumulator
q_proj, 896 to 896, with bias    bit-exact  50494 cycles
down_proj, 4864 to 896           bit-exact  272702 cycles
rmsnorm, 896 wide, real gains    bit-exact
rope, 14 heads at position 4     bit-exact
attention head 0 over 5 cached positions bit-exact
silu on the 4864-wide gate       bit-exact
residual add after attention     bit-exact
```

Every block type in a Qwen layer, fed real layer-0 data for the prompt
"The capital of France is", matches the integer model on every output:
the projections with per-channel constants and bias, RMSNorm with the
checkpoint's own gains, the rotation of all 14 query heads, attention
over the real five-position cache, SiLU on the full gate, and the
residual add. So the chain holds block by block: the real checkpoint,
quantized, through the generated RTL, equals the arithmetic that
decodes "Paris" from it. What it is not is one sequenced layer in RTL at
this size; each block runs on its own inputs.

## A real Qwen token through one generated sequencer

```
python3 qwen_full.py --layers 1 --prompt "Paris"     # 14 minutes
python3 qwen_full.py                                  # all 24 layers, hours
```

The co-simulations above run one block at a time. `qwen_full.py`
generates one sequencer for the whole decode step at Qwen2.5-0.5B's own
size, over the same blocks at 8-bit weights and 16-bit activations: the
per-column 16-lane projection, RMSNorm, the multi-lane attention head,
RoPE, SiLU, the requantizer and the residual add. What would be DDR on a
board is behind ports: a 256-bit weight word a cycle, which the
testbench serves from a 1 GB image of the checkpoint with $fread, a
memory of every projection column's bias, scale and shift, the norm
gains, and the KV cache, written a lane at a time. The embedding lookup
reads the tied head's own weight words and requantizes them with the
token's constants, and the head runs as 32 projection chunks over
151936 tokens with a streaming argmax, so no logit is stored. For this
the integer model's two edges became hardware operations too: the
embedding a per-token requantize, the head a per-column requantize to one
calibrated logit scale. It still decodes "The capital of France is
Paris."

All 24 layers, on the prompt "The capital of France is":

```
STEP pos=0 tok=785  cycles=22809961
STEP pos=1 tok=6722 cycles=22811641
STEP pos=2 tok=315  cycles=22813321
STEP pos=3 tok=9625 cycles=22815001
STEP pos=4 tok=374  cycles=31375553 next=12095
```

Token 12095 is " Paris", the integer model's choice: the generated RTL,
sequencing real Qwen2.5-0.5B weights through every layer and the full
vocabulary head, completes the sentence. A prompt position is 22.8
million cycles, and the head adds 8.6 million; at 16 lanes and 100 MHz
that is 0.23 s a position, compute bound, where the 32-lane projection
and a DDR-fed weight port would bring it to the bandwidth bound. The run
took about three and a half hours of simulation. With the stack cut to
its first layer the same sequencer also matches, in 9.5 million cycles.

The testbench above answers every port the next cycle, which no DRAM
does, so `zybo.py` puts the same core, unchanged, on a memory system
shaped like the Zybo's: the weights, the column constants and the KV
cache in DDR behind one 64-bit AXI3 master, the width of a Zynq-7000 HP
port, and the norm gains in block RAM. Each DDR-backed port has a line
buffer filled by 16-beat bursts, KV writes go out as strobed single
beats, and while any port's word is still on the bus the core's clock is
held, a BUFGCE on the FPGA, so none of the generated blocks needed a
stall input. Against a DDR model with 30 cycles of latency, the
one-layer run chooses the same token, 70526, in exactly the same
9,510,162 core cycles as the direct run: holding the clock changes
nothing the core computes.

The first version took 126 million bus cycles for it, 13.3 per core
cycle, because it had one read in flight at a time and every weight
line paid the full DDR latency. The weights now have their own read
master, as they would have their own HP port, with a streamer that keeps
up to four bursts outstanding ahead of the line the core is on, into
eight line slots. The same run then takes 41.7 million bus cycles, 4.39
per core cycle, against a floor of 4.0 set by one 64-bit port feeding a
256-bit word a cycle; the token and the core's 9,510,162 cycles are
unchanged. At 100 MHz a full Qwen2.5-0.5B token is then about 1.4
seconds on one HP port.

The Zynq has four HP ports, so the weight lines are now striped across
four masters, line L on port L mod 4, with 32 slots and each slot always
filled through the same port so its responses stay in order. The first
four-port run managed only 2.15, and counting stalled cycles by cause
put almost all of it on weight misses in the head, where the address
stream is perfectly sequential: the streamer took a full window as a
jump, restarted at the core's line and re-walked 32 lines it already
held. It now waits when the window is full and restarts only when the
core's line jumps. The same run then takes 12.86 million bus cycles,
1.35 per core cycle, token and core cycles again unchanged:

| weight path | bus cycles per core cycle | full token at 100 MHz |
|---|---|---|
| one port, one read at a time | 13.3 | 4.2 s |
| one port, 4 bursts outstanding | 4.39 | 1.4 s |
| four ports, 32-line window | 1.35 | 0.42 s, 2.4 tokens/s |
| four ports, weights int8 in DDR | **1.17** | **0.37 s, 2.7 tokens/s** |

The last row fixes a waste that also mattered for fitting the board.
The core's word carries each weight in a 16-bit lane, since the
datapath is 16 bits, and the DDR image had stored them that way: 988 MB
of the Zybo's 1 GB, leaving the processor almost nothing. DDR now holds
the weights as int8, 16 to a 128-bit word, 494 MB, and the bridge
sign-extends each byte on the way into the core, which halves the bus
traffic too. Same token, same core cycles. What remains between this and
the sizing model's 4 tokens/s is the constant port's line misses, which
are not prefetched, and projection restarts. The first run of
this deadlocked, and the fault was the testbench's: before reset the
master's valids are X, !X is false, and the DDR model ran a phantom
write at cycle 2, then waited forever for its data beat.

It is also a design, not only a simulation. Yosys maps the whole
sequencer with every block under it to 25650 LUTs, 3240 more as LUT RAM,
13600 flip-flops, 148 DSPs and 16 block RAMs on 7-series: 54% of a Zynq
7020's LUTs and 67% of its DSPs. The weights, column constants and KV
cache are the ports' side, which on the team's Zybo is its 1 GB of DDR,
reached through the HP ports; `board_zybo.py`, below, writes that
connection.

## A Zybo package

```
python3 board_zybo.py                                  # board_zybo/, from the 24-layer build
python3 board_zybo.py --work build_qfull1 --out build_bz1 --sim --jitter
```

`board_zybo/` is what the team's Zybo Z7-20 needs to run the sequencer
above, generated rather than written by hand. `rtl/` has the core and
its blocks, the DDR bridge with every region at its real DDR address
(weights from 0x08000000, below them the ARM's program), and
`fpgai_zybo`, a top level with an AXI-Lite register block and five AXI4
masters: the four weight streams on HP0 to HP3, the constants and the KV
cache on ACP. `build.tcl` is the Vivado block design: the PS7 with
Digilent's preset, the ports and their interconnects, FCLK0 at 50 MHz,
then the bitstream and the `.xsa`. `sw/main.c` is the bare-metal ARM
side: it reads the SD card's four files into DDR, zeroes the KV cache,
flushes its caches, runs the prompt a position at a time through the
registers and prints each generated token's text over the USB-UART. The
software reads the layout registers first, so SD files and a bitstream
from different builds are refused before anything runs. `sd/` holds the
494 MB int8 weights, the column constants, the vocabulary and the
prompt, and is not committed.

Yosys maps the whole top level, 24 layers, bridge and registers, to
33169 LUTs, 7380 more as LUT RAM, 15922 flip-flops, 148 DSPs and 36
block RAMs on 7-series: 76% of a Zynq 7020's LUTs counting the LUT RAM,
67% of its DSPs. (That was before the streamer's later fixes, which add a
few dozen flip-flops; the routed utilization below is current.)

The testbench drives the design only through AXI-Lite, as the ARM would,
against the DDR model, and the one-layer build chooses token 70526, the direct testbench's, in
11,140,965 bus cycles, 1.17 a core cycle as before.
The DDR model used to accept every read the cycle it came and send every
burst without a gap, which no interconnect shared with a processor does.
With `--jitter` every port drops its ready and gaps its beats at random,
and the first such run deadlocked at core cycle 6339: the weight
streamer saw a jump back in the core's address for one cycle only, and
when a request was still waiting for arready that cycle, the restart was
skipped and forgotten, leaving the stream 255 lines ahead of a core that
waited for good. The always-ready model could never show it; the board
would have. The jump is now held until the stream can restart, and a
stream past its window counts as one. With that fix the jittered run
chooses 70526 too, in 11,452,288 bus cycles, 1.20 a core cycle
(11,452,707 on the streamer as corrected later, in "32 lanes on the
ZC706"; always ready, its count is unchanged). Both run the core's
9,510,162 cycles; the core counter they print is a few
higher because it is read before the AXI-Lite write that starts the
step, and the idle core ticks through that write.

The whole model then ran the same way: all 24 layers and the head,
the prompt's five positions, through the registers only, against the
stalling DDR model. It chose token 12095, " Paris", as the direct run
and the integer model do, each prompt position in the direct run's
22.81 million core cycles and 26.7 to 27.0 million bus cycles, the head
step in 31,375,557 core cycles and 37,457,732 bus cycles: 1.18 bus
cycles a core cycle, 0.54 s a position and 0.75 s for the head at
50 MHz. Four hours of simulation. That run used the streamer before the
fixes in "32 lanes on the ZC706"; the rerun on the corrected one chose
" Paris" too, its positions in 26.75 to 27.05 million bus cycles and the
head step in 37,459,001, 1.19 bus cycles a core cycle.

Three tests keep it: `test_weight_streamer_survives_stalls` runs the
streamer alone on four stalling ports under address streams of 1502
runs and jumps on two seeds, checks every word it hands the core, and
fails on both earlier streamers; `test_bridge_end_to_end` runs the
bridge around generated 16- and 32-lane cores on small checkpoints and
requires the integer model's tokens; `test_zybo_register_block` runs the
registers around a stub core whose clock ticks one bus cycle in four,
and fails if a start is a one-cycle pulse or done is not kept.

## A routed Zynq bitstream, without Vivado

```
board_zybo/open/build_open.sh     # Yosys, nextpnr-xilinx, Project X-Ray
```

Vivado needs an x86 host, but the open 7-series flow (openXC7:
nextpnr-xilinx on Project X-Ray's database) supports the Zybo's
XC7Z020, and it builds and runs on this ARM Mac. `board_zybo/open/` has
what the flow needs in place of the IP integrator: `fpgai_ps7.v`
instances the PS7 directly and wires GP0 to the register block (the
CPU's register accesses are single beats; their IDs are kept and
returned), the four weight masters to HP0 to HP3 and the fifth to ACP.
The whole design, 24 layers, bridge, registers and PS7, then places and
routes on the real part:

| clock | post-route Fmax | at 50 MHz |
|---|---|---|
| core (gated, through a BUFGCTRL) | 73.4 MHz | passes |
| bus and registers (FCLK0) | 65.4 MHz | passes |

using 28% of the LUT sites (LUT RAM included), 132 of 220 DSPs and 49
block RAMs (first routed at 53.4 and 50.9 MHz; then 65.5 and 54.5 once
the streamer's issue decision was registered, see "32 lanes on the
ZC706"; these figures are the current design's, with the attention head's
wider score lanes, from "Spec in, verified RTL out"). The bitstream is 4,045,667 bytes, the
XC7Z020's full size, and it round-trips: Project X-Ray decodes it back
to features which, encoded again, give the routed design's frames
exactly, all 7,802 of them.

Two things stood in the way, and both are fixed. nextpnr-xilinx pins
every DSP48E1 that starts no cascade to the lower DSP of its tile, which
leaves 110 sites for this design's 132 lone multipliers, so placement
failed on the same attention multiplier at every seed and with either
placer; `nextpnr-xilinx-dsp.patch`, written by `board_zybo.py`, lets a
lone DSP take either site. Then the bus clock missed 50 MHz by 0.25 ns,
at 49.4 MHz: its worst path was the weight streamer's issue decision, 11
LUTs through the window's two adders. `cur + NS` and `pcur + 1` are now
registered at the same edges as `cur` and `pcur`, so every cycle sees
the same values and no adder is left on the path; the same one-layer
runs through the registers give the same token in the same bus cycles.

What this does not settle: the PS7's own setup (DDR, clocks, the PL
level shifters) is software's job on any Zynq, ps7_init from Vivado or
Digilent's reference design, and nobody has loaded this bitstream on a
board. Vivado's `build.tcl` route remains the one the handoff describes;
this one says the design fits and closes timing on the real part.

## The ZC706, and any Zynq board

```
python3 board_zybo.py --board zc706      # board_zc706/
board_zc706/open/build_open.sh           # its bitstream, without Vivado
```

The package generator takes the board from `boards.py`: its part,
Vivado preset, pins, DDR size and text. Everything else, the RTL, the
registers, the DDR layout and the ARM program, is the same on every
Zynq, whose HP ports and 1 GB of DDR look alike. The ZC706 (XC7Z045-2)
is the second board; its package runs the same open flow on its own
chip database, and the whole design routes there too:

| | Zybo Z7-20 (XC7Z020) | ZC706 (XC7Z045) |
|---|---|---|
| core width | 16 lanes | 32 lanes |
| core clock, post-route | 73.4 MHz | 62.7 MHz |
| bus clock, post-route | 65.4 MHz | 53.9 MHz |
| LUT sites / DSPs used | 28% / 132 of 220 | 9% / 196 of 900 |
| bitstream | 4.0 MB, 7,802 frames round-trip | 13.3 MB, 30,722 frames round-trip |

The Qwen3-0.6B build (`board_zc706_qwen3/`) routes there too, at 69.5 MHz
core and 51.5 MHz bus. Its conversion to a bitstream first aborted inside
Project X-Ray: `fasm2frames` adds a glue bit ten rows above the first
column-62 ground tie it sees, a rule observed on another part, and here
that tile, `INT_L_X62Y354`, is off the XC7Z045's grid.
`prjxray-glue.patch` skips the glue where its tile does not exist. The
bitstream it then writes round-trips, all 30,722 frames. The ZC706's
core is now twice the Zybo's width (next section). Its LEDs sit in banks of more than one voltage, so
its package drives none rather than guess an IOSTANDARD; STATUS says the
same thing over the registers.

## Spec in, verified RTL out: one board or a heterogeneous cluster

```
python3 spec2rtl.py examples/tiny_qwen3.json                                   # one board
python3 spec2rtl.py examples/tiny8_qwen3.json --boards zc706 zybo_z7_20 zc706 zybo_z7_20 zc706 zybo_z7_20 zc706 zybo_z7_20
FPGAI_QWEN=qwen3 python3 spec2rtl.py --weights qwen_weights/qwen3-0.6b --boards zc706 zybo_z7_20 --package
```

`spec2rtl.py` is the whole chain as one command, with `report.md` saying
what checked each part. The input is a model's shape (`model_spec.json`'s
keys) or a checkpoint; with `--boards`, a list of boards in chain order,
any mix and any number, several of one kind included. It refuses a shape
the generator cannot build, with the reason. The planner (`cluster.py`)
splits the layers over the boards by each one's speed and memory, each
board's core at its own width. Every generated block the design compiles,
fifteen of them for a Qwen3-shaped model, goes through the signoff gates
at that shape's own parameters and at every width in the cluster, leaves
first, each composite block's testbench compiling the sub-blocks signed
off before it. The decode step is then generated around exactly those
files. For one board it is simulated against the integer model; for a
cluster, the whole pipeline is, each board on its own clock and sharing
only CRC-checked messages, against the one-board integer model. Every
chosen token and its logit has to match. With `--package` every board's
package follows: its layers, its images, its place in the chain.

| run | blocks signed off | simulated | result |
|---|---|---|---|
| tiny Qwen3 shape, one Zybo | 15 | 2 of 2 layers, 5 positions | tokens and logits exact |
| the same over a ZC706 (32 lanes) and a Zybo (16) | 30, at both widths | the 2-board pipeline | exact, no CRC error |
| tiny 8-layer shape over 8 boards, 32/16/32/16/32/16/32/16 lanes | 30 | the 8-board pipeline | exact, no CRC error |
| Qwen2.5-0.5B, its checkpoint, the Zybo | 14 | 1 of 24 layers, 2 generated tokens | exact; through the registers, stalling DDR: exact, 1.20 bus cycles a core cycle |
| Qwen3-0.6B, its checkpoint, the ZC706 | 15 | 1 of 28 layers, 2 generated tokens | exact; through the registers, stalling DDR: exact, 1.70 bus cycles a core cycle |
| Qwen2.5-0.5B, its checkpoint, the Zybo, in Verilator | 14 | all 24 layers, 2 generated tokens | exact; all 24 through the registers, stalling DDR: exact, 1.19 bus cycles a core cycle |
| Qwen3-0.6B over a ZC706 (layers 0-15) and a Zybo (16-27) | 30 | 2 of 28 layers, one a stage, the head on the Zybo, 2 generated tokens | exact, no CRC error; each stage's cycles within 0.03% of the planner's; both boards' packages from the checkpoint |

Running the gates at the widths the design actually runs found two
faults the earlier checks had passed. Every block used to be signed off
at `model_spec.json`'s 8-bit activations; the sequencer runs 16.

The attention head could overflow its scores. A score is q times k, both
16-bit activations at a16, summed over head_dim, and the score lanes ran
in the model's MAC, whose accumulator is sized for an int8 weight times an
activation: 36 bits for Qwen3-0.6B against 39 for the worst score. Real
prompts stayed inside it, which is why every decode above matched, but a
large enough q.k would have wrapped. The head's testbench, run at 16 bits,
failed on the Qwen3 shape and on the tiny one. The score lanes now run a
MAC of their own, derived by the same rule for q times k over head_dim
(`specgen.derive_score_mac_spec`); where that is no wider than the model's
MAC, as at 8 bits, nothing changes. Because the integer model computes
scores exactly, and every earlier run matched it, no value of any of them
changes: the full Qwen runs, rerun on the new head, take the same cycles
and choose the same tokens. `test_attention_scores_cannot_overflow`
fails on the old width.

The projection's testbench could fail a correct design. It drew weights
over the whole 16-bit data port, where the spec's accumulator is derived
for int8 weights, so at a 256-wide model's 32-bit accumulator a 100-deep
case overflowed a sum no model's weights can reach. It now draws weights
in the spec's `weight_bits`, which the spec records; at 8 bits the
testbench is byte-identical.

The planner's cycle estimate is now the sequencer's own, state by state:
each projection (rows / lanes) x (depth + 5) + 32, each RMSNorm 2D + 28,
each head's scores, softmax and weighted sum from the context length, the
head in chunks. Against simulation:

| | model | simulated |
|---|---|---|
| Qwen2.5-0.5B, 24 layers, a position | 22,807,652 | 22,809,961 |
| its head step | 31,375,800 | 31,375,553 |
| Qwen2.5-0.5B at 32 lanes, 24 layers, a position | 11,577,380 | 11,582,377 |
| Qwen3-0.6B at 32 lanes, 28 layers, a position | 14,513,296 | 14,519,235 |
| Qwen3-0.6B, a layer, a position | 519,331 | 519,546 |
| tiny shape, each of five positions | 14,242 to 17,276 | 14,243 to 17,277 |

within 0.1% on the real models, where counting only weight bytes had been
2 to 6% low on them and half the real figure on a small one. The planner
splits layers by it, at a mid-decode context of 128.

The links now recover from bit errors. A stage takes a message only while
idle and drops one whose CRC fails; the host resends a position whose
answer is lost or corrupt, waiting twice as long each time, and knows a
late answer by its position. A position run twice rewrites its KV entries
with the same values, so a resend is safe. With a bit flipped in every
10,000 on every link, 100 times the proposal's 1e-6, the corrupt message is
caught and resent and the tokens and logits are one board's
(`test_gals_two_boards`).

## Splitting the weights: every board on every layer

```
python3 tp.py --ranks 2
python3 spec2rtl.py examples/tiny_tp4.json --boards zybo_z7_20 zc706 zybo_z7_20 zc706 --split weights
```

The layer split fits a model bigger than any one board, but one token
still crosses every board's layers in turn and goes no faster. Decode
time is set by how fast the weights arrive, so the other split divides
them: every board works on every layer with a slice of every matrix
(`tp.py`, `qwen_full.py`'s `tp`). The slices are by output column, so
every value a board computes is one a single board computes, whole and
requantized, and the boards only gather each other's slices, four times
a layer: the attention context (each board's heads), o's output, the
gated product (each board's share of d_ff) and down's output, and once
at the head, whose vocabulary they also split, to agree on the argmax
(the lowest index on a tie, as the integer model's). Norms, RoPE, the
residual adds and the embedding run on every board on identical vectors.
A board can be at most one gather ahead of another, and a slice it sends
early only lands where the receiver is not working, so no buffering is
needed. Each board reads 1/T of every layer's weights: T boards bring T
times the memory bandwidth to one token.

| ranks | widths | shape | result |
|---|---|---|---|
| 2 | 16/16 | Qwen3's (q/k norms) | tokens and logits exact, 51 gathers |
| 2 | 32/16 | Qwen3's | exact |
| 2 | 16/16 | Qwen2.5's (q/k/v biases) | exact |
| 4 | 16/32/16/32 | Qwen3's, 8 heads over 4 | exact |
| 4 boards through `spec2rtl.py` | Zybo, ZC706, Zybo, ZC706 | tiny_tp4 | 30 blocks signed off, exact, 42 gathers |

`test_weights_split_over_boards` runs the first four and fails on a
network that skips the context's gather. The testbench's network uses
nothing a board lacks: once every rank has asked for the same gather it
reads each rank's slice through that rank's gather port, `gx_addr` and
`gx_rdata` beside `g_req`, a word a clock on that rank's clock, and
writes it into the others' through `gx_wdata`.

### The weight split on the boards

```
python3 spec2rtl.py examples/tiny_tp4.json --boards zybo_z7_20 zc706 zybo_z7_20 zc706 --split weights --bridge --package
```

On a board the network is the ARM's. A rank's package adds six
registers: GADDR and GDATA read the rank's slice of the vector being
gathered and write the others' into it, a word an access, each waiting
for the core's edges as a stage's XDATA does; GMOVE has the PL move a run
of words between the core and a gather buffer in DDR (GBUF) instead;
GATHER says whether the core waits on a gather and which, and a write of
1 lets it go on; TP says which rank of how many the bitstream is. The ARM
program (`board_zybo.TP_C`) has the PL move the rank's slice out, sends
it to every other rank over UDP in parts of at most 600 words, writes
theirs into the buffer as they arrive, has the PL move the whole vector
back in, and at the head every rank takes the same winner. A rank can be one gather
ahead of another and never more, so it keeps parts that come early for
the next gather and its own previous slice for a rank one behind; a rank
still missing parts after 250 ms asks their sender again, and the answer
is the slice once more.

`tp.build_boards` runs every rank as its package does: the register
block and DDR bridge around the core, a DDR model of its own that stalls
at random, its own clock, and every gather done through the registers by
a model of the ARM's side. Only the network between the ARMs is the
testbench's.

| ranks, through their registers | widths | shape | result |
|---|---|---|---|
| 2 | 16/16 | Qwen3's | tokens and logits exact, 51 gathers |
| 2 | 16/32 | Qwen3's | exact |
| 2 | 16/16 | Qwen2.5's | exact |
| 4 | 16/32/16/32 | Qwen3's, 8 heads over 4 | exact |
| 4 boards through `spec2rtl.py` | Zybo, ZC706, Zybo, ZC706 | tiny_tp4 | 30 blocks signed off; exact directly and through the registers, 42 gathers; four packages, each elaborating whole |

The first register block read GATHER as the core's request itself,
which stays up until the core's next edge after the answer. An ARM that
read it again in that window served one gather twice, and the two-rank
run hung after 33 gathers. GATHER now reads clear once answered;
`test_weights_split_through_registers` runs the first two rows, and
uneven shares, with the mover, and again with the ARM moving every word,
and fails on the old block, which the testbench now stops on at the
first gather posted out of step, and on a mover that packs its words in
the wrong lanes.

The program's network half has run on this host
(`test_tp_network_on_host`): three ranks compiled from the generated
`main.c`, talking UDP over localhost through the lwIP shim, each with a
stand-in PL that computes its slice of a fixed integer model and waits
on the same gathers as the core. One rank's second datagram is lost on
the wire and another's fifth arrives corrupt; the ranks ask again and
rank 0 prints the reference's tokens. An ARM that writes the other
ranks' slices one word off fails it.

The cycle model covers a rank (`cluster.tp_rank_cycles`): its own heads
and KV heads, its share of d_ff and of o's and down's columns over their
full depth, the norms and residual adds whole, and its run of the head's
chunks. Against the cycles a rank computes in the direct simulation, not
those it spends waiting on a gather:

| ranks | widths | shape | off by, rank by rank |
|---|---|---|---|
| 2 | 16/16 | Qwen3's | 0.0%, 0.05% |
| 2 | 16/16 | Qwen2.5's | 0.13%, 0.09% |
| 4 | 16/32/16/32 | Qwen3's, head_dim 32 | 0.0%, 0.13%, 0.0%, 0.14% |


### Shares by speed

```
python3 spec2rtl.py examples/tiny_tp4.json --boards zc706 zybo_z7_20 zybo_z7_20 --split weights --bridge --package
```

Even shares waste the faster board of a mixed set: every gather waits for
the slowest rank, so a ZC706 beside a Zybo idles a quarter of each layer.
The shares no longer have to be equal. Each rank holds whole KV heads with
their query heads, and d_ff and d_model columns in multiples of every
rank's lanes; the head's chunks go in contiguous runs in rank order, so a
tie still goes to the lower token (`qwen_full.tp_share`). The planner
(`cluster.tp_plan`) starts from each board's rate and moves a head or a
group of columns at a time while the layer gets shorter, the layer's time
being, gather by gather, the slowest rank's. `--mode even` keeps equal
shares.

For Qwen3-0.6B at a context of 128, from the cycle model and each
package's measured bus cycles a core cycle:

| boards | shares, KV heads / d_ff | a layer, computing | its gathers | a token |
|---|---|---|---|---|
| ZC706 alone | | 18.3 ms | | 0.68 s |
| ZC706 + Zybo, even | 4/4, 1536/1536 | 12.4 ms | 1.7 ms | 0.51 s |
| ZC706 + Zybo, by speed | 5/3, 1760/1312 | 11.1 ms | 1.8 ms | 0.46 s |
| ZC706 + 2 Zybos | 3/3/2, 1248/928/896 | 8.2 ms | 1.7 ms | 0.35 s |
| 4 Zybos | 2 each, 768 each | 6.3 ms | 1.5 ms | 0.28 s |
| 8 Zybos | 1 each, 384 each | 3.2 ms | 1.5 ms | 0.16 s |

Eight KV heads do not split three ways evenly, so three boards had no even
split at all. A gather's words move in the PL: GMOVE's mover takes a rank's
slice out of the core to DDR and the whole vector back in, 5.9 and 2.9 bus
cycles a word, measured through both ranks' registers on Qwen3-0.6B's
shares against the stalling DDR model, about 0.9 ms a layer; the ARM
only copies the buffer. The rest, 200 us of Ethernet latency a gather and
100 MB/s, is an estimate. The first cut had the ARM move every word
through GADDR and GDATA, which a guess of 0.25 us a register access put at
1.8 ms a layer; the mover is measured, and needs no guess.

| run | shares | result |
|---|---|---|
| 4 ranks, 16/32/16/32 lanes, direct | d_ff 32/96/32/96, d_model 16/32/16/64, two ranks with no head chunk | tokens and logits exact |
| 2 ranks, 16/32 lanes, through the registers | d_ff 96/160 | exact, the mover carrying every slice |
| ZC706 + 2 Zybos through `spec2rtl.py` | 1/2/1 KV heads, d_ff 96/96/64, d_model 64/32/32 | 30 blocks signed off; exact directly and through the registers; three packages |
| the ARM program on the host, 3 ranks | slices of 300/500/400, 100/300/200, 600/900/600 words | the reference's tokens after a lost and a corrupt datagram |
| Qwen3-0.6B's checkpoint over a ZC706 and a Zybo through `spec2rtl.py` | 5/3 KV heads, d_ff 1760/1312, d_model 608/416, head chunks 0-28/29-49 | 30 blocks signed off; exact at 1 of 28 layers; each rank's computing cycles within 0.003% of `cluster.tp_rank_cycles`; both packages from the checkpoint |
| the same through `spec2rtl.py --bridge` in Verilator | the same | all 28 layers exact directly, and again through each rank's registers and DDR bridge against stalling DDR (1.69 and 1.13 bus cycles a core cycle), 674 gathers; computing cycles within 0.01% of the cycle model; 37 minutes in all |
| the same two packages' ARM programs on their RTL (`cosim.py`) | the same | "The capital of France is Paris. The capital of the United States is Washington, D.C. The capital", every token and logit the integer model's, 2,256 gathers a rank over UDP, 222 s |

`test_tp_plan` checks the planner's shares on four board sets: they cover
the model, sit in every rank's lanes, tile the vocabulary, and are never
slower than even ones.

The cycle model's projections were 16 cycles short at 32 lanes: the last
group's sums drain through the one requantizer a cycle a lane, 32 at 32
lanes, where the model had a fixed 32 for starting and draining in all.
With that, both real models at 32 lanes are within 0.02% at every
position. What remains is the weighted sum's cost per cached position,
which the model gives as 3 + head_dim / lanes and the block takes as 5 at
short contexts: within 0.2% of a layer at the real models' head_dim of 64
and 128, and 1 to 3% on a tiny head_dim-32 shape at 32 lanes.

## 32 lanes on the ZC706

```
python3 qwen_full.py --lanes 32 --work build_q25l32     # all 24 layers, direct
python3 board_zybo.py --board zc706 --work build_q25l32 --sim --jitter
FPGAI_QWEN=qwen3 python3 qwen_full.py --lanes 32 --work build_q3l32
```

The core's width is now a build option, `qwen_full.py --lanes`, and each
package in `boards.py` names its own: 16 on the Zybo, whose XC7Z020 has
220 DSPs, and 32 on the ZC706, whose XC7Z045 has 900. Everything that
depended on 16 is derived from it: the weight images, the KV cache's
layout, the embedding lookup, the bridge's lines and bursts, and the
DDR model. At 16 lanes, rebuilding both real models gives byte-identical
RTL, blocks, weights, constants and gains. At 32 the DDR layout, the
ARM's header and the register block are unchanged too, since the same
bytes sit at the same addresses, grouped into words twice as wide;
`board_zybo.py` refuses a build whose width is not its board's.

On the direct testbench, all layers and the head, the prompt's five
positions:

| | 16 lanes | 32 lanes |
|---|---|---|
| Qwen2.5-0.5B, a position | 22.81 M core cycles | 11.58 M |
| Qwen2.5-0.5B, the head step | 31.38 M | 15.87 M |
| Qwen3-0.6B, a position | 28.3 M | 14.52 M |
| Qwen3-0.6B, the head step | 38.1 M | 19.42 M |
| token chosen | " Paris" | " Paris", both models |

Through the ZC706 package's registers, one layer of each model chooses
the direct testbench's token in its core cycles, against the DDR model
always ready and stalling at random. The bus is the limit now: 32 lanes
take 32 bytes a core cycle, all four 64-bit HP ports carry at best,
where 16 lanes took half. One Qwen2.5 layer, five steps, takes 1.35 bus
cycles a core cycle with the DDR model always ready and 1.70 with it
stalling, where 16 lanes take 1.17 and 1.20. Part of the first is the DDR model's
own: it leaves a cycle idle after every burst, which caps a port at
8/9 of its beats. With bursts back to back, as a real HP port can
deliver them, a position takes 1.22 bus cycles a core cycle, and a
larger streamer window changes nothing (1.22). So 32 lanes buy 1.7
times the tokens a second on a well-behaved bus and 1.4 times on the
stalling one, not 2. The whole of Qwen2.5 through the ZC706's registers,
against the stalling DDR model, was still simulating when this was
written: its first three positions took 19.57 to 19.68 million bus
cycles for the direct run's core cycles, 1.69 a core cycle, which is
0.39 s a position at 50 MHz against the Zybo's 0.54 s.

The routed design uses 196 of the XC7Z045's 900 DSPs (198 for Qwen3) and
9% of its LUT sites (10%), and closes 50 MHz on both clocks in the open
flow: 65.5 MHz core and 50.3 MHz bus for Qwen2.5, 71.5 and 60.3 for
Qwen3 (62.7 and 53.9, 69.5 and 51.5 since the attention head's score
lanes were widened). Every bitstream round-trips, all 30,722 frames.

### Two bridge faults, found on the way

The weight image was in the wrong byte order for the board. The ARM
copies `weights8.bin` into DDR byte for byte, and AXI is little-endian:
the byte at a word's lowest address arrives on `rdata[7:0]`, which the
bridge takes as lane 0. The image held each word lane 15 first, and the
testbench's DDR model handed bytes back the other way round, so the two
agreed with each other and not with the board, where every word's
lanes would have landed reversed. The image now holds lane j at byte j,
the DDR model returns the bytes at a beat's address lowest first, and
`test_bridge_end_to_end` checks the image against the model's own
weights and fails on an image in the old order. An SD card made before
this has the old image; its sizes are the same, so the layout check
cannot tell, and `HANDOFF.md` says so.

The weight streamer could hand the core another line's weights. It
allowed a second request to a slot while the first was still in flight.
After the stream moved on and then jumped back, one slot could have
three bursts on their way, X, then Y, then X again; X's first copy
marked the slot valid for X, and Y's beats then overwrote it while it
still read as X. The stress test that guards the streamer ran 120 jumps
and passed; with 1500 it fails on every one of 24 seeds, 4 to 31 wrong
words in 88,000. A real decode step has only a few jumps, which is why
every model run above still chose the right tokens, but a board's DDR
timing could have hit it on any of them. A slot now takes no request
while a burst is in flight to it: the stream moves past a line that is
held or on its way, and waits for the slot otherwise. Every seed tried
since passes at 1500 jumps, the test now runs two, and with the DDR model
always ready the one-layer runs take exactly the bus cycles they did
before, on both boards.

The 32-lane build first closed only 47.3 MHz on the bus clock (Qwen3:
46.4). Its worst path was the stream's issue decision: a 32-way mux of
line tags and a compare against the next line, then every tag's enable,
ten LUTs, most of it wire on a die the wider core spreads over. The
decision now reads three registered flags, computed a cycle ahead for
the line the stream moves to, which is exact for the tag and the
held-or-coming flag; the in-flight flag can lag a landing burst by a
cycle, which delays an issue and never lets one into a busy slot. The
Zybo's bus clock went from 50.9 to 54.5 MHz with it.

A pipeline need not be one kind of board. `gals.build` takes each
stage's width, and three boards of 32, 16 and 32 lanes give the one-board
integer model's tokens and logits exactly, with no CRC error: only the
hidden state crosses between them.

## The Zybo at 32 lanes

```
python3 board_zybo.py --board zybo_z7_20_32 --work build_q25l32
board_zybo_32/open/build_open.sh
python3 cosim.py board_zybo_32 --jitter
```

The Zybo's XC7Z020 holds the ZC706's 32-lane core too: 38% of its LUT
sites and 196 of its 220 DSPs. In the open flow it placed and did not
route. With 89% of the DSPs used, nextpnr-xilinx found no free LUT beside
eleven of them to make the constant zeros their unused C inputs need,
all on the attention head's score multipliers. With 16 of the 32 score
lanes multiplying in LUTs, three were left, all in the DSP column beside
the PS; with 24, none. The package's open flow (`board_zybo_32/`, board
`zybo_z7_20_32`) retypes those 48 multipliers just before Yosys's DSP
mapping; Vivado's build keeps every multiplier on a DSP.

| | 32 lanes on the XC7Z020 |
|---|---|
| LUT sites / DSPs / block RAM | 50% / 148 of 220 / 26 RAMB36 and 23 RAMB18 |
| core, post-route | 57.8 MHz, passes 50 |
| bus and registers, post-route | 58.5 MHz, passes 50 |
| bitstream | 4,045,667 bytes, all 7,802 frames round-trip |
| its own ARM program on its RTL, its own SD card, every port stalling (`cosim.py`) | "The capital of France is Paris. Paris is the capital of France. Paris is the capital of France.", every logit the integer model's |

Its RTL is the ZC706 package's byte for byte, so it is verified as that
one is. On the same four HP ports at the same 50 MHz it runs at the
ZC706's 1.70 bus cycles a core cycle, and the planner (board
`zybo_z7_20_32`) puts it at the ZC706's speed: Qwen3-0.6B 0.68 s a token
against 0.92 s at 16 lanes, Qwen2.5-0.5B 0.55 s against 0.75 s. Two of
them split by weights come to 0.39 s a token, 2.55 tokens/s, for about
\$600; eight to 0.13 s. Against Redwood's 12.1 tokens/s on a \$17,995
board, one 32-lane Zybo is about 7 times the tokens a second per dollar
and two about 6, from the cycle model, the measured bus ratio and the
mover, with the Ethernet's latency estimated; the proposal's 40 tokens/s
on two boards is out of reach at int8, where two Zybos' DDR3 streams the
model's 600 MB about 10 times a second at most.

## Every layer of a real model, in minutes

Icarus took two and a half hours for Qwen3-0.6B's 28 layers and the head
over five positions, so `spec2rtl.py` simulated a real checkpoint at one
layer. Verilator compiles the same testbenches unchanged, delays, forks
and hierarchical references included (`vsim.py`), and runs that one in
two minutes: 19,418,465 cycles for the head step, token 12095 at logit
15025, exactly as Icarus. `spec2rtl.py` now uses it where it is
installed (`--sim fast`, the default) and then simulates every layer of
a real model, and every layer again through the registers with
`--bridge`; `--sim iverilog` keeps the old behaviour. Every test in
`tests.py` still runs on Icarus, which is four-state, so a word never
written reads as x; `test_simulators_agree` holds Verilator to Icarus's
steps, tokens, logits and cycles on a small checkpoint and on a weight
split's ranks. The layer split's bit-error test stays on Icarus: the two
draw different random streams, and Verilator's flipped no bit on that
run.

| run | Icarus | Verilator |
|---|---|---|
| Qwen3-0.6B at 32 lanes, 28 layers and the head, five positions, direct | 2.5 hours | 2 minutes: the same cycles, token and logit |
| a weight split's two ranks, and its two ranks through their registers | 14 s and 24 s | under a second each: the same tokens and logits |
| `spec2rtl.py` on Qwen3-0.6B split by weights over a ZC706 and a Zybo, `--bridge` | 1 layer | all 28 layers on both ranks, then all 28 through each rank's registers, 37 minutes in all |
| `spec2rtl.py --weights qwen_weights --bridge`, Qwen2.5-0.5B on the Zybo | 1 layer, then 1 layer through the registers | all 24 layers and the head, every position within 0.01% of the cycle model, then all 24 through the registers against stalling DDR at 1.19 bus cycles a core cycle: " Paris" and "." with the integer model's logits, 19 minutes in all |

## Half the bytes: int4 weights, measured and not taken

At 32 lanes the core waits on its four HP ports, so weights of half the
width would carry twice as many a beat. Whether the model survives them
decides it; the integer model with every layer's weights rounded to 4
bits (the tied table kept at 8), the same teacher-forced check as above:

| weights | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|
| int8 per channel (this design) | 14/16, "Paris. Paris is" | 13/16, "Paris. The capital of the" |
| int4 per channel | 11/16, "18,000." | 5/16, "the capital of the world" |
| int4, a scale per 64 columns | 14/16, "located in the city of Paris." | 11/16, "12,000," |
| int4, a scale per 32 columns | | 8/16, "the capital city of the country" |
| int4, a scale per 16 columns | | 9/16, "Paris, and the capital of England" |

Qwen2.5 keeps its answer at int4 in groups of 64; Qwen3, the proposal's
model, does not keep it reliably at any group size, as its massive
activations (above) would suggest. Round-to-nearest is the simplest
scheme; one calibrated on activations (GPTQ, AWQ) might hold Qwen3, and
is where halving the bytes would start. The design stays at int8.

## The ARM programs on their own RTL

```
python3 cosim.py board_zc706 --jitter
FPGAI_QWEN=qwen3 python3 cosim.py board_zc706_qwen3 --jitter
```

Until now each half of a board ran against a model of the other: the RTL
against a Verilog model of the ARM's register accesses, and the ARM
programs against a stand-in PL written in C. `cosim.py` joins the real
halves. It compiles a package's `sw/main.c`, unchanged, for this host
and links it with the package's own RTL built by Verilator: the register
block, the DDR bridge and the core. Every `Xil_In32` and `Xil_Out32` is
an AXI-Lite transaction on that RTL. DDR is one array: the program's
loads from the SD card write into it at the header's bus addresses, and
the PL's five AXI masters read and write the same bytes, answered after
30 cycles and, with `--jitter`, stalling and gapping at random. So the
weight image's byte order, the constants' format, the KV cache's
clearing and the vocabulary all cross the path they will on a Zynq, and
the program prints what its UART will. Beside the program's own read of
each step's token, the harness reads the logit too, so a run is checked
to the logit. The stand-ins left are FatFs (a directory), lwIP (a UDP
socket on localhost) and coherent caches. Verilator runs the ZC706's
core at about two million bus cycles a second, so a real model's whole
run takes minutes.

The three packages in this repo, each with its SD card from the real
checkpoint, every port stalling at random, the prompt "The capital of
France is" and the program's 16 tokens:

| package | the program printed | 16 head steps against the integer model | bus cycles | wall time |
|---|---|---|---|---|
| `board_zybo/`, Qwen2.5-0.5B, 16 lanes | The capital of France is Paris. Paris is the capital of France. Paris is the capital of France. | every token and logit | 710 million | 6.6 min |
| `board_zc706/`, Qwen2.5-0.5B, 32 lanes | the same | every token and logit | 476 million | 3.9 min |
| `board_zc706_qwen3/`, Qwen3-0.6B, 32 lanes | The capital of France is Paris. The capital of the United States is Washington, D.C. The capital | every token and logit | 608 million | 5.0 min |

That is the whole of what a board would print, where simulation had
checked the first generated token and the ARM programs had never run on
the RTL at all. Both Qwen2.5 packages print the continuation `HANDOFF.md`
tells the team to expect. Qwen3-0.6B split by weights over a ZC706 and a
Zybo, its shares by speed, runs the same way as two processes, each rank's
program on its own package's RTL, gathering over UDP: rank 0 prints the
single board's text, and the logit the ranks agree on at every step is
the integer model's (2,256 gathers a rank, 222 s; 216 s once the PL
moves the slices). Split by layers instead, over the same two boards
(`spec2rtl.py`, every layer in Verilator: the ZC706 layers 0 to 15, the
Zybo 16 to 27 and the head, stage cycles within 0.011% of the planner's),
both stages' programs on their RTL print the same text, the last stage's
logits the integer model's, in 7 minutes.

The harness's clock is the PL's: `XTime` reads simulated nanoseconds,
the bus cycles so far, so a program's timeouts mean what they mean on a
board, and the step times it prints are the board's less the ARM's own
work and the Ethernet's latency, which localhost does not have. The first
two-stage run kept wall time, and the first stage resent every hidden
state after five seconds of it while the Zybo's twelve layers were still
being simulated. For the planner's cycle model, which the proposal holds
to 15%, these are the closest thing to a measurement this host can make:

| run | the program's last step | the planner, at that context |
|---|---|---|
| `board_zc706/`, Qwen2.5-0.5B alone | 25,499,870 bus cycles, 0.51 s | 0.54 s |
| Qwen3-0.6B split by weights over a ZC706 and a Zybo | 0.465 s | 0.425 s, less the Ethernet's latency |

`test_arm_programs_on_their_rtl` runs all three programs on small
checkpoints and checks tokens and logits against the integer model: the
single board's with every port stalling; two stages of a layer split with
the first datagram lost; and two ranks of a weight split, a Zybo's 16
lanes beside a ZC706's 32 with uneven shares, a datagram lost. A weight
image in the byte order the ARM once had fails it, and so do ranks whose
ARM writes the others' slices one word off, which moved no token of that
small checkpoint and every logit.

## Boards on their own clocks: GALS

```
python3 cluster.py zc706 zybo_z7_20           # the split and the links
python3 gals.py                                # stages over fabric UARTs, simulated
FPGAI_QWEN=qwen3 python3 cluster.py zc706 zybo_z7_20 --package build_cluster
```

Several boards share one model as a pipeline of stages, globally
asynchronous and locally synchronous: each board runs its own clock and a
contiguous range of layers, and boards share nothing but CRC-checked
messages, the hidden state down the chain and the chosen token back. The
sequencer generator builds a stage on request: the layers it holds, the
vocabulary table only on the first and last stage, and a port to load
the hidden state it starts from and read back the one it ends with.
`cluster.py` splits the layers by each board's speed (the smaller of its
DDR rate and what the design consumes at its clock) and capacity, and
picks each link: UDP over the ARM's Ethernet between two Zynq boards,
otherwise a UART in the fabric on two pins, which any FPGA can build.

The fabric link is simulated whole. `gals.py` puts each stage's
sequencer beside `stage_ctrl`, a UART, a byte-wise CRC32 and a message
parser, and runs the stages on unrelated clocks, 10, 7.9 and 12.3 ns,
whose UART bit times differ by up to 1.25%, with a host at its own rate
feeding the prompt and every generated token back. With two boards and
with three, the middle one holding no vocabulary table, every token and
logit is the one-board integer model's and no link sees a CRC error; a
link that sends the hidden state's bytes swapped is caught
(`test_gals_two_boards`). On the Zynq side, the stage's register block
gains XADDR and XDATA, whose accesses wait for the core's own edges
since its clock is held while lines fill (`test_zybo_stage_registers`,
which fails on a write that does not wait), and the ARM program carries
the same messages over lwIP UDP, a hidden state split into datagrams
under one frame.

For Qwen3-0.6B over the ZC706 and a Zybo, the balanced split was layers
0 to 13 on the ZC706 with the embedding and 14 to 27 on the Zybo with
the head, both at the 50 MHz bus clock: 0.88 s a token for one stream,
the same as one board, since a single stream's time is the sum of its
stages, and 1.8 tokens/s with both stages busy on separate streams. With
the ZC706's core at 32 lanes the planner gives it layers 0 to 15: one
stream then takes 0.78 s, slower than the ZC706 alone (0.63 s), and both
stages busy give 1.97 tokens/s, which is the split's point.
`--package` wrote both boards' packages from the real weights, each
elaborating whole. The network half of the ARM
program has run too, on this host: `test_stage_network_on_host` compiles
the generated program twice, a first and a last stage, links them over
localhost UDP through a small shim for lwIP with a stand-in PL whose
layers are a fixed integer map, and drops the first stage's first
datagram on the wire. The first stage sends again after its timeout,
the hidden state crosses in two parts, and the printed tokens are the
reference's; a program that sends the hidden state's bytes swapped fails
it. Resending is safe because running a position twice writes its KV
cache entries again with the same values. What is not run: real lwIP on
a Zynq, and a board without an ARM, which cannot yet hold a layer,
having no DRAM this design reaches.

## Both directions of every link

The fabric models each connection as full duplex, two one-way links
(`fabric.connect`), but the ring all-reduce the sizing layer charges sent
one way round: on three boards or more, each link's other direction
carried only acknowledgements. `ring_allreduce(boards, data, bidir=True)`
sends half the vector each way at once, the same 2(n-1) steps each
serializing half as much. On two boards one ring already uses both
directions of the only link, so it stays one ring and takes the same
time. The model spec's `ring_bidirectional` turns it on in
`sizing.predict_config` and `simulate_decode`. It is off by default, so
every number in this file is unchanged, and the one-way ring's times are
the same to the nanosecond as before.

An all-reduce on KC705-class boards over 10 Gb/s links, in ns:

| boards | 8 KB one way | both ways | | 512 KB one way | both ways | |
|---|---|---|---|---|---|---|
| 2 | 9,335 | 9,335 | 1.00× | 484,669 | 484,669 | 1.00× |
| 3 | 13,284 | 8,118 | 1.64× | 601,791 | 342,619 | 1.76× |
| 4 | 16,081 | 10,888 | 1.48× | 681,227 | 378,116 | 1.80× |
| 8 | 23,612 | 16,982 | 1.39× | 813,952 | 435,963 | 1.87× |

Small vectors gain less: each step still pays the link's latency once.
Decode on the large class (Alveo U250, the Qwen2.5-0.5B spec at batch 8),
in tokens/s, predicted and then simulated on the fabric:

| boards | one way | both ways |
|---|---|---|
| 4 | 1,368 / 1,250 | 1,881 / 1,735 |
| 8 | 1,622 / 1,423 | 2,592 / 2,306 |
| 16 | 1,524 / 1,337 | 2,470 / 2,140 |

The peak stays at eight boards and rises 1.62× on the fabric, and the
prediction is as close as it was one way, within 13%.
`test_bidirectional_ring` holds it exact on two to eight boards and a
mixed three, the same on two boards, 1.6× faster on 512 KB over four,
and the 8-board decode within 15% of its prediction.

This is the sizing layer's fabric. The Zynq packages' weight split does
not ring: each rank sends its slice to every other rank over UDP through
a switch, which already sends and receives on every port at once.

## Qwen3-0.6B, the model Architect Labs hosted

```
python3 fetch_qwen.py --model qwen3           # ~1.5 GB, git-ignored
FPGAI_QWEN=qwen3 python3 qwen_real.py         # float
FPGAI_QWEN=qwen3 python3 qwen_int.py --act-bits 16 --per-channel
```

Everything built for Qwen2.5 takes the other checkpoint through one
variable. Qwen3's structure differs in three places the generator now
handles: no q, k or v biases; a head dimension of its own, 128, so q is
2048 wide over a 1024 hidden state; and an RMSNorm over every head of q
and k before RoPE, which the integer model runs through the same norm
block sized for a head and the sequencer through a second instance of
it, normalising each head in place. `test_full_sequencer_both_qwens`
generates and simulates the sequencer on small random checkpoints with
each structure and requires every chosen token and its logit to be the
integer model's; a Qwen3 sequencer that skips the head norms fails it
even though its tokens do not change. Qwen2.5's generated RTL and images
are byte-identical to before.

| model | float | integer (w8 per-channel, a16) | teacher-forced |
|---|---|---|---|
| Qwen2.5-0.5B | "Paris. It is the" | "Paris. Paris is" | 14/16 |
| Qwen3-0.6B | "Paris. The capital of Italy" | "Paris. The capital of the" | 13/16 |

Qwen3 is harder on static integer scales than Qwen2.5 because of its
massive activations. Over the prompt, the residual stream reaches 6,500
against a median of 1.7, and in layer 2 the MLP product reaches 3,658
against a median of 0.065, a ratio past what 16 bits hold. Holding that
product in 32 bits did not help, 5/8 against 6/8 on the first eight
positions, so it is not the limit; of the three misses in sixteen, one
is a near tie in float (0.12), one is the first position, where float's
own top two are 0.07 apart, and one, position 7, is a real miss.

In RTL, the Qwen3 sequencer runs the real weights one layer deep and
chooses the integer model's token, the head norms and all 151,936 head
columns included. The first attempt hung in the head: its chunk was a
fixed 4864 columns, Qwen2.5's MLP width, and Qwen3's projection block,
sized for its 3072, has a 12-bit column port, so it ran 768 columns while
the sequencer waited for 4864. The chunk is now the widest matrix the
model's projection block is sized for, with a last chunk that is never
empty; the synthetic test's vocabulary now spans several chunks, and the
testbench prints a heartbeat, so a hang reads as one. Qwen2.5's
generated RTL is unchanged by it.

The whole Qwen3-0.6B then decodes in RTL: all 28 layers and the full
head on its own weights, the prompt's five positions, choosing token
12095, " Paris", as the integer model does. A prompt position takes
28.3 million core cycles and a step with the head 38.1 million; through
the ZC706 package's registers against the stalling DDR model, one layer
of the same design matches the direct testbench's token and core cycles
at 1.19 bus cycles a core cycle, so a Qwen3 token is about 0.9 s at the
50 MHz the design closes (0.63 s at the ZC706's later 32 lanes). That board-level run first read back a token
of 0: the DDR model had no weight image to serve, since the package's
simulation path relied on one a different script writes, and every
weight read returned X while the data-independent timing still looked
right. The simulation now writes its own image and stops if one is
missing.

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
| multi-lane attention | not yet attempted | | |
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

## Every block of a decoder by the LLM agents

`spec2rtl.py --agent llm` on a two-layer checkpoint of Qwen3's shape
(`examples/tiny_qwen3.json`) takes all fifteen blocks through the agent
chain, Haiku with eight drafts, then Sonnet, then the rules agent, and
then simulates the decode step they make against the integer model. The
fourth such run, the first after the rename fix, took about ten hours:

| block | signed off by | iterations | checks | OpenSTA |
|---|---|---|---|---|
| multiply-accumulate | Haiku | 4 | 613 | 139 MHz |
| requantizer | Sonnet | 5 | 220 | 112 MHz |
| exponential | Haiku | 2 | 153 | 169 MHz |
| reciprocal | Haiku | 1 | 200 | 228 MHz |
| inverse square root | Haiku | 2 | 248 | 249 MHz |
| matrix-vector sequencer | Haiku | 1 | 11 | 167 MHz |
| softmax | rules, after both models | 2 | 132 | 124 MHz |
| score multiply-accumulate | Haiku | 5 | 613 | 107 MHz |
| projection | rules, Sonnet timed out | 2 | 252 | 113 MHz |
| attention head | rules, Sonnet timed out | 2 | 236 | 107 MHz |
| RMSNorm | rules, after both models | 2 | 268 | 106 MHz |
| RMSNorm over a head | rules, after both models | 2 | 140 | 106 MHz |
| rotary embedding | Sonnet | 4 | 1004 | 115 MHz |
| SiLU | Haiku | 6 | 318 | 110 MHz |
| residual add | Haiku | 5 | 218 | 102 MHz |

Ten of the fifteen are the models' own. The attention head is the block
the third run lost: an agent that wrote `module mac(` left two modules
named mac, and nothing could compile the head, the rules agent included.
It now signs off. Sonnet ran into the 25 minute limit three times on the
projection and three times on the attention head; on softmax and the two
RMSNorms both models answered and ran out of drafts.

Every block passed its gates, and the decode step they made did not
match: tokens 504 and 336 where the integer model has 49 and 49.
Swapping each of the ten model-written blocks alone into the rules
agent's design, which matches, found one: Haiku's residual add. It held
scale_a and scale_b in signed registers. They are unsigned 18-bit
magnitudes, and a scale with its top bit set, as the real design's
scales have, became negative. The testbench ran one stream, at scales
from 512 to 2047, so it passed all 218 checks.

The residual add's testbench now runs three more streams after the
profiled one, each at its own held scales: the largest scale beside one
with only its top bit set, both at the largest with the sum saturating,
and a shift of zero. Haiku's block fails at the first of them, the scales
in the message, and so does a rules design mutated to read its scales as
signed; the profile, from the first stream alone, is unchanged, and the
spec now says the scales' top bit is magnitude.

Signed off again against that testbench, Haiku's drafts passed all 305
checks three times and missed timing at 100 MHz, by 0.4 ns at best;
Sonnet signed the block off in five. With it in place of Haiku's first
one and the other nine model-written blocks as they were, the decode
step matches the integer model: tokens 49 and 49, logits 14065 and 14504.

### Why Sonnet ran out of time on the attention head

Sonnet's "timed out" on the attention head was not a hung call. Streamed,
the CLI shows it thinking from the second second on, at low effort:
13 kB of thinking in three minutes and not a character of answer, 128 kB
after 36 minutes, when the call ended on the CLI's 32,000 output token
limit, which its thinking counts against. It could not have answered at
any timeout. Read, the thinking is about the spec: the multi-lane head's
spec had been built by appending its lane sentences to the one-lane
head's, so it said the caches were row-major and, a few sentences on,
that each key word holds one dimension of sixteen positions; and that
matvec with cols = n gives the key address, which is true of one lane
only. Sonnet kept returning to "cols must actually be n_groups". The
rules agent, which reads no prose, never noticed.

The lanes' spec now rewrites those sentences instead: the word layout,
matvec as optional, and if it runs the score pass, cols = ceil(n / 16),
with lane l holding position g*16 + l and only positions below n written
to sbuf; a test holds it to one layout and one column count. The CLI is
now read as a stream of events, so a call that goes silent for three
minutes is retried as hung, and a call still streaming at its cap fails
that model's attempt at once instead of asking the same question twice
more, 75 minutes a block. And each agent that does not converge now
keeps its report and last draft under its own name: the rules agent
used to write over both, and run 4 kept no record of why the models had
failed softmax, the projection and the two RMSNorms.

### What the kept attempts showed

Run 5 put the five fallback blocks through the chain again with the
attempts kept. The projection's thirteen drafts, eight from Haiku and
five from Sonnet, none compiled, and every one for the same reason:
the spec said "the supplied mac module" and "the supplied requant
module" and never gave their ports, so each draft guessed them
(enable, p, result, o, q; acc, sum, in). The attention head's spec had
been given its sub-blocks' exact port lists after the same trace; the
projection's had not. It now gives both, with the MAC's and the
requantizer's latencies, and a test requires every composite built from
supplied modules to give their ports.

Softmax's showed two gaps in the feedback. Haiku's drafts failed to
compile on a select taken of an expression, (s_buf[i] - mx)[12:0], and
on a variable declared inside an unnamed begin ... end; both are now
explained with the rule. Sonnet's gave 16383 for a two-score row, the
weight of a row whose scores are all equal, five drafts running, and
read only "expected 14668 got 16383"; the testbench now says when a
weight is exactly that.

A block's own testbench is where a model's reading of the spec is
checked, and a gap there lets a wrong block through every gate. The
decode step against the integer model is what caught this one, which is
why `spec2rtl.py` runs it after the blocks and fails the run when it
differs.

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

The testbench first ran one stream, at scales from 512 to 2047. An LLM's
residual add that read the scales as signed passed it and broke the
decode step it sat in (see "Every block of a decoder by the LLM
agents"); three more streams now follow, at the largest scales and a
shift of zero.

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
15.5 million cycles, 7.4 tokens/s at 115 MHz, so the weight matmuls no
longer bind: the DDR's 4 tokens/s does. The one-lane attention head was
then the slowest block, and it is widened next.

## Attention on lanes

The multi-lane head runs 32 lanes on both of its matrix halves. For the
scores, each lane is one cached position, an instance of the generated
MAC reading its own lane of a key word that holds element d of 32
consecutive positions; the sums drain through the one-lane head's score
quantizer into the same score buffer. The softmax between the halves
stays one lane. For the weighted sum each lane is one output dimension,
reading its lane of a value word that holds 32 elements of one
position's row, and the sums drain through the requantizer. On a full
256-position row it matches the one-lane head bit for bit, its score and
weight buffers checked directly as well as its outputs, in 1927 cycles
against 34,851: 18 times fewer. 112 MHz, 6953 LUTs and 74 DSPs. The rules
agent's first cut reads each position's key from the mirrored lane and
fails at the score buffer. All six model variants converge and kill every
applicable mutant. The first sweep of it reported one mutant as unproven
on every variant, and the hole was in the tooling: the adder mutant
flipped the first ' + ' in the file, which here was in a comment, so it
changed nothing and the prover timed out on a design that large. It now
edits code only; SiLU had the same comment ahead of its first addition,
and both blocks now kill a mutant on a real add.

With both halves widened, a Qwen2.5-0.5B token at 256 positions adds up,
from measured block cycles, to about 16.5 million cycles: 15.5 million in
the projections, 0.65 million in 336 heads and about 0.3 million in the
norms, adds and activations. That is 6.8 tokens/s of compute at 112 MHz,
above the 4 the Zybo's DDR can feed, so one board is bandwidth bound, as
the sizing model assumes. Two limits stand: the head holds 256 positions,
the softmax's row tile, so Qwen's 1024-token context needs row tiling
that does not exist, and those cycles are summed per block rather than
run as one sequence at that size.

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

1. **Nothing has run on hardware.** No device has been configured and
   clocked, so every timing number is from a timing model rather than
   from silicon in operation. The Zynq parts are no longer out of reach:
   the open 7-series flow (nextpnr-xilinx on Project X-Ray, with one
   patch in `open/`) places and routes the whole design on the XC7Z020
   and the XC7Z045 at 50 MHz and writes bitstreams that round-trip. The
   team has a ZC706, so what remains is the board itself, `HANDOFF.md`'s
   steps: Vivado's build or the open-flow bitstream with a ps7_init, the
   Vitis program, the SD card, and a UART log. `board_basys3/` is the
   same for the Basys 3, with the small Qwen-shaped decoder.
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
   size `qwen_full.py` sequences the blocks through a whole decode step,
   every layer and the head, with weights streamed from DDR. Qwen's MLP is the gated block,
   down(SiLU(gate(x)) * up(x)); the older MLP block, two matmuls with a
   ReLU, is not Qwen's and remains as the simpler case. The composite
   layer blocks still hold 64-entry activation banks; only the
   projection block streams activations from memory at full size, and
   the attention and MLP blocks have not been moved onto it. In `generate.py` those run on the host, and the
   output says so each run.
3. **The real model has run in simulation, not on silicon.** A generated
   sequencer decodes Qwen2.5-0.5B's own weights in iverilog, all 24
   layers and the full head, and chooses " Paris" as the integer model
   does; that integer model agrees with float on 14 of 16 positions at
   int16 activations, and Qwen3-0.6B's on 13 of 16. The weights reach it through AXI masters shaped
   like the Zynq's HP ports, against a DDR model that stalls and gaps at
   random, and the packages' own ARM programs have driven it through
   their registers from their own SD images (`cosim.py`); the Zynq's own
   DDR controller, interconnect and Cortex-A9 have not.
4. **Throughput is simulated, not measured on a board.** The decode's
   own cycles are exact: through the Zybo package's registers against
   the stalling DDR model, a Qwen2.5-0.5B position takes 26.9 million bus
   cycles and a step with the head 37.5 million, 0.54 s and 0.75 s at
   the 50 MHz the design closes: a generated token every 0.75 s, about
   1.3 tokens/s for one stream. On the ZC706's 32-lane core a position
   takes 19.6 million bus cycles through the same registers, 0.39 s.
   `cluster.py`, from the measured
   bus ratios, puts Qwen3-0.6B at 0.63 s a token on the ZC706. The real
   DDR controller, the interconnect and the ARM have not run it, and
   the 32-lane core is bound by the four HP ports at the shared 50 MHz
   clock. The fabric-level tokens/s of the sizing model remains a model
   checked against a fabric simulation.

5. **Formal coverage is one property of one block.** The accumulator
   proof holds for the int8 targets; with 16-bit operands the MAC splits
   its multiplier, the proof becomes multiplier equivalence, and z3 runs
   out of time. Those variants rest on the closed-form bound, the
   testbenches and mutation testing. No other block has a formal proof.

6. **The scale-out's links stop at simulation.** The layer split's
   links are fabric UARTs in simulation (6.25 Mbaud) and, between Zynq
   boards, UDP over the ARM's Ethernet, run on a host with lwIP shimmed;
   the weight split's gathers run through each rank's registers in
   simulation and over UDP on a host with lwIP shimmed, both programs
   against their packages' own RTL; neither has run on a Zynq's ARM or
   its Ethernet. The generated
   CRC32 fabric endpoint, signed off at 10G and beyond as a block, is not
   in the board-to-board path. A board without an ARM holds no layer: it
   has no DRAM the design reaches, since no fabric DDR controller is
   generated.

Item 1 is the one that would let this claim what Architect Labs
demonstrated, and it is no longer blocked on tooling: the board is on
hand and the package for it is written. Items 2 to 4 are ordinary
remaining work.
