# Agentic FPGA Scaleout: Given an LLM, Generate and Synthesize the Fabric to Host It (Prototype)

The input is an LLM: a model spec (GPT-2 124M here) with a target token rate. The hardware is derived from how the model computes: the MAC chiplet's datapath width comes from the model's quantization and its accumulator width from the longest dot-product reduction in the network, and agents generate that chiplet and the fabric endpoint that carries every packet as real RTL through real tools with no human in the loop, both passing simulation, synthesis, and timing signoff and emitting measured performance profiles. A sizing layer then derives the right fabric for the model, the smallest cluster of a given FPGA class that hosts it at the target rate, modeling compute, DDR bandwidth, on-chip SRAM residency of the weight shards, KV cache traffic, and ring all-reduce cost, and validates the choice by running decode on a discrete-event fabric where every all-reduce is packetized, CRC-checked, credit-gated traffic through the synthesized endpoint's measured rate. Scaling is adding more FPGAs running the same synthesized fabric: the same profiles fit any board class, clusters may mix classes, and the numerics stay bit-exact at any board count, even on lossy links.

Pure Python 3 stdlib, no dependencies. The RTL half runs real open tools (Icarus Verilog, Yosys, OpenSTA); the scaleout half is a discrete-event simulator in nanoseconds on a `heapq` event queue.

## How it fits together

```
                       model_spec.json
        (n_layer, d_model, d_ff, weight_bits, activation_bits,
         seq_len, target_tokens_per_s)        one file you edit
                             |
                             v
                    +-----------------+
                    |   specgen.py    |   derives every block's spec AND
                    +-----------------+   its self-checking testbench.
                             |            Golden values are computed in
                             |            Python, so a design can never
                             |            mark its own work.
                             v
                    +-----------------+
                    |      agent      |   writes the Verilog
                    +-----------------+
                     |              |
              --agent rules   --agent llm / swarm
              deterministic,  a real model, re-prompted
              free, offline   with parsed tool feedback
                             |
                             v
        +-----------------------------------------+
        |  iverilog  ->  yosys  ->  OpenSTA        |   the gates
        |     ->  synth_xilinx  ->  mutation DV    |
        +-----------------------------------------+
                             |
              fails ---------+--------- passes
                |                            |
        parsed feedback                      v
        back to the agent       signed-off RTL + measured profile
                                             |
                                             v
                                    +-----------------+
                                    |  bitstream.py   |  nextpnr, icepack,
                                    +-----------------+  then the packed
                                             |           bits are unpacked
                                             v           and re-simulated
                                 sizing, fabric, board fit
```

The nine generated blocks, and what each is for:

```
  arithmetic
    mac        signed multiply-accumulate, the thing that does the work
    requant    scale, round, saturate between two matmuls
    exp        e^x by table and shift, for softmax
    recip      1/x, for the softmax denominator
    rsqrt      1/sqrt(x), for RMSNorm
    crc32      the link endpoint, for board to board

  sequencing, which is what turns the above into a layer
    matvec     walks a matrix through the mac
    wmem       the weight tile the matvec reads from
    softmax    drives exp and recip across a row
    mlp        two chained matmuls with requant between them
```

Four of those are composite: they instantiate the blocks below them
rather than reimplementing the arithmetic, and the flow feeds the
dependencies in as extra sources. The three table-driven units (exp,
recip, rsqrt) also take a generated constant ROM as a dependency, since
emitting 256 exact table entries is a job for a generator and not for a
writer of RTL.

## The three files that connect everything

Everything downstream is driven by three machine-readable files:

| file | role |
|---|---|
| `model_spec.json` | the input workload: layer count, d_model, d_ff, quantization (weight_bits, activation_bits), seq_len, dtype, and the target tokens/s. Per-token MACs, weight and KV cache traffic, all-reduce bytes, and the chiplet architecture all derive from it |
| `chiplet_profile.json` | written by the agentic flow only after the compute chiplet passes simulation, synthesis, timing, and FPGA mapping: measured cycles/MAC, cell count, fmax, real LUT/FF/DSP counts, and the derivation record |
| `fabric_profile.json` | same flow, same signoff, for the fabric endpoint (a CRC32 datapath whose width is derived from the link rate): measured cycles/byte, FPGA resources, and the endpoint's achievable Gbps |

The chiplet spec is not a checked-in input. `specgen.py` derives it from the model before every run:

```
data_width = max(weight_bits, activation_bits)
acc_width  = weight_bits + activation_bits + ceil(log2(max(d_model, d_ff)))
```

The accumulator gets exactly the guard bits its longest dot-product reduction needs (12 for GPT-2's 3072-deep c_proj), so it can never overflow and is never wider than the model requires. The self-checking testbench is generated alongside the spec at the same widths, golden model and max-value stimulus included. A different model yields different signed-off hardware through the identical loop with no code changes: re-quantize the model to 4 bits and the flow converges on a 4x4-bit datapath with a 20-bit accumulator that synthesizes measurably smaller (the test suite does exactly this).

The fabric endpoint is derived the same way, from the link it has to keep up with rather than from the model. An endpoint narrower than the wire throttles it, so `specgen.derive_endpoint_spec` picks a standard Ethernet MAC datapath (bytes per cycle and core clock) that sustains the target rate, and generates the matching testbench with golden values computed by `zlib.crc32`. That closes a loop the RTL agent cannot: when a datapath misses timing, widening it is not an RTL fix, it is a different architecture, so `run_endpoint_flow` advances to the next standard width/clock pair and reruns the whole agent loop. At a 10 Gbps target the 8-byte datapath at 156.25 MHz misses by 0.02 ns and the 16-byte datapath at 78.125 MHz signs off at +1.10 ns, which is exactly the call a human designer makes at that point.

Profile provenance, identical for both generated blocks:

| field | where it comes from |
|---|---|
| `cycles_per_mac` / `cycles_per_byte` | measured by the testbench: a back-to-back burst, span cycles divided by units retired (never hardcoded) |
| `latency_cycles` | measured by the testbench, first valid in to first valid out |
| `cell_count`, `area` | yosys `stat` against the generic liberty |
| `fmax_estimate_mhz` | OpenSTA worst slack at the target clock: fmax = 1000 / (period - slack); falls back to a yosys gate-depth proxy if OpenSTA is absent |
| `endpoint_gbps` (fabric only) | bytes/cycle times the measured fmax: what the synthesized endpoint can actually carry |
| `fpga` | real device primitives from yosys `synth_xilinx`: LUTs, flip-flops, DSP slices, block RAM. The MAC infers one DSP; the endpoint is pure LUT logic |
| `derivation` (chiplet only) | the record of how the model produced this architecture: quantization, reduction depth, guard bits |
| `sim_checks_passed` | self-checking testbench vs golden vectors (the CRC testbench checks against zlib-computed CRC32 values) |

`boards.py` consumes both with `fit(board, chiplet_profile, fabric_profile, transport)` against real device budgets (Artix-7 XC7A100T, Zynq UltraScale+ ZU9EG, Alveo U250). The fit is multi-resource and that matters: because the MAC infers a DSP slice and the endpoint is LUT logic, the two blocks compete for different things, and a single capacity number cannot express it. Compute comes out DSP-bound on all three parts (168, 1764, and 8601 instances), while the transport cores and endpoints are charged against LUTs and flip-flops per link. The usable link rate is min(wire rate after 64b/66b coding, synthesized endpoint rate). Nothing upstream changes when the board class changes, and a cluster may mix classes.

## Connecting the boards

Three transports are modeled, and the choice is a measurable trade rather than a preference:

| transport | hop latency | wire overhead | core cost | switchable |
|---|---|---|---|---|
| Aurora 64B/66B, direct attach | 215 ns | 8 B/frame | ~1500 LUT/port | no |
| Ethernet, direct attach | 565 ns | 38 B/frame | ~5000 LUT/port | yes |
| Ethernet through a cut-through switch | 1015 ns | 38 B/frame | ~5000 LUT/port | yes |

On the same 8-board Alveo cluster, Ethernet direct-attach costs about 8% of peak throughput against Aurora and about 41% at 16 boards. The gap widens with board count because the ring's chunk size shrinks as 1/n while the per-hop latency stays fixed, so latency dominates exactly where Aurora is strongest.

**Ethernet is still the right default here, and the prototype uses it.** The project's claim is scaling across any FPGA of any class in any count, and Aurora is point-to-point and vendor-locked: it cannot express a topology that is not directly cabled, and it does not cross vendors. Ethernet buys commodity DAC cables, real switching (so the topology is not limited to a physical ring), vendor neutrality, and the same substrate the industry is standardizing scaleout on. The measured cost of that flexibility is 8% at the board counts sizing actually recommends, which is cheap.

One convenient alignment: Ethernet's frame check sequence is CRC-32 with polynomial 0x04C11DB7, the reflected form of which is the 0xEDB88320 the generated endpoint already implements. The agent-generated block is therefore the FCS engine, not a layer bolted above it. Note that the FCS protects a single hop; the sequence numbers, go-back-N retransmission, and credit flow control in `fabric.py` are what make the end-to-end ring reliable, and those remain necessary above any transport.

## Model assumptions and hardware mapping

| Component | What it models | Real hardware target |
|---|---|---|
| `model_spec.json` + `sizing.py` | per-token MACs, weight bytes (batch-1 identity: one byte moved per MAC at int8), KV cache traffic, and all-reduce bytes from the transformer shapes; per block, compute overlaps memory (max of the two), then the ring term is added, per candidate board count | capacity planning for LLM serving on a scaleout fabric, including the memory wall and SRAM residency |
| `sweep.py` + `dv.py` + `inference.py` | the verification layer: every spec through every gate, mutation testing of the generated testbenches, and co-simulation proving the generated RTL computes the model's arithmetic. See RESULTS.md | signoff: proving the flow works for specs nobody tuned it for, and that the DV can actually fail |
| `specgen.py` | derives the chiplet spec and its testbench from the model: datapath width from the quantization, accumulator width from the longest reduction | the architecture step: choosing the datapath the model actually needs before any RTL is written |
| `chiplet_flow.py` + `agent.py` / `llm_agent.py` | agent proposes RTL, tools verify, parsed failures feed back until signoff; the same loop runs both the derived chiplet and the fabric endpoint, with a deterministic rule-based agent by default and a real LLM behind the same interface (`--agent llm`) | the LLM-driven RTL generation loop on production verification and synthesis flows |
| generated `tb_mac.v` / `tb_crc.v` | golden-model checking plus a throughput burst that measures cycles per unit, at whatever widths the derivation chose | a full UVM environment plus performance characterization on the real testbench |
| yosys + `cells.lib` | mapping to a tiny generic liberty, cell count as the footprint metric | ASIC synthesis against a standard-cell library |
| `fpga.py`, yosys `synth_xilinx` | real device mapping: LUT, FF, DSP, and BRAM counts per block, plus a lint gate that fails the iteration on inferred latches, multiple drivers, and combinational loops | Vivado synthesis and implementation, with utilization and post-route timing reports |
| OpenSTA slack to fmax | achievable clock from static timing at the target period | vendor STA after place-and-route, speed-grade specific |
| `boards.py` devices | datasheet LUT/FF/DSP/BRAM budgets, clock caps, transceiver rates, effective DDR bandwidth; transport cores and endpoints reserved per link; weight shards that fit SRAM are modeled resident | the actual named parts: Artix-7 XC7A100T, Zynq UltraScale+ ZU9EG (ZCU102), Alveo U250 |
| `boards.TRANSPORTS` | Aurora and Ethernet (direct and switched): per-hop latency, per-frame wire overhead, MTU, and the LUT/FF cost of the core itself | Xilinx Aurora 64B/66B IP, or a 10/25G Ethernet MAC plus PCS over SFP+/QSFP28 |
| generated `crc.v` endpoint | word-serial CRC32 datapath, measured Gbps caps the usable link rate | the framing/CRC stage of an Aurora-style or Ethernet-style link endpoint |
| `Link` (rate, prop delay, BER knob) | serialization delay, flight time, random corruption | one transceiver lane running a framed stream |
| `_Tx` / `_Rx` in `fabric.py` | seq numbers, CRC32, go-back-N retransmit, cumulative ACKs | the reliable link-layer RTL block: replay buffer in BRAM, ACK sideband |
| Credit flow control | 64-packet RX buffer, sender stalls at zero credits, overflow impossible by construction | RTL credit counters and credit-return words on the reverse channel |
| `Board.compute` / `Board.matmul` | instances chiplets in parallel, cycles from the measured profile | the generated chiplet array the fabric feeds |
| `collectives.py` | ring all-reduce (reduce-scatter + all-gather) for arbitrary N, heterogeneous rings gated by the slowest segment | CCL-style hardware collectives over multi-board topologies |
| `sizing.simulate_decode` | full decode loop, n_layer x (attn, all-reduce, MLP, all-reduce) per token, every all-reduce real fabric traffic | tensor-parallel LLM decode across the cluster |

## How to run

```
python3 demo.py                    # end-to-end, all nine stages, writes results.json (about 30 s)
python3 tests.py                   # 143-check suite (both flows, spec derivation, DV mutation, FPGA mapping, transports, sizing physics)
python3 chiplet_flow.py            # just the two agentic RTL loops, writes both profiles
python3 chiplet_flow.py --agent llm    # same loops with a real LLM writing the RTL (Ollama, API, or Claude CLI)
python3 chiplet_flow.py --agent swarm  # three roles behind the same interface, escalating on tool failure
python3 specgen.py                 # just the model-to-chiplet derivation

python3 sweep.py                   # every spec end to end: 6 model specs, 4 link rates, all gates plus DV
python3 train_tiny.py              # one off: trains the small transformer, writes tiny_llm.json
python3 generate.py                # decode that checkpoint through the generated hardware's arithmetic
python3 bitstream.py --block mac   # place, route and pack a real iCE40 bitstream (needs nextpnr-ice40)
python3 dv.py --rtl build/mac.v --tb tb_mac.v   # mutation-test a generated testbench
python3 inference.py               # real quantized transformer dot products through the generated RTL
python3 bench.py --agent swarm --runs 5         # convergence rate, iterations, model calls by role
CHIPLET_LINK_GBPS=25 python3 chiplet_flow.py   # derive the endpoint for a different link rate
CHIPLET_FPGA_FAMILY=xc7 python3 chiplet_flow.py  # map to 7-series instead of UltraScale+
```

Python 3 stdlib only. Requires `iverilog` and `vvp` on PATH; `yosys` and `sta` (OpenSTA) are used when present and skipped gracefully when not (timing then falls back to the gate-depth proxy, clearly labeled). Build artifacts go to `build/`; `spec_mac.json` and `tb_mac.v` are regenerated from the model spec on every run.

## Sample output

```
Agentic FPGA scaleout: given an LLM, generate and synthesize the fabric to host it

========================================================================
Stage 1: the input LLM (model_spec.json drives everything downstream)
========================================================================

the workload the agents must build hardware for
quantity                 value            meaning
-----------------------  ---------------  ------------------------------------------------------------------------
                  model        gpt2_124m  GPT-2 124M decoder-only transformer, the LLM the fabric is sized to host
layers x d_model x d_ff  12 x 768 x 3072                                                  decoder-only transformer
  MACs per decode token            84.9M                                     n_layer * (4*d^2 attn + 2*d*d_ff MLP)
      KV-attention MACs            18.9M                           n_layer * 2 * seq_len * d_model at seq_len 1024
 weight bytes per token          84.9 MB                  batch-1 identity: every block weight read once per token
               KV cache          18.9 MB                   read fully from DDR every token; sharded with the heads
  all-reduces per token               24                        Megatron TP: one per attn block, one per MLP block
   bytes per all-reduce             6144                                  [1, d_model] activation, 8-byte elements
   comm bytes per token           147456                                      what the fabric must carry per token
             batch size                8       concurrent sequences per decode step; weights are read once for all
                 target        320 tok/s                                       8 concurrent users at 40 tok/s each

========================================================================
Stage 2: derive the chiplet spec from the model, then agents generate it
========================================================================

mac8_gpt2_124m: spec_mac.json and tb_mac.v both generated from the model
quantity      value       derivation
------------  ----------  --------------------------------------------------------------
MAC datapath  8 x 8 bits        the model quantization: weight_bits=8, activation_bits=8
 accumulator     28 bits  16 product bits + 12 guard bits, overflow-free by construction
  guard bits          12      ceil(log2(3072)), the longest dot-product reduction (d_ff)


iter  fixes applied                                sim                                   synth              timing                 fpga
--------------------------------------------------------------------------------------------------------------------------------------------------------------
1     none                                         FAIL wide_product: exp 65328 got 304  -                  -                      -
2     widen_product_register                       FAIL sync_clear: exp 0 got 129845     -                  -                      -
3     implement_sync_clear,widen_product_register  pass (594 checks)                     pass (1052 cells)  pass (slack +4.44 ns)  pass (34 LUT, 46 FF, 1 DSP)

CONVERGED in 3 iteration(s)

========================================================================
Stage 3: derive the fabric endpoint from the link rate, then generate it
========================================================================
   target link: 10 Gbps. The endpoint datapath has to sustain it or it
   throttles the wire, so width and clock are derived, not chosen by hand.

Endpoint datapath option 1/2 for 10 Gbps: 8 bytes/cycle at 156.25 MHz

iter  fixes applied          sim                                            synth              timing                 fpga
--------------------------------------------------------------------------------------------------------------------------------------------------
1     none                   FAIL zero_word: exp 1696784233 got 2598183062  -                  -                      -
2     apply_final_inversion  pass (5 checks)                                pass (3815 cells)  fail (slack -0.02 ns)  pass (522 LUT, 65 FF, 0 DSP)
3     apply_final_inversion  pass (5 checks)                                pass (3815 cells)  fail (slack -0.02 ns)  pass (522 LUT, 65 FF, 0 DSP)
4     apply_final_inversion  pass (5 checks)                                pass (3815 cells)  fail (slack -0.02 ns)  pass (522 LUT, 65 FF, 0 DSP)
5     apply_final_inversion  pass (5 checks)                                pass (3815 cells)  fail (slack -0.02 ns)  pass (522 LUT, 65 FF, 0 DSP)

DID NOT CONVERGE in 5 iteration(s)
   timing missed by -0.02 ns at 156.25 MHz, widening the datapath and slowing the clock

Endpoint datapath option 2/2 for 10 Gbps: 16 bytes/cycle at 78.125 MHz

iter  fixes applied          sim                                           synth              timing                 fpga
--------------------------------------------------------------------------------------------------------------------------------------------------
1     none                   FAIL zero_word: exp 3971697493 got 323269802  -                  -                      -
2     apply_final_inversion  pass (5 checks)                               pass (7333 cells)  pass (slack +1.10 ns)  pass (1004 LUT, 65 FF, 0 DSP)

CONVERGED in 2 iteration(s)

datapath search: timing rejected the narrow option
bytes/cycle  clock MHz  sustains Gbps  signed off
-----------  ---------  -------------  ----------
          8     156.25           10.0          no
         16     78.125           10.0         yes
   this is architecture-level feedback: a datapath that cannot close timing
   is not an RTL bug, so the flow widened it and halved the clock and retried

   the fabric is synthesized, not assumed: the endpoint that checks every
   packet in stage 7 is this RTL, at 10.94 Gbps measured

========================================================================
Stage 4: both profiles deploy on any board class (boards.py)
========================================================================
   per-instance cost from real yosys synth_xilinx runs: chiplet 34 LUT / 46 FF / 1 DSP,
   endpoint 1004 LUT / 65 FF / 0 DSP

fit(board, chiplet_profile, fabric_profile) over Ethernet, direct attach (no switch), usable device fraction = 70%
board                                price  instances  bound by  GMAC/s  DDR GB/s  SRAM MB  link Gbps  ports
-----------------------------------  -----  ---------  --------  ------  --------  -------  ---------  -----
    Arty A7-100T (Artix-7 XC7A100T)   $250        168      dsps    25.2         1      0.6       0.12      1
    KC705, used (Kintex-7 XC7K325T)   $350        588      dsps   105.8         9      2.0      10.00      1
Alveo U250 datacenter card (XCU250)  $3000       8601      dsps  1546.9        64     56.4      10.94      2
   the MAC infers a DSP slice, so compute is DSP-bound on every class: a single
   capacity number cannot express that, because the endpoint is pure LUT logic
   link rate = min(wire, synthesized endpoint), so the endpoint only gates boards
   whose wire is faster than the 10G target it was derived for
   one high-speed port on arty_a7_100t, kc705: those can be cabled to exactly one peer, so any
   cluster past two boards needs a switch, which is a physical argument for Ethernet

========================================================================
Stage 5: how the boards are wired: transport choice, measured
========================================================================

same cluster (alveo_u250), three ways to wire it
transport                              link Gbps  hop ns  frame B  MAC cost  switchable  peak tok/s  tok/s @16
-------------------------------------  ---------  ------  -------  --------  ----------  ----------  ---------
        Aurora 64B/66B, direct attach      10.94     215        8  1500 LUT          no        4093       4036
  Ethernet, direct attach (no switch)      10.94     565       38  5000 LUT         yes        3789       3506
Ethernet through a cut-through switch      10.94    1015       38  5000 LUT         yes        3676       3070
   Ethernet costs 7% of peak throughput and 13% at 16 boards versus Aurora,
   because latency matters more as chunks shrink. Batching works in Ethernet's
   favour here: at batch 8 each all-reduce carries 49152 bytes instead of 6144, so the
   fixed per-hop delay is amortised over a much larger message. It buys commodity
   cabling, real switching, and vendor neutrality, which is what "any FPGA, any
   count" requires.

========================================================================
Stage 6: the right fabric: smallest cluster per board class that hosts the model
========================================================================

analytic sizing vs the 320 tok/s target
board class   boards needed  predicted tok/s  bound   weights in SRAM
------------  -------------  ---------------  ------  ---------------
arty_a7_100t    unreachable        48 @ n=16    comm
       kc705              2              566  memory               no
  alveo_u250              1             2170  memory               no
   decode touches every weight once per step, so a board streaming from DDR is
   memory-bound no matter how many MACs it has. Even at batch 8, every row above
   is bound by memory or the link, not by the 25 to 1547 GMAC/s of compute on offer.

batching on kc705: one step reads the weights once and serves the whole batch
batch  1 board tok/s  2 boards tok/s  bound   weight MB/step  KV MB/step
-----  -------------  --------------  ------  --------------  ----------  ------
    1             87             169  memory            84.9        18.9
    2            147             282  memory            84.9        37.7
    4            224             424  memory            84.9        75.5
    8            305             566  memory            84.9       151.0  chosen
   16            353             649  memory            84.9       302.0
   32            366             671  memory            84.9       604.0
   weight traffic per step is fixed, so batching converts idle compute into tokens.
   It stops paying once the KV cache read, which does scale with the batch,
   overtakes the weights. That is why real serving systems fight over KV size.
   Throughput, not latency: each sequence still waits a full step for its token.

candidate sweep on the large class (weight shards become SRAM-resident at 2 boards)
boards  pred tok/s  compute/tok  mem/tok   comm/tok  bound   resident
------  ----------  -----------  --------  --------  ------  --------  ------
     1        2170     536.8 us   3.69 ms      0 ns  memory        no  chosen
     2        3533     268.4 us   1.18 ms  938.6 us  memory       yes
     4        3789     134.2 us  589.8 us   1.45 ms    comm       yes
     8        3780      67.1 us  294.9 us   1.78 ms    comm       yes
    16        3506      33.6 us  147.5 us   2.12 ms    comm       yes
   throughput peaks at 4 boards (3789 tok/s) and then falls: past residency the
   ring all-reduce grows as 2*(n-1) while there is no memory traffic left to save,
   so more boards is actively worse. That is the number the sizing layer exists to find.

========================================================================
Stage 7: host the model: decode on the chosen fabric, every all-reduce real traffic
========================================================================

2 x kc705 hosting gpt2_124m
quantity               value     provenance
---------------------  --------  -------------------------------------------------
               boards         2                        chosen by sizing in stage 6
       tokens decoded        64                        full n_layer loop per token
       measured tok/s       562  discrete-event fabric, packetized + CRC + credits
      predicted tok/s       566                        analytic model from stage 6
     prediction ratio      0.99                               measured / predicted
collectives per token        24                          n_layer * 2 = 24 expected
weights SRAM-resident     False                42.5 MB shard vs 2.0 MB SRAM budget
           wire bytes  19685376                           headers and CRC included
activations identical      True          every board holds the same reduced vector
   target met: 562 tok/s measured vs 320 required

========================================================================
Stage 8: scale by adding boards, and survive a lossy fabric
========================================================================

same synthesized fabric, more kc705 boards
boards  tok/s  time/token  weights in SRAM  activations identical
------  -----  ----------  ---------------  ---------------------
     2    562     1.78 ms               no                   True
     4    956     1.05 ms               no                   True
     8   1451    689.1 us               no                   True
    16   1868    535.4 us               no                   True
   no residency cliff on this class: the shard never fits on-chip, so every
   added board only divides the DDR traffic, and the ring term (2*(n-1)) erodes it

   BER 1e-6 on every link, 8 boards: 589 CRC drops, 9633 retransmits,
   activations still identical on every board = True, throughput 591 tok/s
   (2.5x slower than the clean fabric: reliability costs latency, never bits)

========================================================================
Stage 9: numerics check at reduced dimensions (real arithmetic through the fabric)
========================================================================
   homogeneous 4-board MLP 64x64 W1 64x512 W2 512x64: time 64.3 us, max err vs single-board reference 8.9e-15
   heterogeneous 2 small + 2 large: bit-identical output to homogeneous = True
   work-proportional shards ([4, 4, 252, 252]) change the time by 1.01x: with the Arty limited to
   100 Mbit Ethernet this ring is link-bound, so rebalancing compute barely helps.
   The result that matters here is the first line: the answer is identical either way.

results written to results.json
```

Reading the numbers: the chiplet the agents build is the one this model asked for, an 8x8-bit MAC from the int8 quantization with a 28-bit accumulator (16 product bits plus 12 guard bits for the 3072-deep c_proj reduction), and the endpoint is the one the link asked for, a 16-byte datapath reached only after timing rejected the narrower 8-byte option. Both seeded first-cut bugs (a truncated product register, a missing CRC final inversion) are caught by the generated testbenches and fixed from parsed feedback, and both blocks now also clear a real FPGA mapping stage: 34 LUT / 46 FF / 1 DSP for the MAC, 1004 LUT / 65 FF / 0 DSP for the endpoint. That split is why the fit is multi-resource, and it makes compute DSP-bound on every part, 168 instances on the Artix-7 up to 8601 on the U250.

Sizing then runs into the real wall of batch-1 decode: every token touches all 84.9 MB of block weights, so a board streaming from DDR is memory-bound no matter how much compute it has, and every row of the sizing table is bound by memory rather than by the 25 to 1547 GMAC/s on offer. The small class never reaches the target, the mid class needs 8 boards, and the large class reaches it on one. The large-class sweep is the interesting one: weight shards become SRAM-resident at 2 boards, throughput jumps from 617 to 3259 tok/s, and then *falls* at 4, 8, and 16 as the ring's 2*(n-1) term grows with no memory traffic left to save. More boards is actively worse past the peak, which is precisely the number the sizing layer exists to find. The chosen 8x ZCU102 configuration hosts decode with every all-reduce as real link traffic and measures 642 tok/s against 670 predicted, and at BER 1e-6 the fabric retransmits its way to the same bit-exact activations. Tensor parallelism here is not primarily buying FLOPs, it is buying enough aggregate on-chip SRAM to stop streaming weights.

## How an LLM agent slots in

`agent.py` exposes one interface: `propose(spec, feedback_history) -> (verilog_source, notes)`, and `llm_agent.py` ships a real LLM behind it. The prompt serializes the spec, the agent's own previous attempt, and the parsed tool feedback (failing test, expected vs got, compile errors, negative slack); the response is fence-stripped and trimmed to the module, and anything malformed simply fails simulation and feeds back. Three backends, all stdlib urllib, autodetected or forced with `CHIPLET_LLM`: the Anthropic API (`ANTHROPIC_API_KEY`), a local Ollama server (lightweight local models such as qwen2.5-coder), or the Claude Code CLI in print mode. Select it with `--agent llm` or `CHIPLET_AGENT=llm`; the deterministic `RuleBasedAgent` stays the default so the demo and test suite are reproducible offline.

A real run, Claude Haiku through the CLI backend, both blocks first try:

```
Agent: LLMAgent (haiku@claude-cli)

iter  fixes applied           sim                synth              timing
-----------------------------------------------------------------------------------------
1     llm:haiku@claude-cli#1  pass (594 checks)  pass (1058 cells)  pass (slack +4.44 ns)

CONVERGED in 1 iteration(s)

iter  fixes applied           sim              synth              timing
---------------------------------------------------------------------------------------
1     llm:haiku@claude-cli#1  pass (5 checks)  pass (1756 cells)  pass (slack +6.26 ns)

CONVERGED in 1 iteration(s)
```

The generated RTL is structurally its own: the MAC pipelines a combinational product wire with explicit valid handling, and the CRC unrolls the 32 iterations with a generate loop instead of a function, which synthesized 13% smaller than the rule-based render (1756 vs 2016 cells) and timed faster (267 vs 251 MHz, an 8.56 Gbps endpoint). Both cleared every golden-vector check, the seeded-bug traps included (full-width product, synchronous clear priority, CRC final inversion), on the first attempt. The committed profiles remain the rule-based ones so the repo is reproducible without a network; rerun with `--agent llm` to regenerate these. Because the chiplet spec itself is generated from the model, pointing the pipeline at a different LLM workload is a one-file change to `model_spec.json`: the derivation produces a different spec and testbench, the agent produces different RTL, and the measured profile produces a different fit, link rate, and sizing answer with no code changes anywhere downstream. The derivation step (`specgen.py`) is likewise where an LLM architecture agent would slot in next, proposing datapath structure from the model rather than applying closed-form width rules.

## Honest simplifications

`RESULTS.md` carries the measured state and, next to it, the list of what is
not verified: bitstreams that are real but for an iCE40 rather than the
Xilinx boards the team owns, nothing yet loaded onto hardware, generated blocks that are the
arithmetic of an inference engine rather than the whole of one, a committed
checkpoint that is real but small, and a throughput figure that is a model
checked against another model. Read that list before quoting any number here.


- FPGA resources come from yosys `synth_xilinx`, which is real technology mapping but not place-and-route: there is no routing congestion, no floorplan, and no post-route timing. Vivado would give different and more pessimistic numbers, and yosys's own DSP inference varies by family (the 7-series mapping absorbs the accumulator into the DSP48E1, the UltraScale+ one does not).
- fmax still comes from OpenSTA against the toy generic liberty, so it is an ASIC-flavored number used as an FPGA clock estimate. A real FPGA fmax needs Vivado timing; the endpoint datapath search is therefore honest about *relative* timing pressure rather than absolute megahertz.
- Feeding 8601 MAC instances would need on-chip operand bandwidth the model does not check. DSP count is the right first-order capacity ceiling, not a claim that the array is routable at that size.
- A 25 Gbps endpoint does not close timing in this flow at either standard datapath. Reaching it needs a pipelined or matrix-form CRC that the rule-based agent does not write, so the default target is 10 Gbps, which is also what the mid board class actually exposes.
- There is no place-and-route anywhere; timing is real OpenSTA static timing but against a toy illustrative liberty, so fmax is an estimate of an estimate.
- The derivation now covers nine blocks, not just the matmul datapath: the exponential, reciprocal and inverse square root are generated and signed off, and softmax and an MLP layer sequence them. What is still missing above that is attention sequencing and tiling. The weight tile holds 1024 entries and the activation bank 64, so a real matrix needs tiling logic that does not exist yet, and the blocks are the arithmetic of an inference engine rather than the whole of one.
- Boards are simulated, not real: link rates, propagation delays, clock caps, and capacities are representative class parameters, not measured silicon.
- The 30% fabric reservation for NIC and routing logic is a stated guess, not a floorplan; the endpoint reservation per link is real, from the synthesized cell count.
- Decode compute time is charged from the measured chiplet profile at full model dimensions; the weights are not materialized. The sharded numerics (real arithmetic, bit-exactness across cluster shapes) are validated at reduced dimensions in stage 8 and in the test suite.
- The memory model is deliberately coarse: DDR bandwidth and SRAM budgets are representative class parameters; a weight shard either fits SRAM (no per-token DDR traffic) or streams in full each token, with compute and memory overlapped per block as max(compute, memory), modeling perfect double buffering. The KV cache always lives in DDR and is read in full every token at seq_len depth.
- The LM head and the embedding table are excluded from both compute and memory traffic (modeled as host-side work), so per-token block MACs equal block parameters exactly (the batch-1 identity the weight-traffic term is built on).
- Activations travel as 8-byte IEEE doubles end to end so every all-reduce check is exact; a real deployment would use fp16 or fp32 and halve or quarter the fabric traffic.
- ACK/credit control packets share link bandwidth but are assumed error-free (in RTL they are short, heavily protected control words); topology is a full mesh of point-to-point links; payloads are fixed 1024 B with a 20 B header.
- The compute cycle model charges cycles per unit from the profile; it ignores on-board operand distribution to the chiplet array and memory bandwidth limits.
- The committed RTL and profiles come from the rule-based agent, so they are reproducible offline. The LLM agent is measured separately in `RESULTS.md`: it signs off six of the nine blocks, including the three fixed-point table units once their specs stated the exact bit ranges, and has not yet been rerun on the requantizer, softmax and MLP layer.
