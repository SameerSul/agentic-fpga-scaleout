# Agentic FPGA Scaleout: Given an LLM, Generate and Synthesize the Fabric to Host It (Prototype)

The input is an LLM: a model spec (GPT-2 124M here) with a target token rate. The hardware is derived from how the model computes: the MAC chiplet's datapath width comes from the model's quantization and its accumulator width from the longest dot-product reduction in the network, and agents generate that chiplet and the fabric endpoint that carries every packet as real RTL through real tools with no human in the loop, both passing simulation, synthesis, and timing signoff and emitting measured performance profiles. A sizing layer then derives the right fabric for the model, the smallest cluster of a given FPGA class that hosts it at the target rate, modeling compute, DDR bandwidth, on-chip SRAM residency of the weight shards, KV cache traffic, and ring all-reduce cost, and validates the choice by running decode on a discrete-event fabric where every all-reduce is packetized, CRC-checked, credit-gated traffic through the synthesized endpoint's measured rate. Scaling is adding more FPGAs running the same synthesized fabric: the same profiles fit any board class, clusters may mix classes, and the numerics stay bit-exact at any board count, even on lossy links.

Pure Python 3 stdlib, no dependencies. The RTL half runs real open tools (Icarus Verilog, Yosys, OpenSTA); the scaleout half is a discrete-event simulator in nanoseconds on a `heapq` event queue.

## The three files that connect everything

Everything downstream is driven by three machine-readable files:

| file | role |
|---|---|
| `model_spec.json` | the input workload: layer count, d_model, d_ff, quantization (weight_bits, activation_bits), seq_len, dtype, and the target tokens/s. Per-token MACs, weight and KV cache traffic, all-reduce bytes, and the chiplet architecture all derive from it |
| `chiplet_profile.json` | written by the agentic flow only after the compute chiplet passes sim, synth, and timing: measured cycles/MAC, cell count, fmax, and the derivation record |
| `fabric_profile.json` | same flow, same signoff, for the fabric endpoint (a word-serial CRC32 datapath): measured cycles/byte and the endpoint's achievable Gbps |

The chiplet spec is not a checked-in input. `specgen.py` derives it from the model before every run:

```
data_width = max(weight_bits, activation_bits)
acc_width  = weight_bits + activation_bits + ceil(log2(max(d_model, d_ff)))
```

The accumulator gets exactly the guard bits its longest dot-product reduction needs (12 for GPT-2's 3072-deep c_proj), so it can never overflow and is never wider than the model requires. The self-checking testbench is generated alongside the spec at the same widths, golden model and max-value stimulus included. A different model yields different signed-off hardware through the identical loop with no code changes: re-quantize the model to 4 bits and the flow converges on a 4x4-bit datapath with a 20-bit accumulator that synthesizes measurably smaller (the test suite does exactly this).

Profile provenance, identical for both generated blocks:

| field | where it comes from |
|---|---|
| `cycles_per_mac` / `cycles_per_byte` | measured by the testbench: a back-to-back burst, span cycles divided by units retired (never hardcoded) |
| `latency_cycles` | measured by the testbench, first valid in to first valid out |
| `cell_count`, `area` | yosys `stat` against the generic liberty |
| `fmax_estimate_mhz` | OpenSTA worst slack at the target clock: fmax = 1000 / (period - slack); falls back to a yosys gate-depth proxy if OpenSTA is absent |
| `endpoint_gbps` (fabric only) | bytes/cycle times the measured fmax: what the synthesized endpoint can actually carry |
| `derivation` (chiplet only) | the record of how the model produced this architecture: quantization, reduction depth, guard bits |
| `sim_checks_passed` | self-checking testbench vs golden vectors (the CRC testbench checks against zlib-computed CRC32 values) |

`boards.py` consumes both with `fit(board, chiplet_profile, fabric_profile)`: endpoint cells are reserved off the capacity proxy per link, chiplet instances fill what remains, and the usable link rate is min(board transceiver, synthesized endpoint). On the mid and large board classes the endpoint is the limit, so the fabric the cluster actually gets is the one the agents built, not a datasheet number. Nothing upstream changes when the board class changes, and a cluster may mix classes.

## Model assumptions and hardware mapping

| Component | What it models | Real hardware target |
|---|---|---|
| `model_spec.json` + `sizing.py` | per-token MACs, weight bytes (batch-1 identity: one byte moved per MAC at int8), KV cache traffic, and all-reduce bytes from the transformer shapes; per block, compute overlaps memory (max of the two), then the ring term is added, per candidate board count | capacity planning for LLM serving on a scaleout fabric, including the memory wall and SRAM residency |
| `specgen.py` | derives the chiplet spec and its testbench from the model: datapath width from the quantization, accumulator width from the longest reduction | the architecture step: choosing the datapath the model actually needs before any RTL is written |
| `chiplet_flow.py` + `agent.py` / `llm_agent.py` | agent proposes RTL, tools verify, parsed failures feed back until signoff; the same loop runs both the derived chiplet and the fabric endpoint, with a deterministic rule-based agent by default and a real LLM behind the same interface (`--agent llm`) | the LLM-driven RTL generation loop on production verification and synthesis flows |
| generated `tb_mac.v` / `tb_crc.v` | golden-model checking plus a throughput burst that measures cycles per unit, at whatever widths the derivation chose | a full UVM environment plus performance characterization on the real testbench |
| yosys + `cells.lib` | mapping to a tiny generic liberty, cell count as the footprint metric | FPGA synthesis to LUTs/FFs/DSPs with a place-and-route utilization report |
| OpenSTA slack to fmax | achievable clock from static timing at the target period | vendor STA after place-and-route, speed-grade specific |
| `boards.py` profiles | capacity proxy, clock cap, transceiver class, effective DDR bandwidth, and SRAM budget per board family; endpoint cells reserved per link; weight shards that fit SRAM are modeled resident (no per-token DDR traffic) | Artix-7, Kintex/Zynq UltraScale, and Versal/Alveo class parts with GTP/GTH/GTY-class serial links, DDR controllers, and BRAM/URAM |
| generated `crc.v` endpoint | word-serial CRC32 datapath, measured Gbps caps the usable link rate | the framing/CRC stage of an Aurora-style or Ethernet-style link endpoint |
| `Link` (rate, prop delay, BER knob) | serialization delay, flight time, random corruption | one transceiver lane running a framed stream |
| `_Tx` / `_Rx` in `fabric.py` | seq numbers, CRC32, go-back-N retransmit, cumulative ACKs | the reliable link-layer RTL block: replay buffer in BRAM, ACK sideband |
| Credit flow control | 64-packet RX buffer, sender stalls at zero credits, overflow impossible by construction | RTL credit counters and credit-return words on the reverse channel |
| `Board.compute` / `Board.matmul` | instances chiplets in parallel, cycles from the measured profile | the generated chiplet array the fabric feeds |
| `collectives.py` | ring all-reduce (reduce-scatter + all-gather) for arbitrary N, heterogeneous rings gated by the slowest segment | CCL-style hardware collectives over multi-board topologies |
| `sizing.simulate_decode` | full decode loop, n_layer x (attn, all-reduce, MLP, all-reduce) per token, every all-reduce real fabric traffic | tensor-parallel LLM decode across the cluster |

## How to run

```
python3 demo.py                    # end-to-end, all eight stages, writes results.json (about 6 s)
python3 tests.py                   # 43-check correctness suite (both flows, a 4-bit model variant, sizing physics)
python3 chiplet_flow.py            # just the two agentic RTL loops, writes both profiles
python3 chiplet_flow.py --agent llm  # same loops with a real LLM writing the RTL (Ollama, API, or Claude CLI)
python3 specgen.py                 # just the model-to-chiplet derivation
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
                 target        500 tok/s                                           the rate the fabric is sized to

========================================================================
Stage 2: derive the chiplet spec from the model, then agents generate it
========================================================================

mac8_gpt2_124m: spec_mac.json and tb_mac.v both generated from the model
quantity      value       derivation
------------  ----------  --------------------------------------------------------------
MAC datapath  8 x 8 bits        the model quantization: weight_bits=8, activation_bits=8
 accumulator     28 bits  16 product bits + 12 guard bits, overflow-free by construction
  guard bits          12      ceil(log2(3072)), the longest dot-product reduction (d_ff)

Agent: RuleBasedAgent

iter  fixes applied                                sim                                   synth              timing
---------------------------------------------------------------------------------------------------------------------------------
1     none                                         FAIL wide_product: exp 65328 got 304  -                  -
2     widen_product_register                       FAIL sync_clear: exp 0 got 129845     -                  -
3     implement_sync_clear,widen_product_register  pass (594 checks)                     pass (1052 cells)  pass (slack +4.44 ns)

CONVERGED in 3 iteration(s)

========================================================================
Stage 3: agents generate the fabric endpoint (CRC32 datapath, same signoff)
========================================================================
Agent: RuleBasedAgent

iter  fixes applied          sim                                           synth              timing
-------------------------------------------------------------------------------------------------------------------
1     none                   FAIL zero_word: exp 558161692 got 3736805603  -                  -
2     apply_final_inversion  pass (5 checks)                               pass (2016 cells)  pass (slack +6.02 ns)

CONVERGED in 2 iteration(s)

   the fabric is synthesized, not assumed: the endpoint that checks every
   packet in stage 6 is this RTL, and its measured rate caps the link below

========================================================================
Stage 4: both profiles deploy on any board class (boards.py)
========================================================================

fit(board, chiplet_profile, fabric_profile), usable fabric fraction = 70%
board class                         instances  clock MHz  GMAC/s  xcvr Gbps  endpoint Gbps  link Gbps  DDR GB/s  SRAM MB
----------------------------------  ---------  ---------  ------  ---------  -------------  ---------  --------  -------
             small (Artix-7 class)          5      150.0    0.75        6.6           4.80       4.80       1.6      0.6
mid (Kintex/Zynq UltraScale class)         64      179.9   11.51       16.3           8.04       8.04       3.2      4.5
        large (Versal/Alveo class)        268      179.9   48.20       32.0           8.04       8.04      12.0     24.0
   link rate = min(transceiver, synthesized endpoint): the endpoint gates
   the mid and large boards, an honest measured limit, not a datasheet number

========================================================================
Stage 5: the right fabric: smallest cluster per board class that hosts the model
========================================================================

analytic sizing vs the 500 tok/s target
board class   boards needed  predicted tok/s  bound    weights in SRAM
------------  -------------  ---------------  -------  ---------------
artix7_small    unreachable       105 @ n=16  compute
 zynq_us_mid    unreachable       384 @ n=16   memory
versal_large              4             1049  compute              yes
   decode is memory-bound until the weight shard fits on-chip SRAM: the small
   and mid classes never get there, so only the large class can host the target

candidate sweep on the large class (the SRAM residency cliff sits between 2 and 4 boards)
boards  pred tok/s  compute/tok  mem/tok   comm/tok  bound    resident
------  ----------  -----------  --------  --------  -------  --------  ------
     1         116      2.15 ms   8.65 ms      0 ns   memory        no
     2         223      1.08 ms   4.33 ms  162.1 us   memory        no
     4        1049     538.4 us  393.2 us  266.1 us  compute       yes  chosen
     8        1413     269.2 us  196.6 us  364.2 us     comm       yes
    16        1477     134.6 us   98.3 us  505.4 us     comm       yes

========================================================================
Stage 6: host the model: decode on the chosen fabric, every all-reduce real traffic
========================================================================

4 x versal_large hosting gpt2_124m
quantity               value    provenance
---------------------  -------  -------------------------------------------------
               boards        4                        chosen by sizing in stage 5
       tokens decoded        8                        full n_layer loop per token
       measured tok/s     1007  discrete-event fabric, packetized + CRC + credits
      predicted tok/s     1049                        analytic model from stage 5
     prediction ratio     0.96                               measured / predicted
collectives per token       24                          n_layer * 2 = 24 expected
weights SRAM-resident     True               21.2 MB shard vs 24.0 MB SRAM budget
           wire bytes  7483392                           headers and CRC included
activations identical     True          every board holds the same reduced vector
   target met: 1007 tok/s measured vs 500 required

========================================================================
Stage 7: scale by adding boards, and survive a lossy fabric
========================================================================

same synthesized fabric, more versal_large boards
boards  tok/s  time/token  weights in SRAM  activations identical
------  -----  ----------  ---------------  ---------------------
     2    222     4.51 ms               no                   True
     4   1007    992.8 us              yes                   True
     8   1295    772.2 us              yes                   True
    16   1340    746.3 us              yes                   True
   the jump is the SRAM residency cliff: once the weight shard fits on-chip,
   decode stops streaming DDR; past that the ring all-reduce term (2*(n-1)) pushes back

   BER 1e-6 on every link, 8 boards: 63 CRC drops, 446 retransmits,
   activations still identical on every board = True, throughput 420 tok/s
   (3.1x slower than the clean fabric: reliability costs latency, never bits)

========================================================================
Stage 8: numerics check at reduced dimensions (real arithmetic through the fabric)
========================================================================
   homogeneous 4-board MLP 64x64 W1 64x512 W2 512x64: time 152.4 us, max err vs single-board reference 8.9e-15
   heterogeneous 2 small + 2 large: bit-identical output to homogeneous = True
   work-proportional shards ([4, 4, 252, 252]) recover 11.06x over the equal split

results written to results.json
```

Reading the numbers: the chiplet the agents build is the one this model asked for, an 8x8-bit MAC from the int8 quantization with a 28-bit accumulator (16 product bits plus 12 guard bits for the 3072-deep c_proj reduction), and both seeded first-cut bugs (a truncated product register, a missing CRC final inversion) are caught by the generated testbenches and fixed from parsed feedback. The measured endpoint (4 bytes/cycle at 251 MHz = 8.04 Gbps) becomes the usable link rate on the mid and large boards, gating their 16.3 and 32 Gbps transceivers, so the sizing answers reflect the fabric the agents actually built. Sizing then runs into the real wall of batch-1 decode: every token must touch all 84.9 MB of block weights, so a board streaming from DDR is memory-bound no matter how many MACs it has. The small class is compute-starved, the mid class is memory-bound at every board count (its 16-way shard of 5.3 MB still misses the 4.5 MB SRAM budget, peaking at 384 tok/s), and the large class crosses the SRAM residency cliff at 4 boards: 223 tok/s streaming at n=2 jumps to 1049 predicted once the 21.2 MB shard fits the 24 MB budget. The chosen 4-board configuration hosts decode with every all-reduce as real link traffic and lands at 1007 tok/s, within 5% of the prediction. Past residency the ring's 2\*(n-1) serialization term takes over as the bound (comm at n=8 and 16), and at BER 1e-6 the fabric retransmits its way to the same bit-exact activations at 3.1x the latency. This is the quantitative version of why the fabric exists at all: tensor parallelism is not primarily buying FLOPs here, it is buying enough aggregate SRAM to stop streaming weights.

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

- Generic liberty cells are not FPGA LUTs: `lut_capacity_proxy` is in generic-cell units so the yosys cell count maps onto it directly. Real capacity fit needs LUT/FF/DSP utilization from vendor place-and-route.
- There is no place-and-route anywhere; timing is real OpenSTA static timing but against a toy illustrative liberty, so fmax is an estimate of an estimate.
- The derivation covers the matmul datapath only: MAC and accumulator widths from closed-form rules. Softmax, layernorm, and nonlinearity hardware are not generated and their cost is not modeled; a fuller version derives those blocks the same way.
- Boards are simulated, not real: link rates, propagation delays, clock caps, and capacities are representative class parameters, not measured silicon.
- The 30% fabric reservation for NIC and routing logic is a stated guess, not a floorplan; the endpoint reservation per link is real, from the synthesized cell count.
- Decode compute time is charged from the measured chiplet profile at full model dimensions; the weights are not materialized. The sharded numerics (real arithmetic, bit-exactness across cluster shapes) are validated at reduced dimensions in stage 8 and in the test suite.
- The memory model is deliberately coarse: DDR bandwidth and SRAM budgets are representative class parameters; a weight shard either fits SRAM (no per-token DDR traffic) or streams in full each token, with compute and memory overlapped per block as max(compute, memory), modeling perfect double buffering. The KV cache always lives in DDR and is read in full every token at seq_len depth.
- The LM head and the embedding table are excluded from both compute and memory traffic (modeled as host-side work), so per-token block MACs equal block parameters exactly (the batch-1 identity the weight-traffic term is built on).
- Activations travel as 8-byte IEEE doubles end to end so every all-reduce check is exact; a real deployment would use fp16 or fp32 and halve or quarter the fabric traffic.
- ACK/credit control packets share link bandwidth but are assumed error-free (in RTL they are short, heavily protected control words); topology is a full mesh of point-to-point links; payloads are fixed 1024 B with a 20 B header.
- The compute cycle model charges cycles per unit from the profile; it ignores on-board operand distribution to the chiplet array and memory bandwidth limits.
- The "agents" shipped here are rule-based (see above), so convergence in 3 and 2 iterations demonstrates the loop mechanics, not LLM capability.
