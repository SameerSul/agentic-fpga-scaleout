# Agentic FPGA Scaleout: Given an LLM, Generate and Synthesize the Fabric to Host It (Prototype)

The input is an LLM: a model spec (GPT-2 124M here) with a target token rate. Agents generate the hardware to host it as real RTL through real tools with no human in the loop, and both generated blocks, the compute chiplet and the fabric endpoint that carries every packet, pass simulation, synthesis, and timing signoff and emit measured performance profiles. A sizing layer then derives the right fabric for the model, the smallest cluster of a given FPGA class that hosts it at the target rate, and validates the choice by running decode on a discrete-event fabric where every all-reduce is packetized, CRC-checked, credit-gated traffic through the synthesized endpoint's measured rate. Scaling is adding more FPGAs running the same synthesized fabric: the same profiles fit any board class, clusters may mix classes, and the numerics stay bit-exact at any board count, even on lossy links.

Pure Python 3 stdlib, no dependencies. The RTL half runs real open tools (Icarus Verilog, Yosys, OpenSTA); the scaleout half is a discrete-event simulator in nanoseconds on a `heapq` event queue.

## The three files that connect everything

Everything downstream is driven by three machine-readable files:

| file | role |
|---|---|
| `model_spec.json` | the input workload: layer count, d_model, d_ff, dtype, and the target tokens/s. Per-token MACs and all-reduce bytes derive from it |
| `chiplet_profile.json` | written by the agentic flow only after the compute chiplet passes sim, synth, and timing: measured cycles/MAC, cell count, fmax |
| `fabric_profile.json` | same flow, same signoff, for the fabric endpoint (a word-serial CRC32 datapath): measured cycles/byte and the endpoint's achievable Gbps |

Profile provenance, identical for both generated blocks:

| field | where it comes from |
|---|---|
| `cycles_per_mac` / `cycles_per_byte` | measured by the testbench: a back-to-back burst, span cycles divided by units retired (never hardcoded) |
| `latency_cycles` | measured by the testbench, first valid in to first valid out |
| `cell_count`, `area` | yosys `stat` against the generic liberty |
| `fmax_estimate_mhz` | OpenSTA worst slack at the target clock: fmax = 1000 / (period - slack); falls back to a yosys gate-depth proxy if OpenSTA is absent |
| `endpoint_gbps` (fabric only) | bytes/cycle times the measured fmax: what the synthesized endpoint can actually carry |
| `sim_checks_passed` | self-checking testbench vs golden vectors (the CRC testbench checks against zlib-computed CRC32 values) |

`boards.py` consumes both with `fit(board, chiplet_profile, fabric_profile)`: endpoint cells are reserved off the capacity proxy per link, chiplet instances fill what remains, and the usable link rate is min(board transceiver, synthesized endpoint). On the mid and large board classes the endpoint is the limit, so the fabric the cluster actually gets is the one the agents built, not a datasheet number. Nothing upstream changes when the board class changes, and a cluster may mix classes.

## Model assumptions and hardware mapping

| Component | What it models | Real hardware target |
|---|---|---|
| `model_spec.json` + `sizing.py` | per-token MACs and all-reduce bytes from the transformer shapes, analytic compute + ring-step time per candidate board count | capacity planning for LLM serving on a scaleout fabric |
| `chiplet_flow.py` + `agent.py` | agent proposes RTL, tools verify, parsed failures feed back until signoff; the same loop runs both the compute chiplet and the fabric endpoint | the LLM-driven RTL generation loop on production verification and synthesis flows |
| `tb_mac.v` / `tb_crc.v` | golden-model checking plus a throughput burst that measures cycles per unit | a full UVM environment plus performance characterization on the real testbench |
| yosys + `cells.lib` | mapping to a tiny generic liberty, cell count as the footprint metric | FPGA synthesis to LUTs/FFs/DSPs with a place-and-route utilization report |
| OpenSTA slack to fmax | achievable clock from static timing at the target period | vendor STA after place-and-route, speed-grade specific |
| `boards.py` profiles | capacity proxy, clock cap, transceiver class per board family; endpoint cells reserved per link | Artix-7, Kintex/Zynq UltraScale, and Versal/Alveo class parts with GTP/GTH/GTY-class serial links |
| generated `crc.v` endpoint | word-serial CRC32 datapath, measured Gbps caps the usable link rate | the framing/CRC stage of an Aurora-style or Ethernet-style link endpoint |
| `Link` (rate, prop delay, BER knob) | serialization delay, flight time, random corruption | one transceiver lane running a framed stream |
| `_Tx` / `_Rx` in `fabric.py` | seq numbers, CRC32, go-back-N retransmit, cumulative ACKs | the reliable link-layer RTL block: replay buffer in BRAM, ACK sideband |
| Credit flow control | 64-packet RX buffer, sender stalls at zero credits, overflow impossible by construction | RTL credit counters and credit-return words on the reverse channel |
| `Board.compute` / `Board.matmul` | instances chiplets in parallel, cycles from the measured profile | the generated chiplet array the fabric feeds |
| `collectives.py` | ring all-reduce (reduce-scatter + all-gather) for arbitrary N, heterogeneous rings gated by the slowest segment | CCL-style hardware collectives over multi-board topologies |
| `sizing.simulate_decode` | full decode loop, n_layer x (attn, all-reduce, MLP, all-reduce) per token, every all-reduce real fabric traffic | tensor-parallel LLM decode across the cluster |

## How to run

```
python3 demo.py           # end-to-end, all eight stages, writes results.json (about 6 s)
python3 tests.py          # 27-check correctness suite (runs both flows once)
python3 chiplet_flow.py   # just the two agentic RTL loops, writes both profiles
```

Python 3 stdlib only. Requires `iverilog` and `vvp` on PATH; `yosys` and `sta` (OpenSTA) are used when present and skipped gracefully when not (timing then falls back to the gate-depth proxy, clearly labeled). Build artifacts go to `build/`.

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
  all-reduces per token               24                        Megatron TP: one per attn block, one per MLP block
   bytes per all-reduce             6144                                  [1, d_model] activation, 8-byte elements
   comm bytes per token           147456                                      what the fabric must carry per token
                 target        500 tok/s                                           the rate the fabric is sized to

========================================================================
Stage 2: agents generate the compute chiplet (iverilog, yosys, OpenSTA)
========================================================================

iter  fixes applied                                sim                                   synth              timing
---------------------------------------------------------------------------------------------------------------------------------
1     none                                         FAIL wide_product: exp 65328 got 304  -                  -
2     widen_product_register                       FAIL sync_clear: exp 0 got 101583     -                  -
3     implement_sync_clear,widen_product_register  pass (594 checks)                     pass (1044 cells)  pass (slack +4.70 ns)

CONVERGED in 3 iteration(s)

========================================================================
Stage 3: agents generate the fabric endpoint (CRC32 datapath, same signoff)
========================================================================

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
board class                         instances  clock MHz  GMAC/s  xcvr Gbps  endpoint Gbps  link Gbps
----------------------------------  ---------  ---------  ------  ---------  -------------  ---------
             small (Artix-7 class)          5      150.0    0.75        6.6           4.80       4.80
mid (Kintex/Zynq UltraScale class)         65      188.7   12.26       16.3           8.04       8.04
        large (Versal/Alveo class)        270      188.7   50.94       32.0           8.04       8.04
   link rate = min(transceiver, synthesized endpoint): the endpoint gates
   the mid and large boards, an honest measured limit, not a datasheet number

========================================================================
Stage 5: the right fabric: smallest cluster per board class that hosts the model
========================================================================

analytic sizing vs the 500 tok/s target
board class   boards needed  predicted tok/s  comm fraction
------------  -------------  ---------------  -------------
artix7_small    unreachable       126 @ n=16
 zynq_us_mid              8              791            31%
versal_large              1              600             0%

candidate sweep on the mid class (compute shrinks with n, all-reduce grows)
boards  pred tok/s  compute/token  comm/token
------  ----------  -------------  ----------  ------
     1         144        6.93 ms        0 ns
     2         276        3.46 ms    166.9 us
     4         497        1.73 ms    280.5 us
     8         791       865.7 us    397.8 us  chosen
    16         990       432.8 us    577.4 us

========================================================================
Stage 6: host the model: decode on the chosen fabric, every all-reduce real traffic
========================================================================

8 x zynq_us_mid hosting gpt2_124m
quantity               value     provenance
---------------------  --------  -------------------------------------------------
               boards         8                        chosen by sizing in stage 5
       tokens decoded         8                        full n_layer loop per token
       measured tok/s       753  discrete-event fabric, packetized + CRC + credits
      predicted tok/s       791                        analytic model from stage 5
     prediction ratio      0.95                               measured / predicted
collectives per token        24                          n_layer * 2 = 24 expected
           wire bytes  17461248                           headers and CRC included
activations identical      True          every board holds the same reduced vector
   target met: 753 tok/s measured vs 500 required

========================================================================
Stage 7: scale by adding boards, and survive a lossy fabric
========================================================================

same synthesized fabric, more zynq_us_mid boards
boards  tok/s  time/token  activations identical
------  -----  ----------  ---------------------
     2    274     3.66 ms                   True
     4    487     2.05 ms                   True
     8    753     1.33 ms                   True
    16    926     1.08 ms                   True
   throughput scales until the ring all-reduce term (2*(n-1) steps) pushes back

   BER 1e-6 on every link, 8 boards: 64 CRC drops, 464 retransmits,
   activations still identical on every board = True, throughput 315 tok/s
   (2.4x slower than the clean fabric: reliability costs latency, never bits)

========================================================================
Stage 8: numerics check at reduced dimensions (real arithmetic through the fabric)
========================================================================
   homogeneous 4-board MLP 64x64 W1 64x512 W2 512x64: time 146.7 us, max err vs single-board reference 8.9e-15
   heterogeneous 2 small + 2 large: bit-identical output to homogeneous = True
   work-proportional shards ([4, 4, 252, 252]) recover 11.21x over the equal split

results written to results.json
```

Reading the numbers: the agent ships a truncated product register in the chiplet and a missing final inversion in the CRC endpoint; both testbench mismatches are parsed into structured feedback and both flows converge. The measured endpoint (4 bytes/cycle at 251 MHz = 8.04 Gbps) becomes the usable link rate on the mid and large boards, gating their 16.3 and 32 Gbps transceivers, so the sizing answers reflect the fabric the agents actually built. For GPT-2 124M at 500 tok/s, the small class can never get there (16-board cap reaches 126 tok/s), the mid class needs 8 boards, and one large board suffices. The chosen 8-board configuration then hosts decode with every all-reduce as real link traffic and lands at 753 tok/s, within 5% of the analytic prediction. Adding boards keeps scaling throughput (926 tok/s at 16) until the ring's 2\*(n-1) serialization term dominates, and at BER 1e-6 the fabric retransmits its way to the same bit-exact activations at 2.4x the latency.

## How an LLM agent slots in

`agent.py` exposes one interface: `propose(spec, feedback_history) -> (verilog_source, notes)`. The rule-based `RuleBasedAgent` maps parsed mismatches to code revisions; an LLM agent implements the same method by serializing the spec and the structured feedback history (stage, failing test, expected vs observed values, tool errors) into a prompt and returning the model's Verilog. The orchestrator (`chiplet_flow.py`) already runs two different specs through the identical loop (the MAC chiplet and the CRC endpoint), so a third block is one more spec and testbench, not new plumbing. Because everything downstream only reads the profiles and the model spec, an LLM proposing a completely different microarchitecture (more pipeline stages, a wider CRC datapath, a slower but smaller core) flows through automatically: its measured cycles per unit, cell count, and fmax simply produce a different fit, a different link rate, and different sizing answers.

## Honest simplifications

- Generic liberty cells are not FPGA LUTs: `lut_capacity_proxy` is in generic-cell units so the yosys cell count maps onto it directly. Real capacity fit needs LUT/FF/DSP utilization from vendor place-and-route.
- There is no place-and-route anywhere; timing is real OpenSTA static timing but against a toy illustrative liberty, so fmax is an estimate of an estimate.
- Boards are simulated, not real: link rates, propagation delays, clock caps, and capacities are representative class parameters, not measured silicon.
- The 30% fabric reservation for NIC and routing logic is a stated guess, not a floorplan; the endpoint reservation per link is real, from the synthesized cell count.
- Decode compute time is charged from the measured chiplet profile at full model dimensions; the weights are not materialized. The sharded numerics (real arithmetic, bit-exactness across cluster shapes) are validated at reduced dimensions in stage 8 and in the test suite.
- Activations travel as 8-byte IEEE doubles end to end so every all-reduce check is exact; a real deployment would use fp16 or fp32 and halve or quarter the fabric traffic.
- ACK/credit control packets share link bandwidth but are assumed error-free (in RTL they are short, heavily protected control words); topology is a full mesh of point-to-point links; payloads are fixed 1024 B with a 20 B header.
- The compute cycle model charges cycles per unit from the profile; it ignores on-board operand distribution to the chiplet array and memory bandwidth limits.
- The "agents" shipped here are rule-based (see above), so convergence in 3 and 2 iterations demonstrates the loop mechanics, not LLM capability.
