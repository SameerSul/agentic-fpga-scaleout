# Agentic FPGA Scaleout: Generated Chiplets on a Multi-Board Fabric (Prototype)

We are building a pipeline where agents given an LLM generate the compute chip and its chiplets from a spec, carry them through simulation, synthesis, and timing signoff with no human in the loop, and emit a measured performance profile of what the RTL actually delivers. An FPGA hosts the generated chiplet locally, and a board abstraction layer deploys the same measured profile onto any FPGA class, so scaleout coordination over our reliable interconnect fabric works regardless of FPGA type or board count, including heterogeneous mixes. GDSII tapeout is deliberately out of scope: FPGA hosting is the endpoint by design, which keeps the loop from spec to running multi-board system fast, cheap, and repeatable.

Pure Python 3 stdlib, no dependencies. The RTL half runs real open tools (Icarus Verilog, Yosys, OpenSTA); the scaleout half is a discrete-event simulator in nanoseconds on a `heapq` event queue.

## How the two halves connect: chiplet_profile.json

The contract between the halves is one machine-readable file, written by `chiplet_flow.py` only after the design passes simulation, synthesis, and timing:

| field | where it comes from |
|---|---|
| `cycles_per_mac` | measured by the testbench: a 256-MAC back-to-back burst, span cycles divided by MACs retired (never hardcoded) |
| `latency_cycles` | measured by the testbench, first valid in to first valid out |
| `cell_count`, `area` | yosys `stat` against the generic liberty |
| `fmax_estimate_mhz` | OpenSTA worst slack at the target clock: fmax = 1000 / (period - slack); falls back to a yosys gate-depth proxy if OpenSTA is absent |
| `sim_checks_passed` | self-checking testbench vs a golden model |

`boards.py` consumes the profile with `fit(board, chiplet_profile)`: instances = floor(usable capacity / cell_count), clock = min(measured fmax, board cap), per-board MACs/s = instances * clock * 1e6 / cycles_per_mac. The fabric's `Board` compute model then takes its throughput from that fit result instead of a hand-tuned constant. Nothing upstream changes when the board class changes, and a cluster may mix classes.

## Model assumptions and hardware mapping

| Component | What it models | Real hardware target |
|---|---|---|
| `chiplet_flow.py` + `agent.py` | agent proposes RTL, tools verify, parsed failures feed back until signoff | the LLM-driven RTL generation loop on production verification and synthesis flows |
| `tb_mac.v` | golden-model checking plus a throughput burst that measures cycles/MAC | a full UVM environment plus performance characterization on the real testbench |
| yosys + `cells.lib` | mapping to a tiny generic liberty, cell count as the footprint metric | FPGA synthesis to LUTs/FFs/DSPs with a place-and-route utilization report |
| OpenSTA slack to fmax | achievable clock from static timing at the target period | vendor STA after place-and-route, speed-grade specific |
| `boards.py` profiles | capacity proxy, clock cap, transceiver class per board family | Artix-7, Kintex/Zynq UltraScale, and Versal/Alveo class parts with GTP/GTH/GTY-class serial links |
| `Link` (rate, prop delay, BER knob) | serialization delay, flight time, random corruption | one transceiver lane running an Aurora-style framed stream |
| `_Tx` / `_Rx` in `fabric.py` | seq numbers, CRC32, go-back-N retransmit, cumulative ACKs | the reliable link-layer RTL block: replay buffer in BRAM, ACK sideband |
| Credit flow control | 64-packet RX buffer, sender stalls at zero credits, overflow impossible by construction | RTL credit counters and credit-return words on the reverse channel |
| `Board.dma` | one shared DMA engine per board, 16 KB slices | AXI DMA between DDR/URAM and the link FIFOs |
| `Board.matmul` | instances MAC chiplets in parallel, cycles = M\*N\*K\*cpm/instances + fill/drain, real products | the generated chiplet array the fabric feeds |
| `collectives.py` | ring all-reduce (reduce-scatter + all-gather) for arbitrary N, heterogeneous rings gated by the slowest segment | CCL-style hardware collectives over multi-board topologies |

## How to run

```
python3 demo.py           # end-to-end, all five stages, writes results.json (about 5 s)
python3 tests.py          # 17-check correctness suite (runs the flow once)
python3 chiplet_flow.py   # just the agentic RTL loop, writes chiplet_profile.json
```

Python 3 stdlib only. Requires `iverilog` and `vvp` on PATH; `yosys` and `sta` (OpenSTA) are used when present and skipped gracefully when not (timing then falls back to the gate-depth proxy, clearly labeled). Build artifacts go to `build/`.

## Sample output

```
Agentic FPGA scaleout: chiplet RTL to multi-board fabric, one run
========================================================================
Stage 1: agentic RTL generation and signoff (iverilog, yosys, OpenSTA)
========================================================================
Tools: iverilog=/opt/homebrew/bin/iverilog, vvp=/opt/homebrew/bin/vvp, yosys=/opt/homebrew/bin/yosys, sta=/Users/ssuleman/.local/bin/sta

iter  fixes applied                                sim                                   synth              timing
---------------------------------------------------------------------------------------------------------------------------------
1     none                                         FAIL wide_product: exp 65328 got 304  -                  -
2     widen_product_register                       FAIL sync_clear: exp 0 got 101583     -                  -
3     implement_sync_clear,widen_product_register  pass (594 checks)                     pass (1044 cells)  pass (slack +4.70 ns)

CONVERGED in 3 iteration(s), report written to .../agentic-fpga-scaleout/report.json
chiplet profile written to .../agentic-fpga-scaleout/chiplet_profile.json

========================================================================
Stage 2: measured chiplet profile (the contract between the halves)
========================================================================

chiplet_profile.json (method: opensta_slack)
field              value   provenance
-----------------  ------  ------------------------------------------------
   cycles_per_mac   1.000  measured by testbench burst (span cycles / MACs)
   latency_cycles       4                             measured by testbench
       cell_count    1044                       yosys stat, generic liberty
             area  2066.0                       yosys stat, generic liberty
fmax_estimate_mhz   188.7            OpenSTA: 1000 / (period - worst slack)
sim_checks_passed     594           self-checking testbench vs golden model

========================================================================
Stage 3: the same profile deploys on any board class (boards.py)
========================================================================

fit(board, chiplet_profile), usable fabric fraction = 70%
board class                         instances  clock MHz  util  GMAC/s  link Gbps / lanes
----------------------------------  ---------  ---------  ----  ------  -----------------
             small (Artix-7 class)         13      150.0   68%    1.95            6.6 / 4
mid (Kintex/Zynq UltraScale class)         80      188.7   70%   15.09           16.3 / 8
        large (Versal/Alveo class)        301      188.7   70%   56.79            32 / 16
   no upstream change: instances and clock derive from the one measured profile

========================================================================
Stage 4: tensor-parallel MLP scaleout on the mid board class
========================================================================

MLP x:64x64 W1:64x512 W2:512x64, fused ring all-reduce, zynq_us_mid boards
boards  time      GFLOP/s  speedup  max err vs ref
------  --------  -------  -------  --------------
     1  282.5 us     29.7    1.00x         0.0e+00
     2  163.4 us     51.3    1.73x         1.2e-14
     4  105.4 us     79.6    2.68x         8.9e-15
     8   79.6 us    105.4    3.55x         9.8e-15
   sharded output matches the single-board reference on every run

========================================================================
Stage 5: heterogeneous cluster, same chiplet, mixed board classes
========================================================================

equal shard split (2 small + 2 large), time 604.2 us, max err 8.9e-15
board         columns  compute utilization
------------  -------  -------------------
artix7_small      128                  89%
artix7_small      128                  89%
versal_large      128                   3%
versal_large      128                   3%

work-proportional split, time 104.5 us, max err 1.1e-14
board         columns  compute utilization
------------  -------  -------------------
artix7_small        8                  34%
artix7_small        8                  34%
versal_large      248                  37%
versal_large      248                  37%
   proportional split is 5.78x faster than equal split on the mixed cluster (slow boards no longer gate the ring)

results written to results.json
```

Reading the numbers: the agent ships a truncated product register in iteration 1, the testbench mismatch is parsed into structured feedback, iteration 2 exposes the missing synchronous clear, iteration 3 passes everything. The measured profile (1.0 cycles/MAC, 1044 cells, 188.7 MHz fmax) then fits 13, 80, or 301 instances on the three board classes with no upstream change. The MLP scales to 3.55x on 8 mid boards (a small, communication-heavy layer, so the ring all-reduce caps the speedup). On the mixed cluster, equal sharding leaves the large boards 3% utilized because the small boards gate the ring; splitting columns in proportion to each board's measured MACs/s recovers a 5.78x speedup and lands within 1% of the 4-mid-board homogeneous time.

## How an LLM agent slots in

`agent.py` exposes one interface: `propose(spec, feedback_history) -> (verilog_source, notes)`. The rule-based `RuleBasedAgent` maps parsed mismatches to code revisions; an LLM agent implements the same method by serializing the spec and the structured feedback history (stage, failing test, expected vs observed values, tool errors) into a prompt and returning the model's Verilog. The orchestrator (`chiplet_flow.py`), testbench, feedback parsers, profile format, and everything downstream (boards, fabric, collectives) are agent-agnostic and need no changes. Because the scaleout half only reads `chiplet_profile.json`, an LLM proposing a completely different microarchitecture (more pipeline stages, a slower but smaller core) flows through automatically: its measured cycles/MAC, cell count, and fmax simply produce a different fit and different cluster numbers.

## Honest simplifications

- Generic liberty cells are not FPGA LUTs: `lut_capacity_proxy` is in generic-cell units so the yosys cell count maps onto it directly. Real capacity fit needs LUT/FF/DSP utilization from vendor place-and-route.
- There is no place-and-route anywhere; timing is real OpenSTA static timing but against a toy illustrative liberty, so fmax is an estimate of an estimate.
- Boards are simulated, not real: link rates, propagation delays, clock caps, and capacities are representative class parameters, not measured silicon.
- The 30% fabric reservation for NIC and routing logic is a stated guess, not a floorplan.
- ACK/credit control packets share link bandwidth but are assumed error-free (in RTL they are short, heavily protected control words); topology is a full mesh of point-to-point links; payloads are fixed 1024 B with a 20 B header; floats travel as 8-byte IEEE doubles so numerical checks are exact.
- The compute cycle model charges M\*N\*K\*cpm/instances plus a fill/drain term; it ignores on-board operand distribution to the chiplet array and memory bandwidth limits.
- The "agent" shipped here is rule-based (see above), so convergence in 3 iterations demonstrates the loop mechanics, not LLM capability.
