# Capstone plan

Working name: the repository is `agentic-fpga-scaleout`. The project name is still an open decision.

**How to use this doc.** Every task has a checkbox and an owner. Tick yours as you go. Item ids like A1 and C4 are stable, so use them in Discord instead of describing the task again. If a checkbox does not render as a real checklist, select the list and use Format, then Bullets and numbering, then Checklist.

Repo: https://github.com/SameerSul/agentic-fpga-scaleout

---

## 1. Timeline

**The presentation is September 30. That is the date everything else is scheduled backwards from.**

| Date | What | Owner |
|---|---|---|
| Tue Sep 23 | Team meeting, 5 to 6 pm. Decide the presentation story and the board question | Everyone |
| Wed Sep 24 | Report sections drafted. Vaibhav reports synthesis status | Everyone |
| Thu Sep 25 | Report content complete, Sameer assembles | Sameer |
| Fri Sep 26 | Report finalized and submitted | Sameer |
| Sat Sep 27 | Slides drafted from the report | Sameer, Yax |
| Sun Sep 28 | Full rehearsal, timing checked | Everyone |
| Mon Sep 29 | Fixes from rehearsal. Record a fallback demo video | Vaibhav |
| **Tue Sep 30** | **Presentation** | Everyone |
| Mid Oct | Bi-weekly updates with the professor begin, exact date unconfirmed | Sameer, see H1 |
| Unconfirmed | Progress checkpoints, demo day, final report | Sameer, see H3 |

After Sep 30 the project moves to the milestone track in section 9. The hardware work runs from October onward.

---

## 2. The September 30 presentation

### There will be no hardware demo on the 30th, and that is fine

Nothing ordered this week will ship, arrive and be brought up in seven days. Deciding that now is much better than discovering it on the 29th. The software result is strong enough to carry the presentation on its own.

### What we actually have to show

- Agents generate two blocks of real Verilog and carry them through four signoff gates: simulation, synthesis, timing, and FPGA mapping
- The hardware is derived from the model. Datapath width comes from the quantization, accumulator width from the deepest reduction in the network, so the accumulator mathematically cannot overflow
- The fabric endpoint is derived from the link rate. When the obvious datapath missed its timing target by 0.02 nanoseconds, the flow widened it, halved the clock, reran the whole loop and signed off with 1.1 nanoseconds to spare. That is the system changing what it was building, not just how it was written
- A sizing layer answers how many boards a model needs, accounting for compute, DDR bandwidth, on-chip memory residency, KV cache traffic and network cost
- Decode validated end to end at 562 tokens per second measured against 566 predicted, with every all-reduce as real packetized, checksummed, flow-controlled traffic
- 66 automated tests, all passing, running in about 30 seconds
- Claude Haiku wrote both blocks correctly on the first attempt through the same loop

### Suggested running order

1. The problem: compute is the bottleneck for AI, and chip design is the bottleneck for compute
2. What Architect Labs proved with Redwood, and the two things they did not do: derive the architecture from the model, and scale past one chip
3. Our loop, with the agent catching its own bug live on a slide
4. The sizing result and the memory wall finding
5. Hardware plan, with the bandwidth analysis in section 4 as evidence we have costed it
6. What is next and what we are asking for

### Presentation tasks

- [ ] **I1** Agree the presentation story and running order (Everyone, Sep 23)
- [ ] **I2** Assign report sections to each person (Sameer, Sep 23)
- [ ] **I3** Draft your report section (Everyone, Sep 24)
- [ ] **I4** Assemble and finalize the report (Sameer, Sep 25 to 26)
- [ ] **I5** Build the slides from the report (Sameer and Yax, Sep 27)
- [ ] **I6** Full rehearsal with timing (Everyone, Sep 28)
- [ ] **I7** Record a fallback demo video in case anything live fails (Vaibhav, Sep 29)
- [ ] **I8** Decide who presents which section (Everyone, Sep 23)
- [ ] **I9** Pick the project name (Everyone, Sep 23)
- [ ] **I10** Set up the AI note taker for meetings, convene.ai was suggested (Siddh)

---

## 3. Open decisions

1. **Which Zybo variant is it, and what is the second FPGA?** The Z7-20 has 220 DSP slices, the Z7-10 has 80. That decides whether we are memory-bound or compute-starved. Nobody has confirmed the second board at all.
2. **Confirm: proposal and presentation are P0.** Yax raised this and he is right. Everything in sections 8 C through G is October work. This week is writing.
3. **What are we presenting on Sep 30?** See section 2. Recommend leading with the software result and the Redwood comparison.
4. **Agree we are not buying anything this week.** Two boards are enough, OSAP has not landed, and nothing ordered now arrives in time. Revisit in October.
5. **Pick the name.** Candidates so far: fpgAI, fpgLLM, LLMArray, distLLM. The report and slides both need one, so decide tonight rather than renaming later.
6. **Which repo is the project?** Vaibhav asked about `fpga-interconnect`, but the combined work lives in `agentic-fpga-scaleout`. Make sure everyone is in the right place before writing code.
7. **Who writes and who presents which section of the report?**
8. **Is the AMD discount available through Vaibhav?** Only matters for October, but worth knowing now.

Parked, not for tonight: photonic interconnects, and the jake@etched.com contact. Both are good post-presentation threads.

## 4. Three technical questions, answered

### How much RAM do we need (Siddh)

Qwen3-0.6B at int8 needs **294 MB of block weights**, plus about **59 MB of KV cache per concurrent sequence** at a context of 1024. So:

- **1 GB is comfortable.** Every board being considered has this.
- 512 MB is tight once KV cache is counted.
- 256 MB is not enough.

The 1 GB boards Siddh found are correctly sized. RAM is not the thing to optimize.

### How much Ethernet do we need

From our own sizing model, on two boards:

| Link speed | Tokens per second | Time spent communicating | Limited by |
|---|---|---|---|
| 100 Mbps | 69 | 89% | the network |
| 1 Gbps | 342 | 44% | memory |
| 2.5 Gbps | 464 | 24% | memory |
| 10 Gbps | 565 | 7% | memory |
| 25 Gbps | 570 | 7% | memory |

- **Vaibhav is right that 1 Gbps is enough.** It delivers 60 percent of what 10 Gbps does.
- **Going past 10 Gbps buys nothing.** The limit becomes DDR bandwidth, not the wire.
- **100 Mbps is the real floor** and is too slow.
- **10 Gbps costs double**, since both boards need it or the link runs at the slower end.

**Recommendation: stop searching for 10 Gbps boards.** Buy 1 Gbps. If a board has two ports, bond them, which lands between the 1 and 2.5 Gbps rows for free. Spend the saved money on a second board rather than a faster one.

### Is ring all-reduce better than master-slave (Siddh)

**Yes, at every board count, and the gap grows.** Numbers for our actual message size of 64 KB per all-reduce:

| Boards | Ring | Master-slave | Ring advantage |
|---|---|---|---|
| 2 | 542 us | 1,083 us | 2.0x faster |
| 3 | 723 us | 2,164 us | 3.0x faster |
| 4 | 814 us | 3,245 us | 4.0x faster |
| 8 | 954 us | 7,571 us | 7.9x faster |

The reason is simple. In a ring every board sends and receives at the same time, so all links work at once. In master-slave every board sends to one node, so the master's single link carries all the traffic and everyone else waits. Ring moves about 2S bytes per node regardless of board count; master-slave moves (n-1)S through one link.

The one case where master-slave wins is very small messages, where the ring's extra hops cost more latency than the bandwidth saving is worth. Our messages are 64 KB, which is firmly in ring territory. We already implement ring, so this is a question we can close.

## 5. Hardware

### The model target is now Qwen3-0.6B

Sameer's call, and it is the right one: it is a real modern model, it is the **same model Architect Labs ran on Redwood**, which gives us a direct comparison, and it fits the boards we have.

### What we own

| Board | Who | DDR | Ethernet | Can it host Qwen3-0.6B |
|---|---|---|---|---|
| Zynq 7000 Zybo Z7 | Sameer | 1 GB DDR3L | 1 Gbps | Yes |
| Second FPGA, model unconfirmed | Sameer | ? | ? | Confirm at the meeting |
| Basys 3 | Vaibhav | **none** | none | **No** |

### Predicted throughput on Qwen3-0.6B at batch 8

| Board | Compute units | 1 board | 2 boards | Limited by |
|---|---|---|---|---|
| Zybo Z7-20 (XC7Z020) | 154 | 26.2 tok/s | 43.3 tok/s | memory |
| Zybo Z7-10 (XC7Z010) | 56 | 19.9 tok/s | 34.3 tok/s | compute |
| Alinx AX7102, 374 USD | 168 | 31.4 tok/s | 50.2 tok/s | memory |

**Confirm which Zybo variant it is.** The Z7-20 has 220 DSP slices against the Z7-10's 80, which is the difference between being memory-bound and being compute-starved.

### The headline for the presentation

Architect Labs ran **the same model** on an AMD Versal VPK180, list price **17,995 dollars**, and measured **12.1 tokens per second**. Our model predicts **43.3 tokens per second on two Zybo Z7-20 boards**, which cost roughly 600 dollars together.

That is **about 3.6 times their throughput at 3 percent of their hardware cost**, on their model. Caveat it honestly: ours is a prediction from a model without place and route, theirs is silicon that really ran, and their design was a deliberately scaled-down 2x2 tile. Even discounted heavily, the point stands, and it is the strongest slide in the deck.

### The Basys 3 still cannot host a model

It has no external memory at all, just 225 KB of on-chip block RAM and no Ethernet, against Qwen's 294 MB. Do not buy more of them for hosting. It is still useful for two things: early bring-up practice, and Pmod link experiments, since it has four Pmod headers.

### On buying more

Nothing needs to be bought this week. Two boards are enough for the scaleout demo, OSAP has not landed for either Sameer or Yax, and nothing ordered now arrives before the presentation. Revisit in October. If a third board is wanted then, the Alinx AX7102 at 374 USD is well matched, and Vaibhav should check whether his AMD discount applies first.

## 6. Where the project stands

| Measure | Value |
|---|---|
| Blocks generated and signed off by agents | 2, the compute unit and the fabric endpoint |
| Automated signoff gates | 4: simulation, synthesis, timing, FPGA mapping |
| Automated tests passing | 66 |
| Decode throughput validated in simulation | 562 tokens per second measured against 566 predicted |
| Lines of our RTL that have run on real silicon | 0 |
| Current model target | Qwen3-0.6B at int8, 294 MB of weights |

Working today: hardware derived from the model spec, the fabric endpoint derived from the link rate with an automatic retry when timing fails, real device budgets and resource mapping, three transports compared, batching, decode validated in a packet-level fabric simulator, and a real LLM writing the RTL through the same loop.

Not done: anything on a board.

---

## 7. Who owns what

Each person owns a lane end to end so two people are never editing the same files. Yax and Sameer can work entirely in software, so nobody is blocked waiting for hardware.

| Person | Lane | Scope |
|---|---|---|
| **Sameer** | Lead, proposal, agents and sizing software | Owns spec derivation, the sizing model, the repo, and every written deliverable. Decides scope |
| **Yax** | Agent swarm and the fabric | Owns the multi-agent architecture and the literature behind it, plus the interconnect and collectives. All software |
| **Vaibhav** | Board bring-up and the physical flow | Owns Vivado, place and route, bitstreams, and everything that runs on real silicon |
| **Siddh** | Hardware sourcing, the link, and verification | Owns getting boards, designing the board-to-board cable, and raising the verification bar |

---

## 8. Task checklists

### A. Hardware: decide, source, bring up

_Before buying anything, settle what the boards can actually run._

- [ ] **A1** Finish the current Basys 3 bring-up and report the result (Vaibhav, critical)
- [ ] **A2** Confirm the Zybo variant and identify the second FPGA (Sameer, critical)
- [ ] **A3** Confirm the reimbursement policy with Noura (Sameer, critical)
- [ ] **A9** Ask the department whether FPGA boards can be signed out from the teaching labs (Siddh, critical)
- [ ] **A4** Price more boards and check for Digilent academic pricing (Siddh)
- [ ] **A5** Revisit buying a third board in October, Alinx AX7102 at 374 USD is well matched (Siddh, October)
- [ ] **A10** Check whether the AMD discount applies through Vaibhav (Vaibhav)
- [ ] **A6** Vivado project template that builds a bitstream from our generated RTL (Vaibhav, critical)
- [ ] **A7** Blink and UART echo on the board (Vaibhav)
- [ ] **A11** Confirm everyone is working in the agentic-fpga-scaleout repo, not fpga-interconnect (Sameer)
- [ ] **A8** Load an agent-generated MAC onto the real board and verify it (Vaibhav, critical)

### B. The board-to-board link

_Our generated endpoint does not care what the wire is. It needs framing, a checksum and flow control._

- [ ] **B0** Work out the achievable data rate over a Pmod ribbon on paper (Siddh, critical, no hardware needed)
- [ ] **B1** Design the Pmod-to-Pmod link: pinout, signalling, cable, grounds (Siddh, critical)
- [ ] **B2** Measure the maximum reliable bit rate over the cable (Siddh)
- [ ] **B3** Re-derive the fabric endpoint for the measured link rate (Yax)
- [ ] **B4** Link up and frame between two real boards (Yax, critical)
- [ ] **B5** Ring all-reduce across the boards, bit-exact against one board (Yax, critical)
- [ ] **B6** Closed: ring all-reduce beats master-slave at every board count, see section 4 (Siddh)
- [ ] **B7** Decide whether to bond two Ethernet ports on boards that have them (Yax, October)

### C. Agent swarm

_Today there is one agent with one job. A swarm means several roles that check each other._

- [ ] **C1** Summarize the key papers: Redwood, MAGE, ChipNeMo, AlphaChip, VeriGen (Yax, critical, needed for the report)
- [ ] **C2** Write the multi-agent architecture: roles, handoffs, who sees which tool output (Yax, critical)
- [ ] **C3** Implement a second agent role beyond the single propose call (Yax)
- [ ] **C4** Make the LLM agent iterate on a block hard enough to fail first try (Yax, critical, our weakest claim)
- [ ] **C5** Build a comparison harness: which model, how many iterations, what it cost (Yax)
- [ ] **C6** Have an agent generate the testbench as well as the RTL (Yax)

### D. Model and sizing

_The software model has to match the hardware we actually buy._

- [ ] **D1** Add Zybo Z7 and Alinx AX7102 to boards.py with real numbers (Sameer, critical)
- [ ] **D2** Add the Qwen3-0.6B spec: 28 layers, d_model 1024, d_ff 3072, int8 (Sameer, critical)
- [ ] **D3** Model the Pmod link as a transport alongside Aurora and Ethernet (Sameer)
- [ ] **D4** Download and quantize the real Qwen3-0.6B checkpoint (Sameer, critical)
- [ ] **D5** Quantize it to int8 with per-channel scales (Sameer)
- [ ] **D6** Write a bit-accurate Python reference for the full int8 path (Sameer, critical)
- [ ] **D7** Keep GPT-2 124M as the simulated large-model result (Sameer)

### E. Physical implementation

_The biggest gap. We synthesize and map to primitives but never place, route, or build a bitstream._

- [ ] **E1** Add a Vivado place-and-route stage to the flow (Vaibhav, critical)
- [ ] **E2** Parse real utilization and post-route timing into the profile (Vaibhav, critical)
- [ ] **E3** Feed post-route timing failures back to the agent (Sameer)
- [ ] **E4** Replace the toy-library clock estimate with measured post-route timing (Sameer)

### F. Verification

_Redwood claimed 95 percent functional coverage. We claim golden-model testbenches, which is weaker._

- [ ] **F0** Write the board bring-up and acceptance test plan (Siddh, critical, no hardware needed)
- [ ] **F1** Add coverage instrumentation to the generated testbenches (Siddh, critical)
- [ ] **F2** Formal proof that the accumulator can never overflow (Siddh)
- [ ] **F3** Formal proof that the CRC matches the reference for all frame lengths (Siddh)
- [ ] **F4** On-board self-test that runs after a bitstream load (Siddh)

### G. Serving layer

_Something has to tokenize, batch, send, and stream results back._

- [ ] **G1** Tokenizer and detokenizer on the host (Sameer)
- [ ] **G2** Host to board transport over USB UART (Yax)
- [ ] **G3** Request queue and batching scheduler (Yax)
- [ ] **G4** A screen showing tokens streaming with per-board activity (Sameer)

### H. Deliverables and the professor

- [ ] **H1** Confirm the exact date and format of the first bi-weekly update (Sameer, critical)
- [ ] **H2** Finish the proposal (Sameer, critical)
- [ ] **H3** Confirm all remaining capstone deadlines (Sameer, critical)
- [ ] **H4** Build a reusable update deck template for the bi-weekly meetings (Siddh)
- [ ] **H5** Keep the explainer brief current (Sameer)
- [ ] **H6** Keep the repo README current (Sameer)
- [ ] **H7** Demo day script with a recorded fallback (Everyone)
- [ ] **H8** Final report (Everyone)

---

## 9. Scope tiers: MVP, core, stretch

Three tiers, same as the proposal. The point of the split is that **the MVP depends
on neither a hardware purchase nor the unproven LLM iteration**, which are the two
riskiest things in the project.

| Tier | What must work | Modules | Needs |
|---|---|---|---|
| **Bronze, MVP** | Agents generate and sign off both blocks through all five gates. A bitstream from agent-written RTL runs on one board and passes self-test. Simulated cluster hosts Qwen3-0.6B within 15 percent of prediction. 100+ tests passing. | A, C, D | Nothing bought. Both boards already in hand |
| **Silver, core result** | Two boards linked over the generated fabric. Ring all-reduce bit-exact across both, including under injected bit errors. Qwen3-0.6B at 40+ tok/s measured. One block fixed by an LLM from tool output, transcript published. | A, B, C, D | Second board, already owned |
| **Gold, stretch** | Larger models by adding boards. Agent writes the verification as well as the RTL. Interconnect comparison including photonics. Optionally GDSII on a shuttle. | All | Third board, sponsorship |

Cut order if the schedule slips: Gold first, then parts of Silver, never Bronze.

## 9b. Milestones after the presentation

Each is independently demonstrable, so we can stop at any point and still have a result.

1. **Working simulation.** Done. Agents generate and sign off both blocks, sizing picks a cluster, decode runs in the fabric simulator.
2. **Agents that iterate.** An LLM agent fixes a block it got wrong, using nothing but parsed tool output. Needs C4.
3. **Bitstream on a board.** Agent-written RTL placed, routed, loaded and verified. Needs A6, A8, E1.
4. **One board hosting a model.** A real small transformer generating recognizable text on one FPGA. Needs D4, D6, G.
5. **Multiple boards over a generated link.** All-reduce across real FPGAs, bit-exact, measured against the prediction. Needs B. This is the distinctive result.
6. **Optional: silicon.** The compute unit hardened to GDSII on an open process and submitted to a shuttle, for about 300 dollars.

---

## 10. Risks

| Risk | Why it bites | What we do |
|---|---|---|
| Seven days to the presentation | No hardware can arrive and be brought up in time | Present the software result. Decide this today, not on the 29th |
| Buying hardware we do not need | Two boards already cover the scaleout demo, and OSAP has not landed | Buy nothing this week. Revisit in October |
| Chasing 10 Gbps | Doubles cost for throughput we are not bottlenecked on | Section 4. 1 Gbps is enough |
| Unknown later deadlines | We cannot schedule backwards from dates we do not have | H1 and H3 this week |
| One person holds the boards | Everything physical serialises through one person | Yax and Sameer work purely in software |
| Agent iteration unproven | Our headline claim is agents fixing hardware from tool feedback, and only the rule-based agent has shown it | C4. Until then describe it accurately in the report |
| Place and route surprises | Designs that pass synthesis routinely fail after routing, and ours has never been routed | E1 early, on the small blocks |
| Scope creep from good ideas | Photonics and new contacts are tempting with seven days left | Parked until after Sep 30. Proposal and presentation are P0 |

