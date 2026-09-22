# Capstone plan

Working name: this repository, `agentic-fpga-scaleout`. The project name is still an open decision.

Task tracking lives in GitHub issues, one per workstream, listed at the bottom. This file holds the reasoning that does not belong in an issue body.

## Read this first: the Basys 3 will not run GPT-2

The Basys 3 has **no external memory of any kind**. It has 225 KB of on-chip block RAM and no Ethernet. GPT-2 124M needs 84.9 MB of weights at int8. Three boards give 0.68 MB between them, which is short by a factor of **126**. No number of boards fixes this, because adding boards adds compute we already have far too much of.

That matters because three boards at 350 dollars each is 1,050 dollars for hardware that cannot hold the model.

### What the Basys 3 is actually good at

Our own sizing model says decode is fast exactly when the weights fit in on-chip memory, because then there is no memory wall at all. The Basys 3 is a pure on-chip machine, so it lands in that regime by construction. Three of them hold a real transformer of about **330 KB** (3 layers, width 96, context 128) entirely in block RAM, and the model predicts tens of thousands of tokens per second, limited by the link between boards rather than by memory.

It also has **four Pmod headers**, and a ring needs two per board, so a three board ring works with ribbon cables and no switch. We do need a link, but it does not have to be Ethernet. The generated endpoint needs framing, a checksum and flow control, all of which we already generate.

### Three honest paths

| Path | Hardware | What we demo | Extra cost |
|---|---|---|---|
| Tiny model, three boards | 3 x Basys 3 plus Pmod cables | Agent-designed hardware hosting a small transformer across three FPGAs over a generated link. Full scaleout story. | about 700 |
| Real GPT-2, one board | 1 used board with DRAM, such as a KC705 | GPT-2 124M generating real text on agent-designed hardware. No scaleout. | about 350 |
| Both | Keep the Basys 3 boards and add one KC705 | Scaleout on the small boards, real GPT-2 on the big one. Strongest demo. | about 1,050 |

**Recommendation:** decide at the Sept 23 meeting before any purchase. The scaleout story is what makes this different from Redwood, so if only one path is funded, take the tiny model on three boards.

## Status

The software half is real, runs end to end in about 30 seconds, and is covered by 66 automated checks. That is not where the risk is.

| Measure | Value |
|---|---|
| Blocks generated and signed off by agents | 2: the compute unit and the fabric endpoint |
| Automated signoff gates | 4: simulation, synthesis, timing, FPGA mapping |
| Automated checks passing | 66 |
| Lines of our RTL that have run on real silicon | 0 |

Working today: hardware derived from the model spec, the fabric endpoint derived from the link rate with an automatic retry when timing fails, real device budgets and resource mapping, three transports compared, batching, decode validated in a packet-level fabric simulator, and a real LLM writing the RTL through the same loop.

Not done: anything on a board.

## Lanes

Each person owns a lane end to end so two people are never editing the same files. Yax and Sameer can work entirely in software, so nobody is blocked waiting for hardware.

| Person | Discord | Lane | Scope |
|---|---|---|---|
| **Sameer** | sami | Lead, proposal, agents and sizing software | Owns spec derivation, the sizing model, the repo, and every written deliverable. Decides scope. |
| **Yax** | Octane98 | Agent swarm and the fabric | Owns the multi-agent architecture and the literature behind it, plus the interconnect and collectives. All software, so not blocked on hardware. |
| **Vaibhav** | wabbadedabbadi | Board bring-up and the physical flow | Has the only board. Owns Vivado, place and route, bitstreams, and everything that runs on real silicon. |
| **Siddh** | original_heisenberg | Hardware sourcing, the link, and verification | Owns getting more boards, designing the board-to-board cable, and raising the verification bar to something we can defend. |

## Milestones

Each is independently demonstrable, so we can stop at any point and still have a result.

1. **Working simulation.** Done. Agents generate and sign off both blocks, sizing picks a cluster, decode runs in the fabric simulator.
2. **Agents that iterate.** An LLM agent fixes a block it got wrong, using nothing but parsed tool output. Needs C4.
3. **Bitstream on a board.** Agent-written RTL placed, routed, loaded and verified on the Basys 3. Needs A6, A8, E1.
4. **One board hosting a model.** A real small transformer generating recognizable text on one FPGA. Needs D4, D6, G.
5. **Three boards over a generated link.** All-reduce across three real FPGAs, bit-exact, measured against the prediction. Needs B. This is the distinctive result.
6. **Optional: silicon.** The compute unit hardened to GDSII on an open process and submitted to a shuttle, for about 300 dollars.

## Calendar

| What | When | Owner |
|---|---|---|
| Team meeting | Sept 23, 5 to 6 pm | Everyone |
| Team meetings after that | Weekly, same slot unless changed | Sameer |
| Bi-weekly updates with the professor | Start mid-October, exact date unconfirmed | Sameer, see H1 |
| Proposal | Deadline unconfirmed, waiting on A1 | Sameer |
| Progress checkpoints, demo day, final report | All unconfirmed | Sameer, see H3 |

Three of five rows say unconfirmed, which is our biggest planning gap. Hardware has shipping and bring-up time in front of it, so it has to be scheduled backwards from demo day rather than forwards from today.

## Open decisions

1. **Which hardware path.** Tiny model on three Basys 3, real GPT-2 on one DRAM board, or both. Decide Sept 23.
2. **Does the department reimburse, and up to how much.** Sameer is asking Noura. A5 waits on the answer.
3. **Are these lanes right.** The table above is a proposal based on who said what in chat. Anyone can trade.
4. **What do we call this.** The repo is named `agentic-fpga-scaleout` and nothing else has been agreed. The report, slides and professor updates all need one name.
5. **What counts as done for the capstone.** Agree the minimum we commit to the professor, separately from what we hope to reach. M3 is a defensible floor, M5 is the ambition.

## Risks

| Risk | Why it bites | What we do |
|---|---|---|
| Buying the wrong hardware | Three Basys 3 cannot run GPT-2, and that is 1,050 dollars | Settle A2 before A5 |
| Unknown deadlines | We cannot schedule backwards from dates we do not have | H1 and H3, this week |
| One person holds the only board | Everything physical serialises through Vaibhav | Yax and Sameer work purely in software; get board two early |
| Agent iteration unproven | Our headline claim is agents fixing hardware from tool feedback, and only the rule-based agent has shown it | C4. Until then describe it accurately in writing |
| Place and route surprises | Designs that pass synthesis routinely fail after routing, and ours has never been routed | E1 early, on the small blocks |
| Scope | C, D, E and G are each multi-week and all feed the flagship demo | Milestones ordered so M3 alone is a complete result. Cut silicon first, then formal verification |

## Workstreams

Tasks live in the issues below. Tick items there as you go.

- **A. Hardware: decide, source, bring up** (Vaibhav): 8 items
- **B. The board-to-board link** (Siddh): 6 items
- **C. Agent swarm** (Yax): 6 items
- **D. Model and sizing** (Sameer): 7 items
- **E. Physical implementation** (Vaibhav): 4 items
- **F. Verification** (Siddh): 4 items
- **G. Serving layer** (Sameer): 4 items
- **H. Deliverables and the professor** (Sameer): 8 items

