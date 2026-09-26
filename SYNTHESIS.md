# Synthesising this on your board

For Yax. This is what the repo produces, how to build it, and what you
should see. Everything here has been run; if your machine disagrees, that
is worth knowing and worth reporting.

## What this is in one paragraph

You give it a model spec, a small JSON file naming a transformer's
dimensions and its quantization. It derives the hardware that model
needs, writes the Verilog, and checks it with real tools until it passes.
Nine blocks come out, each one sized from the model rather than written
for it. Point it at a different model and every block re-derives.

## The picture

```
        model_spec.json                     one file you edit
   (d_model, d_ff, weight_bits, ...)
                |
                v
        +---------------+
        |   specgen.py  |  derives each block's spec AND its testbench
        +---------------+  golden values computed in Python, not by the
                |          design, so the design cannot mark its own work
                v
        +---------------+
        |     agent     |  writes the Verilog
        +---------------+  rules  = deterministic, free, always works
                |          swarm  = a real LLM, three roles
                v
   +-------------------------------+
   |  iverilog -> yosys -> OpenSTA |  the gates. Anything that fails
   |     -> FPGA map -> mutation   |  comes back as parsed feedback and
   +-------------------------------+  the agent tries again
                |
                v
        signed-off RTL + a measured profile
                |
                v
        +---------------+
        |  bitstream.py |  nextpnr place and route, icepack
        +---------------+  then the packed bits are unpacked and
                           re-simulated against the same testbench
```

## The nine blocks

Arithmetic, bottom to top:

```
   mac        multiply-accumulate, signed, the thing that does the work
   requant    scale, round, saturate between two matmuls
   exp        e^x by table and shift, for softmax
   recip      1/x, for softmax's denominator
   rsqrt      1/sqrt(x), for RMSNorm
   crc32      the fabric link endpoint, for board to board
```

Sequencing, which is what makes the above into a layer:

```
   matvec     walks a weight matrix, drives the MAC, flags each column
   wmem       weight tile plus its loader, streaming write, BRAM read
   softmax    contains exp and recip, does the whole row
   mlp        two matmuls with requantize and relu between, owns the
              activation banks. This is the one that is a layer.
```

## Build it

```
brew install icarus-verilog yosys            # required
brew install nextpnr-ice40 icestorm          # optional, for bitstreams
```

On Ubuntu use the oss-cad-suite release, which carries all of these
except OpenSTA. Without OpenSTA the flow says so and falls back to a
gate-depth estimate, which is an estimate and not timing closure.

```
python3 tests.py        # expect: all 201 tests passed (199 without OpenSTA)
python3 chiplet_flow.py # the agent loop, one block at a time
python3 sweep.py        # every block at every spec, about 20 minutes
```

## Synthesise for your part

The repo targets a Lattice iCE40 because that is the only family with an
open place and route flow. Your Zybo is a Zynq 7020, which needs Vivado.
The RTL is plain Verilog 2005 with no vendor primitives, so it goes
straight into a Vivado project.

Generate the RTL for the current model spec:

```
python3 chiplet_flow.py                # writes mac.v and crc.v into build/
CHIPLET_BUILD_DIR=out python3 -c "
import chiplet_flow as cf
for job in (cf.CHIPLET_JOB, cf.REQUANT_JOB, cf.EXP_JOB, cf.RECIP_JOB,
            cf.RSQRT_JOB, cf.MATVEC_JOB, cf.WMEM_JOB, cf.SOFTMAX_JOB,
            cf.MLP_JOB):
    cf.run_flow(job, verbose=False, agent=cf.make_agent('rules'))
"
```

Everything lands in `out/`. The files you want are the block RTL plus the
generated tables:

```
   mac.v  requant.v  expu.v  recip.v  rsqrt.v  crc.v
   matvec.v  wmem.v  softmax.v  mlp.v
   exp_rom.v  recip_rom.v  rsqrt_rom.v      the constant tables
```

Four blocks instantiate others, so bring their dependencies along:

```
   softmax.v  needs  expu.v recip.v exp_rom.v recip_rom.v
   matvec.v   needs  mac.v      (at the top level, not inside)
   wmem.v     needs  nothing, but is driven by matvec and mac
   mlp.v      needs  matvec.v mac.v requant.v
```

The testbenches are in the repo root as `tb_*.v` if you want to run them
in Vivado's simulator rather than iverilog.

## What it should synthesise to

Measured here with yosys against a generic library, for Qwen2.5-0.5B:

```
   block      fmax      cells     note
   wmem       333 MHz   61712     huge only because the generic library
                                  has no BRAM; on your part it is 1 BRAM
   recip      206 MHz    1010
   matvec     187 MHz    1294
   rsqrt      170 MHz     841
   exp        164 MHz    1428
   mac        169 MHz    1473     infers one DSP48
   mlp        119 MHz   10396
   requant    115 MHz   10445
   softmax    103 MHz    1466
```

A Zynq 7020 has 53200 LUTs and 220 DSPs, so all nine fit with room over.
Expect better numbers than these: the generic library has no carry chain
and no block RAM, and Vivado has both.

## The model it is sized for

Qwen2.5-0.5B, because it is the largest that fits your board:

```
   int8 weights     494 MB of the 1 GB DDR3L      49 percent
   KV cache         6.3 MB at 1024 context        14 query heads over
                                                  only 2 KV heads
   throughput       4.0 tokens/s on one board
                    8.1 on two, weights split
```

That ceiling is memory bandwidth, not arithmetic. Batch-1 decode reads
every weight once per token, so the MAC sits idle most of the time and
adding more of them changes nothing. This is the single most important
number in the project and it is the reason the fabric exists at all.

## Changing the model

Edit `model_spec.json` and rerun. Nothing else. The accumulator width is
`weight_bits + activation_bits + ceil(log2(max(d_model, d_ff)))`, the
pipeline depth follows the datapath width, the address widths follow the
matrix shape. Ten published architectures have been checked this way,
from SmolLM2 135M to Llama 3.1 8B, at four quantizations each: 40 of 40
derive and simulate.

## Which agent to use

`chiplet_flow.py --agent rules` is the deterministic one. It writes every
block, every time, for free, and it is what generated the RTL you are
about to synthesise. Use it.

`--agent swarm` puts a real LLM in the writing seat. Measured across all
nine blocks, it writes the structural ones (the MAC, the matmul
sequencer, the weight memory) and it does not write the ones whose
correctness is exact fixed-point arithmetic (the exponential, the
reciprocal, the inverse square root). That is a real result rather than a
tuning problem: a block that needs a shift to be exactly right, at the
same time as a table index and a clamp, is a poor fit for a model, and
the tools accept nothing approximate. Use the swarm to watch the loop
work, not to produce the RTL you build with.

## Two things that look like failures and are not

The flow is meant to fail its first iterations. The agent gets parsed
tool output rather than hints, and the fixes it applies are derived from
what the simulator said. A run that converges immediately means the
seeded bugs did not trip, which is the case worth investigating.

Some sweep rows say "did not converge in 5 iterations" and still count
clean. Those are the endpoint's narrow datapath options, which are meant
to miss timing so the search widens to the next one. The row after shows
the option that closed.

## If it does not work

Send the command, the whole output, and your tool versions. Built against
yosys 0.68, iverilog 12, OpenSTA 3.1.0, nextpnr-ice40 0.11.1. Cell counts
and fmax will move with other versions; pass and fail should not.
