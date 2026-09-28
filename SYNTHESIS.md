# Synthesising this on your board

For Yax. This is what the repo produces, how to build it, and what you
should see. Everything here has been run; if your machine disagrees, that
is worth knowing and worth reporting.

## Fastest path: a model talking over the Basys 3's USB port

`board_basys3/` is a complete Vivado build for the Basys 3: the
Qwen-shaped decoder and every block under it, a top level on the board's
USB-UART, the weights as a block RAM image, the pin constraints and a
build script. On an x86 machine with Vivado:

```
cd board_basys3
vivado -mode batch -source build.tcl
```

Program the bitstream it prints, open a serial terminal on the board's
COM port at 115200 8N1, type `the agent` and press enter. It should send
back ` writes the rtl ` and a newline, which is what the simulation of
this exact top level sends (`python3 board.py` regenerates the directory
and reruns it). LED 0 is on while it decodes and LED 15 blinks. Yosys puts
it at 8632 LUTs, 34 DSPs and 12 block RAMs of the XC7A35T, and the core
runs at 50 MHz from the MMCM; `timing.rpt` is the number to send back,
since Vivado's is the first real one.

## What this is in one paragraph

You give it a model spec, a small JSON file naming a transformer's
dimensions and its quantization. It derives the hardware that model
needs, writes the Verilog, and checks it with real tools until it passes.
Twenty-one blocks come out, each one sized from the model rather than written
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

## The twenty-one blocks

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
   attn       one attention head for a decode step: q.k over the KV
              cache, softmax, then the weighted sum of V
   rmsnorm    RMSNorm over a d_model row: sum of squares, one rsqrt,
              a scaled product per element
   silu       x * sigmoid(x), streaming, from exp and recip
   gmlp       Qwen's gated MLP: gate through SiLU, times up, then down
   resadd     residual add of two scaled int8 tensors, streaming
   rope       rotary position embedding on q and k, one pair a cycle
   projn      32-lane projection over a 256-bit weight word, sized to
              the board's DDR bandwidth
   attnn      32-lane attention head: positions, then dimensions
   proj       a full-size projection, 896 by 4864 here, over external
              activation and weight memories
```

And the one that runs the model:

```
   decoder    one decode step of the trained 16-dimensional checkpoint,
              embedding to argmax, over one instance each of proj,
              rmsnorm, attn and resadd. The text it emits is its own.
   qwen_decoder  the same for a Qwen-shaped checkpoint: RoPE, two query
              heads over one KV head, SwiGLU, two layers, final norm
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
   attn.v     needs  matvec.v mac.v requant.v softmax.v expu.v recip.v
                     exp_rom.v recip_rom.v
   rmsnorm.v  needs  rsqrt.v rsqrt_rom.v requant.v
   silu.v     needs  expu.v recip.v exp_rom.v recip_rom.v
   gmlp.v     needs  matvec.v mac.v requant.v silu.v expu.v recip.v
                     exp_rom.v recip_rom.v
   proj.v     needs  matvec.v mac.v requant.v
   rope.v     needs  rope_rom.v, its frequency and sine tables
   projn.v    needs  mac.v requant.v
   decoder.v  needs  proj.v rmsnorm.v attn.v resadd.v and everything
                     they need, all derived at the checkpoint's size
```

The testbenches are in the repo root as `tb_*.v` if you want to run them
in Vivado's simulator rather than iverilog.

## What it should synthesise to

Measured here with yosys against a generic library, for Qwen2.5-0.5B:

```
   block      fmax      cells     note
   wmem       333 MHz   61712     huge only because the generic library
                                  has no BRAM; on your part it is 1 BRAM
   recip      206 MHz    1636
   matvec     187 MHz    1294
   rsqrt      170 MHz    1340
   exp        164 MHz    2370
   mac        169 MHz    1473     infers one DSP48
   mlp        119 MHz   23124
   requant    115 MHz   10445
   rmsnorm    115 MHz   19904
   silu       120 MHz   11935
   gmlp       110 MHz   48418
   resadd     136 MHz    4894
   rope       129 MHz   21224     1874 LUTs, most of them the sine
                                  table; on your part that is a BRAM
   proj       115 MHz   13989     1831 LUTs and 9 DSPs; its memories
                                  are outside it, on your part BRAM or DDR
   projn      115 MHz   67136     4516 LUTs and 40 DSPs, 32 lanes
   attnn      112 MHz  246432     6953 LUTs and 74 DSPs, 32 lanes
   attn       115 MHz  132047     the score and weight buffers and the
                                  softmax's become flops here; on the
                                  FPGA they map to 3 BRAMs, 3265 LUTs
   softmax    121 MHz   40262     its exponential buffer, the same way

   Counts include each block's submodules. Before this, a composite
   block reported whichever submodule yosys listed first: the MLP and
   RMSNorm both showed the requantizer's 10396.
```

The decoder is sized for the checkpoint rather than Qwen: 111 MHz,
14117 LUTs, 22 DSPs and 2 BRAMs on the UltraScale+ mapping. On the
7-series mapping, the family of Vaibhav's Basys 3, it is 13247 LUTs and
22 DSPs, 64% and 24% of an XC7A35T, so the whole decoder fits that board.
The full-size sequencer for the real Qwen2.5-0.5B (`qwen_full.py`),
every block at 8-bit weights and 16-bit activations with the weights and
KV cache on external ports, is 25650 LUTs plus 3240 as LUT RAM, 148 DSPs
and 16 block RAMs on 7-series: it fits your Zynq 7020 at 54% of LUTs and
67% of DSPs, with the model's weights in the Zybo's DDR.

The Qwen-shaped decoder, two layers with RoPE and SwiGLU, is 8376 LUTs
plus 164 as LUT RAM, 34 DSPs and 3 block RAMs on 7-series: it fits the
Basys 3 too, at about 41% of its LUTs and 38% of its DSPs.

A Zynq 7020 has 53200 LUTs and 220 DSPs. The nineteen Qwen-sized blocks map to
32119 LUTs, LUT RAM included, and 195 DSPs on yosys's UltraScale+
mapping, about 60% and 89% of the part, and that counts the composite blocks' sub-blocks twice.
These are about half what this file reported before: the resource count
read every submodule table yosys printed after the top module's, then the
hierarchy totals on top, so every composite block was counted about
twice. It now reads the hierarchy totals once.
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

`--agent llm` or `--agent swarm` puts a real LLM in the writing seat.
Measured, it has now signed off nine of the ten layer blocks, eight with Haiku and
the requantizer with Sonnet. Build with the rules agent anyway: its
output is what the numbers below were measured on, and it is
reproducible and needs no network.

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
