# Board handoff: fpgAI on a Zynq board

For whoever has one of the boards and an x86 machine: the Zybo Z7-20 or
the ZC706, the two with a generated package. Everything up to the board
has been generated and simulated in this repo; this is the part that
needs the hardware. Nothing below needs this repo's author.

| | Zybo Z7-20 | ZC706 |
|---|---|---|
| package | `board_zybo/` | `board_zc706/` (`python3 board_zybo.py --board zc706`) |
| part | XC7Z020-1CLG400 | XC7Z045-2FFG900 |
| core width | 16 lanes | 32 lanes, twice the weights a core cycle |
| Vivado board preset | Digilent's board files | ships with Vivado |
| Vivado license | free edition | the ZC706 kit's device-locked license (the XC7Z045 is not in the free edition), or the open flow below |
| open flow, post-route | core 73.4, bus 65.4 MHz | core 62.7, bus 53.9 MHz (Qwen3: 69.5, 51.5) |
| boot from JTAG | JP5 to JTAG | SW11 boot mode to JTAG (UG954) |
| USB-UART | the PROG/UART micro-USB | J21 |
| busy LED | LD0 | none driven (STATUS register instead) |

Below, `board_<name>/` is whichever package matches the board.

## What is already verified, and what is not

| step | state |
|---|---|
| Qwen2.5-0.5B on the generated blocks' integer arithmetic | 14/16 teacher-forced against float |
| the generated sequencer, 24 layers + head, in iverilog | chooses " Paris", the integer model's token |
| the Zybo top level (`board_zybo/rtl/fpgai_zybo.v`) driven only through its AXI-Lite registers, DDR model with random stalls | all 24 layers + head: " Paris", same core cycles, 1.19 bus cycles per core cycle |
| the ZC706 top level at 32 lanes, the same way | one layer of each model: the direct run's token and core cycles, 1.70 bus cycles per core cycle |
| each package's own ARM program (`sw/main.c`, unchanged) on its own RTL in Verilator, its own SD card files, every port stalling (`cosim.py`) | Zybo and ZC706: "The capital of France is Paris. Paris is the capital of France. Paris is the capital of France."; ZC706 Qwen3: "The capital of France is Paris. The capital of the United States is Washington, D.C. The capital"; all 16 tokens of each, every logit the integer model's |
| the same, with no SD card: the files written into DDR as `sw/load_jtag.tcl` writes them over JTAG (`cosim.py --jtag`) | Zybo, ZC706 and ZC706 Qwen3: the same 16 tokens as from the card |
| the weight streamer under 1500 jumps and a stalling bus, every word checked | passes; it failed on every seed before a slot was limited to one burst in flight |
| fits the part (Yosys, nextpnr-xilinx) | Zybo: 28% of LUT sites, 132 of 220 DSPs, 49 BRAMs; ZC706 at 32 lanes: 9%, 196 of 900 DSPs |
| places and routes in the open flow (nextpnr-xilinx), XC7Z020 and XC7Z045 | all three builds close 50 MHz on both clocks, and every bitstream round-trips frame for frame |
| several boards on their own clocks, each a stage of layers (`gals.py`, simulated) | two and three stages over fabric UARTs give one board's tokens and logits exactly |
| the stage ARM program's UDP protocol, run on a host (two processes, lwIP shimmed, one datagram dropped) | resends, reassembles, prints the reference tokens |
| Vivado block design, Vivado timing, the ARM program | **not run** |
| the board | **not run** |

## What you need

- x86 Linux or Windows with Vivado and Vitis 2020.2 or later, licensed
  for the board's part (table above).
- For the Zybo, Digilent's board files, so Vivado knows its DDR and MIO:
  <https://github.com/Digilent/vivado-boards>, copied into
  `<Vivado>/data/boards/board_files/`. The ZC706's preset ships with
  Vivado.
- The board and its USB cable. A microSD card (1 GB or more, FAT32) is
  optional: without one the ARM program takes the same files over JTAG
  (step 4), slower to load but nothing else changes.
- Python 3 (standard library only), `iverilog` only if you want to rerun
  the simulations.

## 1. Make the model files

The RTL in `board_zybo/rtl/` is committed; the model files are not
(500 MB). They land in the package's `sd/` folder on this machine, and
reach the board either over JTAG or on an SD card. Make them from the same commit, since the ARM program checks
that the bitstream's layout registers match its header before it runs.

```bash
python3 fetch_qwen.py                      # Qwen2.5-0.5B, ~1 GB, into qwen_weights/
python3 qwen_full.py --no-sim --work build_qfull   # quantize, write weights.bin, cparams.hex, gains.hex
python3 board_zybo.py                      # board_zybo/, including board_zybo/sd/
```

For the ZC706, whose core is 32 lanes wide, build at its width instead
(`board_zybo.py` refuses a build of the wrong width):

```bash
python3 qwen_full.py --no-sim --lanes 32 --work build_q25l32
python3 board_zybo.py --board zc706 --work build_q25l32   # board_zc706/, with sd/
```

`board_zybo/sd/` then holds `weights8.bin` (494 MB), `cparams.bin`,
`vocab.bin` and `prompt.bin`. `git status` should show no change under
`board_zybo/rtl/`; if it does, the generator and the committed RTL
disagree, and the committed one is what was simulated.

### Qwen3-0.6B on the ZC706

`board_zc706_qwen3/` is the same package for Qwen3-0.6B, the model
Architect Labs hosted: 596 MB of int8 weights, which the ZC706's 1 GB
holds with room to spare. Its open-flow build routes at 50 MHz as well
(69.5 MHz core, 51.5 MHz bus, post-route), and its bitstream round-trips.
Its SD files come from:

```bash
python3 fetch_qwen.py --model qwen3                              # ~1.5 GB
FPGAI_QWEN=qwen3 python3 qwen_full.py --no-sim --lanes 32 --work build_q3l32
FPGAI_QWEN=qwen3 python3 board_zybo.py --work build_q3l32 --board zc706 --out board_zc706_qwen3
```

Everything below is the same, from inside `board_zc706_qwen3/`.
The UART should then print Qwen3's integer greedy continuation, 16
tokens, whose first, " Paris", the RTL chose in simulation through all
28 layers and the head:

```
The capital of France is Paris. The capital of the United States is Washington, D.C. The capital
```

### Before the board: run the package on this host

```bash
python3 cosim.py board_zybo --jitter
```

This runs the package's own `sw/main.c`, unchanged, against the
package's own RTL (Verilator) and the SD card files just written, every
register access an AXI-Lite transaction and every DDR port stalling at
random. It prints what the UART should print, in under ten minutes; if
it does not, the fault is in the files or the RTL, not the board.

### A faster Zybo: 32 lanes

`board_zybo_32/` is the same Zybo at the ZC706's width: the ZC706 package's
RTL, about a third faster a token. It places and routes on the XC7Z020 in
the open flow (57.8 MHz core, 58.5 MHz bus), with 24 of the attention
head's score multipliers in LUTs; Vivado may keep them all on DSPs. Make
its SD card from the 32-lane build and run it the same way:

```bash
python3 qwen_full.py --lanes 32 --work build_q25l32
python3 board_zybo.py --board zybo_z7_20_32 --work build_q25l32
python3 cosim.py board_zybo_32 --jitter
```

## 2. Build the bitstream and the platform

```bash
cd board_zybo          # or board_zc706
vivado -mode batch -source build.tcl
```

It writes `fpgai.xsa`, `timing.rpt` and `utilization.rpt`. Read the
timing report first:

- WNS must be positive. If it is not, lower
  `CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ` in `build.tcl` (40, then 25) and
  rebuild; the design is fully synchronous, so a slower clock only
  slows it down.
- The core's clock is `aclk` through a BUFGCE, gated while a memory
  port waits. Check that Vivado times the core's registers against
  `aclk` and does not report the gated clock as unconstrained.

If a step of `build.tcl` fails, it is most likely the block-design
automation (`apply_bd_automation`), which has not met a real Vivado.
Doing that step by hand in the GUI is fine: the PS7 with the board
preset, M_AXI_GP0 to `fpgai/s_axi`, `fpgai/m_axi_w0..3` to S_AXI_HP0..3,
`fpgai/m_axi_kv` to S_AXI_ACP, everything on FCLK_CLK0, the registers at
0x43C00000.

### If Vivado's block design will not build

`board_zybo/open/` builds the same PL without Vivado's IP integrator:
`build_open.sh` (setup steps in its header) runs Yosys, nextpnr-xilinx
with `nextpnr-xilinx-dsp.patch`, and Project X-Ray, and writes
`fpgai.bit`. That bitstream has no PS7 configuration in it: the ARM
still needs the Zybo's ps7_init (DDR, clocks, and the PL level shifters
enabled), which Vitis generates from any Zybo Z7-20 Vivado project, even
an empty one with just the PS7 and the board preset. Load `fpgai.bit`
after ps7_init runs, then continue at step 3.

## 3. Build the ARM program

In Vitis: a platform from `fpgai.xsa` (standalone, ps7_cortexa9_0), with
the `xilffs` library enabled in the BSP; then an empty C application
with `sw/main.c` and `sw/fpgai_layout.h`. The default linker script is
fine: the program sits below 0x08000000, where the weights start.

## 4. Run it

1. Set the boot mode to JTAG (JP5 on the Zybo, SW11 on the ZC706),
   connect USB, power on, and open the USB-UART at 115200 8N1.
2. From Vitis, program the FPGA and run the application.
3. Without an SD card, the program prints `no SD card: waiting for the
   files over JTAG`. In Vitis's XSCT console (Window > XSCT Console),
   with forward slashes on Windows:

   ```
   cd <path to the repo>/board_zybo
   source sw/load_jtag.tcl
   ```

   It halts the core, writes `sd/weights8.bin`, `cparams.bin`,
   `vocab.bin` and `prompt.bin` into DDR at the addresses in
   `fpgai_layout.h`, then their sizes and a marker, and resumes the
   core; the program checks the sizes against the bitstream's and goes
   on. JTAG takes minutes for the weights.

   With an SD card instead: copy `board_zybo/sd/*` to its root before
   step 1; the program finds the card and loads from it (the weights
   take about a minute).

Expected on the UART: the files load, then the prompt and its
continuation:

```
The capital of France is Paris. Paris is the capital of France. Paris is the capital of France.
```

That is the integer model's greedy continuation, 16 tokens, which the
RTL is built to match exactly. `cosim.py` has run this very program on
this package's RTL and printed exactly that line, every token's logit the
integer model's, so a board that prints anything else is a finding worth
reporting. At 50 MHz each
position should take about 0.5 s and the head step about 0.7 s (on the
ZC706, about 0.4 s a position); the
last line prints the core and bus cycles of the last step.

## What to send back

- `timing.rpt`, `utilization.rpt`
- the UART log, start to end

## If it goes wrong

| symptom | meaning |
|---|---|
| `the bitstream's layout is not this program's` | SD files, header and bitstream are from different commits: redo steps 1 to 3 from one commit |
| `missing weights8.bin` or a size mismatch | the SD card is not FAT32, or the files are not at its root |
| waits at `waiting for the files over JTAG` | `sw/load_jtag.tcl` has not run, or XSCT is on another target: `targets` in the console should list `ARM Cortex-A9 MPCore #0` |
| `weights8.bin is ... bytes ... this bitstream wants` after JTAG | the `sd/` files are from another build: redo step 1 from this commit |
| hangs after loading, LD0 lit | the core is waiting on memory: an AXI connection in the block design is wrong (check the address map: every master must see DDR at 0x00000000) |
| hangs, LD0 dark | the start never reached the core: check the GP0 connection and the register base 0x43C00000 |
| different tokens | send the log: the position where it first differs says which block |
| wrong from the first token, with an SD card made from an older commit | an old `weights8.bin` holds each word's lanes in reverse (RESULTS.md, "the weight image's byte order"), and the layout check cannot see it, since the sizes are the same. Remake step 1 |

## Another model, or another mix of boards

`spec2rtl.py` rebuilds and re-verifies everything above for any
checkpoint of these shapes and any list of boards, and says what it
checked in `report.md`:

```bash
python3 spec2rtl.py --weights qwen_weights --board zybo_z7_20 --package --bridge
FPGAI_QWEN=qwen3 python3 spec2rtl.py --weights qwen_weights/qwen3-0.6b --boards zc706 zybo_z7_20 --package
```

The second writes one package a board, as `cluster.py --package` does
below, after simulating the two-board pipeline against the one-board
model.

## Several boards

Any set of the boards can share one model, each running its own clock
and a contiguous range of layers, passing the hidden state on as
messages. Plan the split, then write one package per board:

```bash
python3 cluster.py zc706 zybo_z7_20                   # which layers where, and each link
FPGAI_QWEN=qwen3 python3 cluster.py zc706 zybo_z7_20 --package build_cluster
```

`build_cluster/stage<i>_<board>/` is each board's package, built and
run exactly as above, with two differences:

- its ARM program talks UDP, so create the Vitis application from the
  **lwIP Echo Server** template (it brings the lwIP BSP and
  `platform_zynq.c`), then replace the template's `main.c` with the
  package's `sw/main.c` and add `sw/fpgai_layout.h`; keep `xilffs`
  enabled;
- the boards need to be on one Ethernet segment: a switch, or a direct
  cable for two. Stage i is 192.168.1.(10+i), UDP port 5000, set in each
  package's `fpgai_layout.h`.

Start the later stages first; stage 0 holds the prompt, prints the text
on its UART and waits for each token to come back from the last stage.
A board with no ARM would take the fabric UART link instead
(`gals.py`'s `stage_ctrl`), but none of those boards can hold a layer
yet: they have no DRAM this design reaches.

### Splitting the weights instead

Every board on every layer, each with a slice of every matrix, so a
token reads each board's share of the weights at once:

```bash
FPGAI_QWEN=qwen3 python3 spec2rtl.py --weights qwen_weights/qwen3-0.6b --boards zc706 zybo_z7_20 --split weights --package
```

`rank<i>_<board>/` is each board's package, its share of every layer
sized by its speed (`report.md` lists them). It builds the same way as a
stage's, lwIP Echo Server template included; rank i is 192.168.1.(10+i),
UDP port 5000. Every rank's SD card holds its own `weights8.bin` and
`cparams.bin` and the prompt; rank 0's also `vocab.bin`, and rank 0
prints the text. Start them in any order: a rank that misses a slice asks
for it again every 250 ms, so one started late only costs the others
that wait. Each prints how many gathers it served and how many times it
asked again.

Before the boards, the same packages run on this host, each rank's own
program on its own RTL, talking UDP on localhost:

```bash
python3 cosim.py <out>/rank0_zc706 <out>/rank1_zybo_z7_20 --jitter
```

Rank 0 prints the text the boards should.
