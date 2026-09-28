# Board handoff: fpgAI on the Zybo Z7-20

For whoever has the team's Zybo Z7-20 and an x86 machine. Everything up
to the board has been generated and simulated in this repo; this is the
part that needs the hardware. Nothing below needs this repo's author.

## What is already verified, and what is not

| step | state |
|---|---|
| Qwen2.5-0.5B on the generated blocks' integer arithmetic | 14/16 teacher-forced against float |
| the generated sequencer, 24 layers + head, in iverilog | chooses " Paris", the integer model's token |
| the Zybo top level (`board_zybo/rtl/fpgai_zybo.v`) driven only through its AXI-Lite registers, DDR model with random stalls | all 24 layers + head: " Paris", same core cycles, 1.18 bus cycles per core cycle |
| fits the XC7Z020 (Yosys) | 33169 LUTs + 7380 LUT RAM, 148 DSPs, 36 BRAMs |
| places and routes on the XC7Z020 in the open flow (nextpnr-xilinx) | closes 50 MHz: core 53.4, bus 50.9 MHz; bitstream round-trips |
| Vivado block design, Vivado timing, the ARM program | **not run** |
| the board | **not run** |

## What you need

- x86 Linux or Windows with Vivado and Vitis 2020.2 or later. The free
  edition covers the XC7Z020.
- Digilent's board files, so Vivado knows the Zybo's DDR and MIO:
  <https://github.com/Digilent/vivado-boards>, copied into
  `<Vivado>/data/boards/board_files/`.
- The Zybo Z7-20, a micro-USB cable, a microSD card of 1 GB or more
  formatted FAT32.
- Python 3 (standard library only), `iverilog` only if you want to rerun
  the simulations.

## 1. Make the SD card files

The RTL in `board_zybo/rtl/` is committed; the model files are not
(500 MB). Make them from the same commit, since the ARM program checks
that the bitstream's layout registers match its header before it runs.

```bash
python3 fetch_qwen.py                      # Qwen2.5-0.5B, ~1 GB, into qwen_weights/
python3 qwen_full.py --no-sim --work build_qfull   # quantize, write weights.bin, cparams.hex, gains.hex
python3 board_zybo.py                      # board_zybo/, including board_zybo/sd/
```

`board_zybo/sd/` then holds `weights8.bin` (494 MB), `cparams.bin`,
`vocab.bin` and `prompt.bin`. `git status` should show no change under
`board_zybo/rtl/`; if it does, the generator and the committed RTL
disagree, and the committed one is what was simulated.

## 2. Build the bitstream and the platform

```bash
cd board_zybo
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

1. Copy `board_zybo/sd/*` to the root of the SD card and insert it.
2. Set the boot jumper (JP5) to JTAG, connect USB, power on.
3. From Vitis, program the FPGA and run the application.
4. Open the USB-UART at 115200 8N1.

Expected on the UART: the four files load (the weights take about a
minute), then the prompt and its continuation:

```
The capital of France is Paris. Paris is the capital of France. Paris is the capital of France.
```

That is the integer model's greedy continuation, 16 tokens, which the
RTL is built to match exactly. What simulation has checked: the first
generated token, " Paris", through all 24 layers and the head, and
several generated tokens in a row on small test models. So a board that
prints " Paris" and then differs later is a finding worth reporting. At 50 MHz each
position should take about 0.5 s and the head step about 0.7 s; the
last line prints the core and bus cycles of the last step.

## What to send back

- `timing.rpt`, `utilization.rpt`
- the UART log, start to end

## If it goes wrong

| symptom | meaning |
|---|---|
| `the bitstream's layout is not this program's` | SD files, header and bitstream are from different commits: redo steps 1 to 3 from one commit |
| `missing weights8.bin` or a size mismatch | the SD card is not FAT32, or the files are not at its root |
| hangs after loading, LD0 lit | the core is waiting on memory: an AXI connection in the block design is wrong (check the address map: every master must see DDR at 0x00000000) |
| hangs, LD0 dark | the start never reached the core: check the GP0 connection and the register base 0x43C00000 |
| different tokens | send the log: the position where it first differs says which block |
