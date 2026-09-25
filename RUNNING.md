# Running this yourself

What to install, what to type, and what you should see. If you see
something else, that is a bug and worth reporting, not something to work
around.

## Install

Everything below is stdlib Python 3 plus open EDA tools. No Python
packages, no network, no accounts.

```
brew install icarus-verilog yosys            # simulation and synthesis
brew install nextpnr-ice40 icestorm          # place and route, optional
```

OpenSTA is optional too. Without it the flow falls back to a gate-depth
proxy and says so in the output; the proxy is an estimate, not timing
closure, and the sweep refuses to count it.

Check what you have:

```
for t in iverilog vvp yosys sta nextpnr-ice40 icepack; do
  printf "%-16s %s\n" "$t" "$(command -v $t || echo MISSING)"
done
```

iverilog and vvp are required. Everything else degrades gracefully.

## The one command that checks everything

```
python3 tests.py
```

**Expect:** a line per check and `all 197 tests passed` at the end.
Takes a few minutes. Any `FAIL` line is a real failure; the suite stops
at the first one.

If nextpnr is not installed you will see one `SKIP` line for the
bitstream check instead of five passes, and the total will be lower.
That is expected, not a failure.

## What the flow actually does

```
python3 chiplet_flow.py
```

**Expect:** a table per block showing an agent proposing Verilog and the
tools judging it, something like

```
iter  fixes applied                    sim    synth  timing  fpga
1     []                               fail   -      -       -
2     ['widen_product_register']       fail   -      -       -
3     ['widen_product_register',       pass   pass   pass    pass
       'implement_sync_clear']
```

The first iterations fail on purpose. The agent is handed parsed tool
output, not a hint, and the fixes it applies are derived from what the
simulator said. A run that converges on iteration 1 means the seeded
bugs did not trip, which is itself worth looking at.

It writes `chiplet_profile.json` and `fabric_profile.json`: the measured
numbers everything downstream consumes.

## Every block across every spec

```
python3 sweep.py
```

**Expect:** `63 cases, 63 clean, 0 failed`, and a table where every
column reads `ok`. Takes roughly twenty minutes, most of it mutation
testing.

Each row is one generated block at one model spec, driven through
derivation, RTL, simulation, synthesis, timing closure, FPGA mapping,
profile sanity and a mutation sweep of its own testbench. The `dv`
column reading `5/5 killed` means every injected defect was caught. A
`SURVIVED` entry names a defect the flow would have signed off on.

Some rows say `did not converge in 5 iterations` in the detail and are
still counted clean: those are the endpoint's narrower datapath options,
which are meant to fail timing so the search widens. The line after them
shows the option that closed.

## Running the model

```
python3 generate.py
```

**Expect:**

```
prompt:    'the agent'
generated: ' writes the rtl and the '
float ref: ' writes the rtl and the '
teacher-forced next-token agreement: 48 of 48 (100%)

  MAC:       12/12 dot products bit exact
  requant:   12/12 requantizations bit exact
  exp:       12/12 exponentials bit exact
  recip:     12/12 reciprocals bit exact
  rsqrt:     12/12 inverse square roots bit exact
```

A trained character-level transformer decoded through bit-accurate
models of the generated RTL, with a sample of that decode's arithmetic
replayed through the actual Verilog in iverilog. The two text lines
matching is the point: the int8 hardware path reproduces the float model
exactly on this checkpoint.

`train_tiny.py` retrains the checkpoint from scratch and takes about
fifteen minutes. You do not need to: `tiny_llm.json` is committed.

## A real bitstream

```
python3 bitstream.py --block mac
```

**Expect:** place and route, then

```
   post-route fmax 166.14 MHz against a 100 MHz target (MET)
   build_bitstream/top.bin  135100 bytes
   the packed bitstream passes the testbench (613 checks)
```

That last line is the one that matters. `icepack` writes the bits a
device would be loaded with, `icebox_vlog` turns those bits back into
logic, and the original testbench runs against them. Also works for
`--block softmax`, `wmem`, `matvec`, `requant`, `exp`, `recip`, `rsqrt`,
`crc`.

The device is a Lattice iCE40 HX8K. **Nobody on this team owns one.** It
is the only family with an open place and route flow that installs on an
ARM Mac; Vivado does not run there and nextpnr has no mainline Xilinx
target. So this proves the flow reaches a working bitstream, on a
different part from the ones we have.

## If something fails

Report the command, the full output, and `python3 -c "import
platform;print(platform.platform())"`. A failure here is information:
every number in RESULTS.md came from these commands on one machine, and
a second machine disagreeing is worth knowing about.

Known-soft spots, in order of likelihood:

- **Tool versions.** Built against yosys 0.68, iverilog 12, OpenSTA
  3.1.0, nextpnr-ice40 0.11.1. Cell counts and fmax will shift with
  other versions; pass/fail should not.
- **`sta` missing.** Timing silently becomes a proxy. The sweep catches
  this and fails the case rather than reporting an estimate as closure.
- **Long sweeps.** Mutation testing runs a full simulation per mutant
  per case. If it looks hung, it is probably the equivalence prover,
  which is capped at ninety seconds and reports `unproven` rather than
  guessing.
