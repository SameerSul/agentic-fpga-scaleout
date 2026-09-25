"""Derive the compute chiplet spec from the model.

The model is the input: given model_spec.json, the MAC datapath width comes
from the model's deployment quantization (weight_bits x activation_bits) and
the accumulator width from the longest dot-product reduction in the network.
Accumulating up to max(d_model, d_ff) full-width products needs
ceil(log2(depth)) guard bits on top of the product width so the accumulation
can never overflow, so:

    data_width = max(weight_bits, activation_bits)
    acc_width  = weight_bits + activation_bits + ceil(log2(max(d_model, d_ff)))

Both the chiplet spec and its self-checking testbench are generated here, so
a different model (a 4-bit quantized network, a wider MLP) yields different
hardware through the identical agentic flow with no code changes. The golden
model, directed tests, and throughput burst in the testbench all follow the
derived widths. Pure Python 3 stdlib."""
import json
import math
import os
import random
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))


def derive_chiplet_spec(ms):
    """model spec -> chiplet spec. The derivation is recorded in the spec so
    the signed-off profile can carry its own provenance."""
    wb, ab = ms["weight_bits"], ms["activation_bits"]
    dw = max(wb, ab)
    assert dw >= 4, "testbench stimulus assumes at least a 4-bit datapath"
    depth = max(ms["d_model"], ms["d_ff"])
    guard = math.ceil(math.log2(depth))
    aw = wb + ab + guard
    # Quantized transformer weights and activations are symmetric signed
    # two's complement, so the datapath is signed. This is not cosmetic: an
    # unsigned multiplier turns every negative weight into a large positive
    # product, which passes an unsigned testbench and computes the wrong
    # model. Signed also makes the accumulator rule conservative rather than
    # exact, since the largest signed magnitude product is 2^(wb+ab-2)
    # rather than 2^(wb+ab), leaving a bit of headroom.
    signed = bool(ms.get("signed", True))
    # Pipeline depth follows the datapath width. A signed multiply is the
    # critical path, and it grows with the operand width: at 16 bits the
    # two-stage form misses a 100 MHz clock by nearly half. Splitting the
    # multiply into two half-width partial products and registering them
    # costs one cycle of latency, not throughput, which is the right trade
    # for a unit that is fed back to back.
    stages = 2 if dw <= 8 else 3
    return {
        "name": "mac%d_%s" % (dw, ms["name"]),
        "description": "%d-bit pipelined multiply-accumulate unit derived "
                       "from %s: datapath from the model quantization, "
                       "accumulator sized for its longest reduction"
                       % (dw, ms["name"]),
        "top_module": "mac",
        "parameters": {
            "data_width": dw,
            "acc_width": aw,
            "signed": signed,
            "pipeline_stages": stages,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "weight_bits": wb,
            "activation_bits": ab,
            "reduction_depth": depth,
            "guard_bits": guard,
            "signed": signed,
            "rule": "acc_width = weight_bits + activation_bits + "
                    "ceil(log2(max(d_model, d_ff)))",
            "signedness_rule": "symmetric quantization is two's complement "
                               "signed, so operands, product and accumulator "
                               "are all signed",
            "pipeline_stages": stages,
            "pipeline_rule": "2 stages up to an 8-bit datapath, 3 above it: "
                             "the signed multiply is the critical path and "
                             "splitting it into half-width partial products "
                             "buys the clock back for one cycle of latency",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "clear", "dir": "input", "width": 1,
             "desc": "synchronous accumulator clear"},
            {"name": "a", "dir": "input", "width": dw,
             "signed": signed,
             "desc": "multiplicand (activation), %s two's complement"
                     % ("signed" if signed else "unsigned")},
            {"name": "b", "dir": "input", "width": dw,
             "signed": signed,
             "desc": "multiplier (weight), %s two's complement"
                     % ("signed" if signed else "unsigned")},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "input operands valid"},
            {"name": "acc", "dir": "output", "width": aw,
             "signed": signed,
             "desc": "accumulator value, %s two's complement"
                     % ("signed" if signed else "unsigned")},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "acc updated this cycle"},
        ],
        "behavior": [
            ("Stage 1 registers the full %d-bit product a*b and the valid "
             "flag." % (2 * dw)) if stages == 2 else
            ("Stage 1 registers two half-width partial products of a*b and "
             "the valid flag; stage 2 combines them into the full %d-bit "
             "product." % (2 * dw)),
            "The final stage adds the registered product into the %d-bit "
            "accumulator when the piped valid is set." % aw,
            "clear synchronously zeroes acc and valid_out, taking priority "
            "over accumulation.",
            "Latency from valid_in to valid_out is %d cycles." % stages,
            "%d accumulator bits = %d product bits + %d guard bits, enough "
            "for a %d-deep dot product with no overflow."
            % (aw, wb + ab, guard, depth),
            "Operands, product and accumulator are %s. The multiply must be "
            "a %s multiply and the product must be sign-extended into the "
            "accumulator." % (("signed two's complement", "signed") if signed
                              else ("unsigned", "unsigned")),
        ],
    }


TB_TEMPLATE = """`timescale 1ns/1ps
// GENERATED by specgen.py from the model spec: do not edit by hand.
// Self-checking testbench for the {dw}-bit MAC with a {aw}-bit accumulator:
// golden pipeline mirror plus directed and random stimulus, machine-parseable
// TB_FAIL / TB_RESULT lines for the agent, and a final back-to-back burst
// that measures steady-state throughput (span cycles per MAC) and pipeline
// latency, printed as a TB_PROFILE line so the flow derives cycles_per_mac
// from simulation instead of hardcoding it.
module tb_mac;
  reg clk = 0, rst_n = 0, clear = 0, valid_in = 0;
  reg  signed [{dwm}:0] a = 0, b = 0;
  wire signed [{awm}:0] acc;
  wire valid_out;
  integer checks = 0, i;
  reg  signed [{awm}:0] m_acc;
  reg  signed [{pwm}:0] m_prod;
{m_extra_decl}  reg m_vpipe, m_vout;
  reg [255:0] testname;

  // Throughput profiling state
  integer cyc = 0;
  integer first_vin_cyc = 0, first_vout_cyc = 0, last_vout_cyc = 0;
  integer vout_count = 0;
  reg profiling = 0;

  mac dut (.clk(clk), .rst_n(rst_n), .clear(clear), .a(a), .b(b),
           .valid_in(valid_in), .acc(acc), .valid_out(valid_out));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    cyc = cyc + 1;
    if (profiling && valid_out) begin
      if (vout_count == 0) first_vout_cyc = cyc;
      last_vout_cyc = cyc;
      vout_count = vout_count + 1;
    end
  end

  // Golden model: full-width SIGNED product, sign-extended into the
  // accumulator, synchronous clear with priority. Signed matters: the
  // quantized weights are two's complement, so an unsigned multiplier
  // computes a large positive product for every negative weight.
  always @(posedge clk) begin
    if (!rst_n) begin
      m_prod <= 0; m_vpipe <= 0; m_acc <= 0; m_vout <= 0;{m_extra_reset}
    end else begin
      m_prod  <= a * b;
      m_vpipe <= valid_in;
{m_extra_logic}      if (clear) begin
        m_acc <= 0; m_vout <= 0;
      end else begin
        if ({m_v}) m_acc <= m_acc + {m_p};
        m_vout <= {m_v};
      end
    end
  end

  task check;
    begin
      checks = checks + 1;
      if (acc !== m_acc || valid_out !== m_vout) begin
        $display("TB_FAIL test=%0s expected_acc=%0d got_acc=%0d expected_vout=%b got_vout=%b",
                 testname, m_acc, acc, m_vout, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  // One clock of stimulus: check state from prior edge, then drive new inputs.
  task step(input v, input signed [{dwm}:0] ai,
            input signed [{dwm}:0] bi, input c);
    begin
      @(negedge clk);
      check;
      valid_in = v; a = ai; b = bi; clear = c;
    end
  endtask

  task idle(input integer n);
    begin
      for (i = 0; i < n; i = i + 1) step(0, 0, 0, 0);
    end
  endtask

  initial begin
    testname = "reset";
    repeat (3) @(negedge clk);
    rst_n = 1;
    idle(2);

    testname = "small_values";
    step(1, {dw}'sd3, {dw}'sd5, 0);
    step(1, {dw}'sd7, {dw}'sd9, 0);
    step(1, {dw}'sd15, {dw}'sd15, 0);
    idle(4);

    // Negative operands. An unsigned multiplier reads -1 as {umax} and
    // produces a large positive product, so it cannot pass this test, and
    // an accumulator that is not sign-extended cannot either.
    testname = "signed_operands";
    step(1, -{dw}'sd1, {dw}'sd1, 0);
    step(1, {dw}'sd1, -{dw}'sd1, 0);
    step(1, -{dw}'sd1, -{dw}'sd1, 0);
    step(1, -{dw}'sd{half}, {dw}'sd3, 0);
    idle(4);

    // Extreme-magnitude operands: the full product needs {pw} bits, so a
    // truncated product register cannot pass this test. -{half} * -{half}
    // is the largest signed magnitude product.
    testname = "wide_product";
    step(1, -{dw}'sd{half}, -{dw}'sd{half}, 0);
    step(1, {dw}'sd{halfm}, {dw}'sd{halfm}, 0);
    step(1, -{dw}'sd{half}, {dw}'sd{halfm}, 0);
    idle(4);

    testname = "sync_clear";
    step(0, 0, 0, 1);
    idle(3);
    step(1, {dw}'sd{c1}, -{dw}'sd{c2}, 0);
    idle(4);

    // Accumulating past zero: the running sum has to go negative and come
    // back, which a truncated or unsigned accumulator gets wrong.
    testname = "clear_then_negative";
    step(0, 0, 0, 1);
    idle(2);
    step(1, -{dw}'sd{half}, {dw}'sd{halfm}, 0);
    step(1, -{dw}'sd{half}, {dw}'sd{halfm}, 0);
    step(1, {dw}'sd{halfm}, {dw}'sd{halfm}, 0);
    step(1, {dw}'sd{halfm}, {dw}'sd{halfm}, 0);
    idle(4);

    testname = "random";
    for (i = 0; i < 300; i = i + 1)
      step($random, $random, $random, ($random % 23) == 0);
    idle(4);

    // Throughput burst: 256 back-to-back valid MACs, every cycle still
    // checked against the golden model. span_cycles / macs is the measured
    // steady-state initiation interval (cycles per MAC).
    testname = "throughput_burst";
    step(0, 0, 0, 1);
    idle(2);
    profiling = 1;
    vout_count = 0;
    first_vin_cyc = cyc;
    for (i = 0; i < 256; i = i + 1)
      step(1, i, {dw}'d3, 0);
    idle(6);
    profiling = 0;
    $display("TB_PROFILE macs=%0d span_cycles=%0d latency_cycles=%0d",
             vout_count, last_vout_cyc - first_vout_cyc + 1,
             first_vout_cyc - first_vin_cyc);

    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def render_testbench(spec):
    dw = spec["parameters"]["data_width"]
    aw = spec["parameters"]["acc_width"]
    stages = spec["parameters"].get("pipeline_stages", 2)
    maxv = (1 << dw) - 1
    half = 1 << (dw - 1)          # magnitude of the most negative operand
    # The golden model mirrors the pipeline it is checking. A deeper DUT
    # compared against a two-stage mirror fails on latency alone, which
    # would read as a datapath bug and send the agent after the wrong thing.
    if stages >= 3:
        m_extra_decl = "  reg  signed [%d:0] m_prod2;\n  reg m_vpipe2;\n" % (
            2 * dw - 1)
        m_extra_logic = ("      m_prod2 <= m_prod;\n"
                         "      m_vpipe2 <= m_vpipe;\n")
        # Reset them too. A pipeline register left out of the reset branch
        # holds X until the first real datum reaches it, and the mirror then
        # reports X where the DUT reports 0.
        m_extra_reset = " m_prod2 <= 0; m_vpipe2 <= 0;"
        m_v, m_p = "m_vpipe2", "m_prod2"
    else:
        m_extra_decl, m_extra_logic, m_extra_reset = "", "", ""
        m_v, m_p = "m_vpipe", "m_prod"
    return TB_TEMPLATE.format(
        m_extra_decl=m_extra_decl, m_extra_logic=m_extra_logic,
        m_extra_reset=m_extra_reset, m_v=m_v, m_p=m_p,
        dw=dw, aw=aw, dwm=dw - 1, awm=aw - 1, pwm=2 * dw - 1, pw=2 * dw,
        maxv=maxv, umax=maxv, half=half, halfm=half - 1,
        v1=maxv - 1, v2=maxv - 2,
        c1=12 & (half - 1), c2=34 & (half - 1))


# Standard Ethernet rates and the MAC datapaths that carry them, ordered
# narrowest first. Every option for a rate sustains it (bytes_per_cycle * 8
# * clock >= rate), but they trade combinational depth against clock period:
# doubling the width halves the clock and roughly doubles the XOR tree, and
# because CRC trees collapse sub-linearly the wider/slower option often
# closes timing when the narrow/fast one does not. The flow tries them in
# order and keeps the first that signs off, which is the same call a human
# designer makes when the first datapath misses.
ETH_DATAPATHS = {
    1.0: ((1, 125.0), (2, 62.5)),
    10.0: ((8, 156.25), (16, 78.125)),
    25.0: ((16, 195.3125), (32, 97.65625)),
    100.0: ((64, 195.3125), (128, 97.65625)),
}
ETH_RATES = tuple(sorted(ETH_DATAPATHS))


# CRC32 next state is a linear function over GF(2), so it can be written
# either as the byte-at-a-time ripple (small, but combinational delay grows
# linearly with the datapath width) or as a flat XOR reduction per output bit
# (larger, but depth grows logarithmically). The ripple is tried first
# because it is the cheaper design; the flat form is what a wide endpoint
# needs to meet its clock.
CRC_ARCHS = ("serial", "matrix")


def crc_matrix(w, poly=0xEDB88320):
    """Return (A, B): column j of A is the next state produced by state bit j
    alone, column k of B the next state produced by data bit k alone. Because
    the step function has no constant term, the next state is exactly the XOR
    of the selected columns, which is what makes the flat form legitimate
    rather than an approximation."""
    def step(state, data):
        x = state
        for i in range(w):
            x ^= (data >> (8 * i)) & 0xFF
            for _ in range(8):
                x = (x >> 1) ^ (poly if x & 1 else 0)
        return x
    assert step(0, 0) == 0, "step has a constant term, so it is not linear"
    return ([step(1 << j, 0) for j in range(32)],
            [step(0, 1 << k) for k in range(8 * w)])


def endpoint_options(link_gbps):
    """Standard rate at or above the target, and its datapath options, each
    a (bytes_per_cycle, clock_mhz, architecture) triple. Every width is
    offered as a ripple first and as a flat XOR tree second, so the search
    only pays for the larger design once the cheaper one misses its clock."""
    rate = next((r for r in ETH_RATES if r >= link_gbps - 1e-9), ETH_RATES[-1])
    base = ETH_DATAPATHS[rate]
    return rate, tuple([(w, clk, a) for a in CRC_ARCHS for w, clk in base])


def derive_endpoint_spec(link_gbps, option=0):
    """Size the fabric endpoint to the link it has to keep up with.

    An endpoint narrower than the wire throttles it: a 4-byte datapath at
    250 MHz carries 8 Gbps no matter what transceiver sits behind it. This
    picks a standard datapath that sustains the target rate; `option`
    selects among the width/clock trades for that rate, which the flow
    advances when timing does not close."""
    rate, opts = endpoint_options(link_gbps)
    w, clk, arch = opts[min(option, len(opts) - 1)]
    return {
        "name": "crc32_endpoint_%dB" % w,
        "description": "Fabric endpoint CRC32 datapath, %d byte(s) per cycle, "
                       "%s next-state form, zlib/Ethernet compatible "
                       "(reflected polynomial 0xEDB88320), sized for a %g "
                       "Gbps link" % (w, arch, rate),
        "top_module": "crc32",
        "unit": "byte",
        "parameters": {
            "data_width": 8 * w,
            "crc_width": 32,
            "bytes_per_cycle": w,
            "target_clock_mhz": clk,
            "architecture": arch,
            "polynomial": "0xEDB88320 reflected form of 0x04C11DB7",
        },
        "derivation": {
            "target_link_gbps": link_gbps,
            "standard_rate_gbps": rate,
            "datapath_option": option,
            "options_for_rate": [list(o) for o in opts],
            "rule": "standard Ethernet MAC datapath whose bytes_per_cycle * "
                    "8 * core_clock sustains the link rate; wider and slower "
                    "options, then the flat XOR form, are tried when timing "
                    "does not close",
            "sustains_gbps": w * 8 * clk / 1000.0,
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "clear", "dir": "input", "width": 1,
             "desc": "synchronous state clear, start of a new frame"},
            {"name": "data", "dir": "input", "width": 8 * w,
             "desc": "one %d-byte word per cycle, bytes LSB-first "
                     "(little-endian packing)" % w},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "data word valid"},
            {"name": "crc_out", "dir": "output", "width": 32,
             "desc": "CRC32 of all words since clear"},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "crc_out updated this cycle"},
        ],
        "behavior": [
            "State initializes to 0xFFFFFFFF on reset and on synchronous "
            "clear.",
            "Each valid word consumes its %d bytes LSB-first; per byte b, "
            "state s becomes 8 iterations of s = (s >> 1) ^ (0xEDB88320 & "
            "-(s & 1)) starting from s ^ b." % w,
            "crc_out registers the updated state XOR 0xFFFFFFFF (the "
            "standard final inversion), matching zlib.crc32 of the byte "
            "stream.",
            "Latency from valid_in to valid_out is 1 cycle; throughput is "
            "one %d-byte word per cycle." % w,
        ],
    }


def crc32_bytes(data):
    """Reference CRC32 (zlib compatible), used to bake golden vectors into
    the generated testbench so the checks are independent of the RTL."""
    return zlib.crc32(data) & 0xFFFFFFFF


def _word_literal(chunk, w):
    """Pack w bytes LSB-first into a Verilog sized literal."""
    v = int.from_bytes(chunk, "little")
    return "%d'h%0*x" % (8 * w, 2 * w, v)


def render_crc_testbench(spec):
    """Generate the endpoint testbench at the derived width, with golden
    CRC values computed by zlib rather than by the design under test."""
    w = spec["parameters"]["bytes_per_cycle"]
    rng = random.Random(42)
    ascii_src = b"123456789abcdefghijklmnopqrstuvwxyz"
    cases = [
        ("zero_word", bytes(w)),
        # Exactly w bytes: a short literal would leave the high bytes of the
        # word zero while the golden value covered only the literal, so the
        # vector has to be padded to the full datapath width.
        ("ascii_word", (ascii_src * (w // len(ascii_src) + 1))[:w]),
        ("two_words", bytes(rng.randrange(256) for _ in range(2 * w))),
        ("eight_words", bytes(rng.randrange(256) for _ in range(8 * w))),
    ]
    zero_lit = _word_literal(bytes(w), w)
    zero_crc = crc32_bytes(bytes(w))
    body = []
    # Reset alone has to initialise the CRC state. Every other test opens
    # with start_frame, whose clear reinitialises the same register, so
    # without this section a design with a dead reset branch passes.
    body.append('    testname = "reset_init";')
    body.append("    expect_quiet(testname);")
    body.append("    word(%s);" % zero_lit)
    body.append("    expect_crc(32'h%08x);" % zero_crc)
    body.append("")
    # Outputs must not move until the rising edge. At frame granularity a
    # block clocked on the wrong edge is only a half cycle shift and is
    # invisible, so the settling point is checked explicitly.
    body.append('    testname = "edge_discipline";')
    body.append("    start_frame;")
    body.append("    @(negedge clk); data = %s; valid_in = 1;" % zero_lit)
    body.append("    #1;")
    body.append("    expect_quiet(testname);")
    body.append("    @(posedge clk); #1;")
    body.append("    checks = checks + 1;")
    body.append("    if (crc_out !== 32'h%08x || valid_out !== 1'b1) begin"
                % zero_crc)
    body.append('      $display("TB_FAIL test=%0s expected_crc=%0d '
                'got_crc=%0d expected_vout=1 got_vout=%b",')
    body.append("               testname, 32'h%08x, crc_out, valid_out);"
                % zero_crc)
    body.append('      $display("TB_RESULT: FAIL");')
    body.append("      $finish;")
    body.append("    end")
    body.append("    @(negedge clk); valid_in = 0;")
    body.append("")
    for name, payload in cases:
        body.append('    testname = "%s";' % name)
        body.append("    start_frame;")
        for off in range(0, len(payload), w):
            body.append("    word(%s);"
                        % _word_literal(payload[off:off + w], w))
        body.append("    expect_crc(32'h%08x);" % crc32_bytes(payload))
        body.append("")
    # Re-check the first vector after a clear, proving state reinitializes.
    body.append('    testname = "clear_reinit";')
    body.append("    start_frame;")
    for off in range(0, len(cases[0][1]), w):
        body.append("    word(%s);" % _word_literal(cases[0][1][off:off + w], w))
    body.append("    expect_crc(32'h%08x);" % crc32_bytes(cases[0][1]))
    return CRC_TB_TEMPLATE.format(w=w, dm=8 * w - 1, cases="\n".join(body))


CRC_TB_TEMPLATE = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the {w}-byte-per-cycle CRC32 fabric endpoint.
// Golden values are computed by Python's zlib.crc32, so a design that
// matches them is wire-compatible with the Ethernet FCS it has to produce.
// Prints machine-parseable TB_FAIL / TB_PROFILE / TB_RESULT lines.
module tb_crc;
  reg clk = 0, rst_n = 0, clear = 0, valid_in = 0;
  reg [{dm}:0] data = 0;
  wire [31:0] crc_out;
  wire valid_out;
  integer checks = 0, i;
  reg [255:0] testname;

  integer cyc = 0;
  integer first_vin_cyc = 0, first_vout_cyc = 0, last_vout_cyc = 0;
  integer vout_count = 0;
  reg profiling = 0;

  crc32 dut (.clk(clk), .rst_n(rst_n), .clear(clear), .data(data),
             .valid_in(valid_in), .crc_out(crc_out), .valid_out(valid_out));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    cyc = cyc + 1;
    if (profiling && valid_out) begin
      if (vout_count == 0) first_vout_cyc = cyc;
      last_vout_cyc = cyc;
      vout_count = vout_count + 1;
    end
  end

  task start_frame;
    begin
      @(negedge clk); clear = 1; valid_in = 0;
      @(negedge clk); clear = 0;
    end
  endtask

  task word(input [{dm}:0] d);
    begin
      @(negedge clk); data = d; valid_in = 1;
    end
  endtask

  // valid_out must be low. Used after reset and between frames, and as the
  // settling check that makes wrong-edge clocking observable.
  task expect_quiet(input [127:0] why);
    begin
      checks = checks + 1;
      if (valid_out !== 1'b0) begin
        $display("TB_FAIL test=%0s expected_vout=0 got_vout=%b",
                 why, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  task expect_crc(input [31:0] want);
    begin
      @(negedge clk); valid_in = 0;
      checks = checks + 1;
      if (crc_out !== want) begin
        $display("TB_FAIL test=%0s expected_crc=%0d got_crc=%0d",
                 testname, want, crc_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);

{cases}

    // Throughput burst: 256 back-to-back words, measuring the steady-state
    // initiation interval in cycles per byte.
    testname = "throughput_burst";
    start_frame;
    profiling = 1;
    vout_count = 0;
    first_vin_cyc = cyc;
    for (i = 0; i < 256; i = i + 1) word(i);
    @(negedge clk); valid_in = 0;
    repeat (4) @(negedge clk);
    profiling = 0;
    $display("TB_PROFILE bytes=%0d span_cycles=%0d latency_cycles=%0d",
             vout_count * {w}, last_vout_cyc - first_vout_cyc + 1,
             first_vout_cyc - first_vin_cyc);

    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_endpoint(link_gbps, spec_file="spec_crc.json",
                      tb_file="tb_crc.v", option=0):
    """Write the derived endpoint spec and its testbench, return the spec."""
    spec = derive_endpoint_spec(link_gbps, option)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_crc_testbench(spec))
    return spec


def load_model_spec(path=None):
    return json.load(open(path or os.path.join(ROOT, "model_spec.json")))


def generate(ms=None, spec_file="spec_mac.json", tb_file="tb_mac.v"):
    """Write the derived spec and testbench next to the flow inputs and
    return the spec. Called by the flow before every chiplet run, so the
    hardware always tracks the current model spec."""
    ms = ms or load_model_spec()
    spec = derive_chiplet_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_testbench(spec))
    return spec


if __name__ == "__main__":
    s = generate()
    print("derived %s from %s: %d x %d bit MAC, %d bit accumulator "
          "(%d guard bits for a %d-deep reduction)"
          % (s["name"], s["derivation"]["model"],
             s["parameters"]["data_width"], s["parameters"]["data_width"],
             s["parameters"]["acc_width"], s["derivation"]["guard_bits"],
             s["derivation"]["reduction_depth"]))
