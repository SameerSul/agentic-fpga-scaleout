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


# Widest add that closes in one stage in the generic cell library, which is
# what the flow gates on. Measured: a 46-bit add leaves +0.26 ns at 100 MHz,
# a 64-bit add is 3.4 ns over, and every wide add violated, so it is a
# property of the library (no carry chain) rather than of one arrangement.
#
# Lowering it to 40 was tried, so that 46-bit adds split too. The generic
# library liked it (102 -> 109 MHz) and the real device did not (97.9 ->
# 93.5 MHz on an iCE40 HX8K). More pipeline stages mean more registers and
# more routing pressure, and on a real fabric this block is routing bound
# rather than logic bound. The two disagree in direction, so the cheaper
# structure wins.
WIDE_ADD_BITS = 48
# Widest accumulator that can go into a partial product whole. Above this
# the multiplicand itself is sliced, because a multiply by a small slice is
# still a sum of shifted copies of the full multiplicand.
ACC_SLICE_BITS = 32


def requant_terms(aw, splits):
    """Partial products: the scale slices times the accumulator slices."""
    return splits * (2 if aw > ACC_SLICE_BITS else 1)


def requant_stages(aw, mw, splits):
    """Pipeline depth of the generated requantizer.

    Lives here so the spec and the renderer cannot drift: the testbench
    waits for the depth the spec declares, and a disagreement shows up as a
    datapath failure rather than as the latency mismatch it is.
    """
    per_add = 2 if (aw + mw) > WIDE_ADD_BITS else 1
    levels, n = 0, requant_terms(aw, splits)
    while n > 1:
        n = (n + 1) // 2
        levels += 1
    # products, the reduction tree, the rounding add, the shift, the saturate
    return 1 + levels * per_add + per_add + 1 + 1


def exp_lut(lut_bits, out_frac):
    """2**f for f in [0,1), the table the exponential unit indexes.

    Stored scaled by 2**out_frac, so entries run from 1.0 to just under
    2.0 in that fixed point. Derived here rather than written out, so the
    table and the golden model cannot drift apart.
    """
    return [int(round((2.0 ** (i / float(1 << lut_bits)))
                      * (1 << out_frac)))
            for i in range(1 << lut_bits)]


LOG2E_Q16 = int(round(1.4426950408889634 * (1 << 16)))


def exp_golden(x, p):
    """Exact fixed-point model of the exponential unit.

    exp(x) = 2**(x*log2(e)), and a power of two splits into a shift and a
    table lookup: t = n + f with n the integer part and f in [0,1), so
    2**t is 2**f shifted right by -n. x is required to be non-positive,
    which is what softmax guarantees after subtracting the row maximum,
    and that is why only right shifts appear.
    """
    fi, fo, lb = p["in_frac"], p["out_frac"], p["lut_bits"]
    t = (x * LOG2E_Q16) >> 16          # x*log2(e), still Q.in_frac
    n = t >> fi                        # floor, negative or zero
    f = t - (n << fi)                  # fractional part, [0, 1)
    idx = f >> (fi - lb)
    m = exp_lut(lb, fo)[idx]
    sh = -n
    if sh > fo:                        # underflows the output format
        return 0
    return m >> sh


def derive_exp_spec(ms):
    """model spec -> exponential unit spec.

    Softmax is the last piece of the attention datapath still running on
    the host, and the exponential is the part of it that actually needs
    hardware: the sum and the reciprocal are an accumulator and a divide,
    but exp is transcendental. Everything else in the block is a shift.
    """
    ab = ms["activation_bits"]
    # The table is indexed by the whole fractional part, so no bits of it
    # are thrown away: at 6 bits the truncation cost 1% absolute error,
    # which is the table and not the arithmetic.
    fi, fo = 8, 15
    lb = fi
    # The input is a score minus the row maximum, so it is non-positive
    # and anything below about -(out_frac+1)/log2(e) underflows the output
    # format entirely. The useful range is therefore fixed by the output
    # format, not by the model's activation width: tying it to 2*ab gave a
    # 32-bit operand for a 16-bit model, which missed timing for range
    # that can never be used.
    span = int(math.ceil((fo + 2) / 1.4426950408889634))
    iw = 1 + max(4, span.bit_length()) + fi
    return {
        "name": "exp%d_%s" % (iw, ms["name"]),
        "description": "Fixed-point exponential for the attention softmax: "
                       "Q%d.%d signed input constrained to be non-positive, "
                       "Q0.%d unsigned output, by a %d-entry table of 2**f "
                       "and an arithmetic shift"
                       % (iw - 1 - fi, fi, fo, 1 << lb),
        "top_module": "expu",
        "unit": "score",
        "parameters": {
            "in_width": iw, "in_frac": fi,
            "out_width": fo + 1, "out_frac": fo,
            "lut_bits": lb,
            "signed": True,
            "pipeline_stages": 3,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "activation_bits": ab,
            "rule": "exp(x) = 2**(x*log2(e)); the integer part of "
                    "x*log2(e) is an arithmetic right shift and the "
                    "fractional part indexes a %d-entry table of 2**f"
                    % (1 << lb),
            "log2e_q16": LOG2E_Q16,
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "x", "dir": "input", "width": iw, "signed": True,
             "desc": "score minus the row maximum, Q%d.%d, non-positive"
                     % (iw - 1 - fi, fi)},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "x valid"},
            {"name": "y", "dir": "output", "width": fo + 1,
             "desc": "exp(x) in Q0.%d, 1.0 represented as %d"
                     % (fo, 1 << fo)},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y updated this cycle"},
        ],
        "behavior": [
            "Stage 1 registers x*log2(e) as a Q.%d value, using a %d-bit "
            "fixed-point constant." % (fi, 17),
            "Stage 2 splits it into an integer part and a fractional "
            "part and registers the table entry the fraction selects.",
            "Stage 3 shifts that entry right by the magnitude of the "
            "integer part and registers the result.",
            "An input small enough that the shift exceeds %d output "
            "fraction bits flushes to zero rather than wrapping." % fo,
            "x is required to be non-positive, which softmax guarantees "
            "by subtracting the row maximum first.",
            "Latency from valid_in to valid_out is 3 cycles.",
        ],
    }


def render_exp_testbench(spec):
    """Golden outputs come from exp_golden here, and the accuracy of the
    scheme itself is checked against math.exp separately, so a design that
    reproduces its own approximation cannot pass on that alone."""
    p = spec["parameters"]
    iw, fi, fo = p["in_width"], p["in_frac"], p["out_frac"]
    rnd = random.Random(23)
    # Everything is clamped to what the port can represent. The input
    # width is derived from the useful exponent range, so a vector past
    # the underflow point is not a harder test, it is an unrepresentable
    # one, and it fails on truncation rather than on anything real.
    lo = -(1 << (iw - 1))
    def rep(v):
        return max(lo, min(0, v))
    xs = [rep(v) for v in
          [0, -1, -(1 << fi), -(2 << fi), -(3 << fi), -(1 << (fi - 1)),
           -((1 << fi) - 1), -(fo << fi), -((fo + 1) << fi),
           -((fo + 4) << fi), lo]]
    xs += [-rnd.randrange(0, -lo) for _ in range(140)]
    # Inputs where the product sits one count below a shift boundary, so a
    # single LSB of error in it carries into the shifted exponent. Without
    # these the testbench cannot see an off-by-one in the multiply at all,
    # which mutation testing demonstrated.
    edge = [x for x in range(-1, lo, -1)
            if ((x * LOG2E_Q16) & 0xFFFF) in (0xFFFF, 0, 1)]
    xs += edge[:40]
    body = []
    for x in xs:
        body.append("    drive(%s, %d'd%d);"
                    % (_slit(x, iw), p["out_width"], exp_golden(x, p)))
    return EXP_TB.format(iwm=iw - 1, owm=p["out_width"] - 1,
                         settle=p["pipeline_stages"] - 1,
                         cases="\n".join(body), n=len(xs))


EXP_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the fixed-point exponential. Golden values
// come from the Python fixed-point model, and tests.py separately checks
// that model against math.exp, so a design that merely reproduces its own
// approximation error cannot pass.
module tb_expu;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg  signed [{iwm}:0] x = 0;
  wire        [{owm}:0] y;
  wire valid_out;
  integer checks = 0, i;
  reg [255:0] testname;

  expu dut (.clk(clk), .rst_n(rst_n), .x(x), .valid_in(valid_in),
            .y(y), .valid_out(valid_out));
  always #5 clk = ~clk;

  task expect_quiet;
    begin
      checks = checks + 1;
      if (valid_out !== 1'b0 || y !== 0) begin
        $display("TB_FAIL test=reset_init expected_y=0 got_y=%0d vout=%b",
                 y, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  task drive(input signed [{iwm}:0] xi, input [{owm}:0] want);
    begin
      @(negedge clk); x = xi; valid_in = 1;
      @(negedge clk); valid_in = 0;
      repeat ({settle}) @(negedge clk);
      checks = checks + 1;
      if (y !== want || valid_out !== 1'b1) begin
        $display("TB_FAIL test=%0s x=%0d expected_y=%0d got_y=%0d vout=%b",
                 testname, xi, want, y, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "reset_init";
    repeat (3) @(negedge clk);
    expect_quiet;
    rst_n = 1;
    @(negedge clk);
    expect_quiet;
    testname = "exp";
{cases}

    $display("TB_PROFILE scores=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, {n} * (2 + {settle}), 1 + {settle});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_exp(ms=None, spec_file="spec_exp.json", tb_file="tb_expu.v"):
    """Write the derived exponential spec and testbench, return the spec."""
    ms = ms or load_model_spec()
    spec = derive_exp_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_exp_testbench(spec))
    return spec


def recip_lut(lut_bits, out_width):
    """1/m for m in [1,2), scaled by 2**out_width.

    Entries run from 2**out_width down to just above half of it. The
    table is derived rather than written out so it cannot drift from the
    golden model.
    """
    n = 1 << lut_bits
    return [min((1 << out_width) - 1,
                int(round((1 << out_width) / (1.0 + i / float(n)))))
            for i in range(n)]


def recip_golden(x, p):
    """Exact fixed-point model of the reciprocal unit.

    Returns (mantissa, shift) with 1/x == mantissa >> (shift + bias), the
    bias being out_width + in_width - 1. Returning a mantissa and a shift
    rather than a pre-shifted number is the whole point: the softmax
    denominator spans ten bits of range, so a single fixed-point output
    would hold as few as five significant bits at the top of that range
    and carry 6% error. The mantissa is always full width, and the
    consumer folds the shift into the multiply it was going to do anyway.
    """
    iw, ow, lb = p["in_width"], p["out_width"], p["lut_bits"]
    if x <= 0:
        return (1 << ow) - 1, 0
    k = iw - x.bit_length()            # leading zeros within in_width
    xn = x << k                        # msb now at bit iw-1
    idx = (xn >> (iw - 1 - lb)) & ((1 << lb) - 1)
    return recip_lut(lb, ow)[idx], k


def recip_apply(num, m, k, p):
    """What a consumer does with (mantissa, shift): num / x, to the
    precision the unit provides."""
    bias = p["out_width"] + p["in_width"] - 1
    return (num * m) >> (bias - k)


def derive_recip_spec(ms):
    """model spec -> reciprocal spec.

    The other half of softmax. Once the exponentials are summed, every
    weight is that exponential divided by the sum, and a divider per
    weight is absurd: the reciprocal is computed once per row and
    multiplied in. This is the reciprocal.
    """
    e = derive_exp_spec(ms)
    ef = e["parameters"]["out_frac"]
    seq = ms.get("seq_len", 1024)
    # The denominator is a sum of exponentials, each at most 1.0, so it is
    # bounded by the number of terms. Attention sums over the context, but
    # the hardware only needs the width, not the count.
    terms = min(seq, 4096)
    iw = (terms << ef).bit_length()
    ow, lb = 17, 8
    return {
        "name": "recip%d_%s" % (iw, ms["name"]),
        "description": "Reciprocal for the attention softmax denominator: "
                       "%d-bit unsigned input, returns a %d-bit mantissa "
                       "and the shift that goes with it, by normalising "
                       "to [1,2) and a %d-entry table"
                       % (iw, ow, 1 << lb),
        "top_module": "recip",
        "unit": "row",
        "parameters": {
            "in_width": iw, "out_width": ow, "lut_bits": lb,
            "shift_bias": ow + iw - 1, "signed": False,
            "pipeline_stages": 3, "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "exp_out_frac": ef,
            "max_terms": terms,
            "rule": "in_width bounds a sum of at most %d exponentials "
                    "each below 2**%d; the output is a mantissa and a "
                    "shift, so a softmax weight is one multiply and one "
                    "shift" % (terms, ef),
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "x", "dir": "input", "width": iw,
             "desc": "softmax denominator, unsigned, non-zero"},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "x valid"},
            {"name": "y", "dir": "output", "width": ow,
             "desc": "reciprocal mantissa, always full width"},
            {"name": "k", "dir": "output", "width": max(4, iw.bit_length()),
             "desc": "normalisation shift; 1/x is y >> (%d - k)"
                     % (ow + iw - 1)},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y updated this cycle"},
        ],
        "behavior": [
            "Stage 1 counts the leading zeros of x and registers both the "
            "count and x shifted left by it, so the value sits in [1,2).",
            "Stage 2 registers the table entry the normalised mantissa "
            "selects and the shift the count implies.",
            "Stage 3 registers the table entry and the count. The "
            "consumer applies the shift, which keeps the mantissa at "
            "full precision instead of truncating it here.",
            "x is required to be non-zero, which the softmax denominator "
            "guarantees because it always contains exp(0) = 1.",
            "Latency from valid_in to valid_out is 3 cycles.",
        ],
    }


def render_recip_testbench(spec):
    p = spec["parameters"]
    iw = p["in_width"]
    rnd = random.Random(31)
    xs = [1, 2, 3, (1 << 15), (1 << 15) + 1, (1 << (iw - 1)),
          (1 << iw) - 1, (1 << iw) - 2, (3 << (iw - 2))]
    xs += [rnd.randrange(1, 1 << iw) for _ in range(120)]
    # Powers of two and their neighbours, where the normalisation count
    # changes and an off-by-one in it is visible.
    for b in range(1, iw):
        xs += [(1 << b) - 1, (1 << b), (1 << b) + 1]
    xs = [x for x in xs if 1 <= x < (1 << iw)]
    kw = max(4, iw.bit_length())
    body = "\n".join("    drive(%d'd%d, %d'd%d, %d'd%d);"
                      % ((iw, x, p["out_width"]) + recip_golden(x, p)[:1]
                         + (kw, recip_golden(x, p)[1]))
                      for x in xs)
    return RECIP_TB.format(iwm=iw - 1, owm=p["out_width"] - 1,
                           kwm=kw - 1, settle=p["pipeline_stages"] - 1,
                           cases=body, n=len(xs))


RECIP_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the softmax reciprocal. Golden values come
// from the Python fixed-point model; tests.py separately checks that
// model against true division.
module tb_recip;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg  [{iwm}:0] x = 0;
  wire [{owm}:0] y;
  wire [{kwm}:0] k;
  wire valid_out;
  integer checks = 0;
  reg [255:0] testname;

  recip dut (.clk(clk), .rst_n(rst_n), .x(x), .valid_in(valid_in),
             .y(y), .k(k), .valid_out(valid_out));
  always #5 clk = ~clk;

  task expect_quiet;
    begin
      checks = checks + 1;
      if (valid_out !== 1'b0 || y !== 0 || k !== 0) begin
        $display("TB_FAIL test=reset_init expected_y=0 got_y=%0d vout=%b",
                 y, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  task drive(input [{iwm}:0] xi, input [{owm}:0] want,
             input [{kwm}:0] wantk);
    begin
      @(negedge clk); x = xi; valid_in = 1;
      @(negedge clk); valid_in = 0;
      repeat ({settle}) @(negedge clk);
      checks = checks + 1;
      if (y !== want || k !== wantk || valid_out !== 1'b1) begin
        $display("TB_FAIL test=%0s x=%0d expected_y=%0d got_y=%0d expected_k=%0d got_k=%0d vout=%b",
                 testname, xi, want, y, wantk, k, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "reset_init";
    repeat (3) @(negedge clk);
    expect_quiet;
    rst_n = 1;
    @(negedge clk);
    expect_quiet;
    testname = "recip";
{cases}

    $display("TB_PROFILE rows=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, {n} * (2 + {settle}), 1 + {settle});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_recip(ms=None, spec_file="spec_recip.json",
                   tb_file="tb_recip.v"):
    """Write the derived reciprocal spec and testbench, return the spec."""
    ms = ms or load_model_spec()
    spec = derive_recip_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_recip_testbench(spec))
    return spec


def rsqrt_lut(lut_bits, out_width):
    """Indexed by the top lut_bits of the normalised input.

    A square root halves the exponent, so the input is normalised by an
    even number of bits and its mantissa spans two octaves rather than
    one: the index runs over [2**(lut_bits-2), 2**lut_bits), standing for
    a value in [1,4). Indexing by those bits directly is what lets the
    hardware skip a divide; an earlier formulation mapped the mantissa
    linearly onto the table and needed a division by three to do it.
    """
    n, lo = 1 << lut_bits, 1 << (lut_bits - 2)
    out = []
    for i in range(n):
        if i < lo:
            out.append((1 << out_width) - 1)      # only reachable at x = 0
        else:
            out.append(min((1 << out_width) - 1,
                           int(round((1 << out_width)
                                     / math.sqrt(i / float(lo))))))
    return out


def rsqrt_golden(x, p):
    """Exact fixed-point model of the inverse square root unit.

    Returns (mantissa, e) with 1/sqrt(x) == mantissa >> (out_width + e).
    x is a mean of squares, so it is non-negative; zero returns the
    saturated mantissa, which is what a normalisation epsilon is for.
    """
    iw, ow, lb = p["in_width"], p["out_width"], p["lut_bits"]
    if x <= 0:
        return (1 << ow) - 1, 0
    e = (x.bit_length() - 1) // 2      # halved exponent
    s = iw - 2 - 2 * e                 # even-aligned left shift, >= 0
    xn = x << s
    idx = (xn >> (iw - lb)) & ((1 << lb) - 1)
    return rsqrt_lut(lb, ow)[idx], e


def rsqrt_apply(num, m, e, p):
    """What a consumer does with (mantissa, e): num / sqrt(x)."""
    return (num * m) >> (p["out_width"] + e)


def derive_rsqrt_spec(ms):
    """model spec -> inverse square root spec.

    RMSNorm, which is what the target model family normalises with,
    divides an activation by the root mean square of its row. The sum of
    squares is the MAC unit and the mean is a shift; this is the only
    part that needs its own hardware.
    """
    ab = ms["activation_bits"]
    depth = max(ms["d_model"], ms["d_ff"])
    # A sum of depth squares of ab-bit values, before the mean shift.
    # Rounded up to an even width: the normalisation has to move the
    # value by an even number of bits so the halved exponent is an
    # integer, and at an odd width that shift goes negative for the
    # largest inputs. One spare bit is cheaper than handling it.
    iw = 2 * ab + math.ceil(math.log2(depth))
    iw += iw & 1
    ow, lb = 17, 8
    return {
        "name": "rsqrt%d_%s" % (iw, ms["name"]),
        "description": "Inverse square root for RMSNorm: %d-bit unsigned "
                       "mean of squares, returns a %d-bit mantissa and a "
                       "halved exponent, by normalising to [1,4) and a "
                       "%d-entry table" % (iw, ow, 1 << lb),
        "top_module": "rsqrt",
        "unit": "row",
        "parameters": {
            "in_width": iw, "out_width": ow, "lut_bits": lb,
            "signed": False, "pipeline_stages": 3,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "activation_bits": ab,
            "reduction_depth": depth,
            "rule": "in_width holds a sum of %d squares of %d-bit values; "
                    "the output is a mantissa and a halved exponent, so a "
                    "normalised activation is one multiply and one shift"
                    % (depth, ab),
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "x", "dir": "input", "width": iw,
             "desc": "mean of squares, unsigned"},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "x valid"},
            {"name": "y", "dir": "output", "width": ow,
             "desc": "1/sqrt mantissa, always full width"},
            {"name": "e", "dir": "output", "width": max(4, iw.bit_length()),
             "desc": "halved exponent; 1/sqrt(x) is y >> (%d + e)" % ow},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y updated this cycle"},
        ],
        "behavior": [
            "Stage 1 finds the position of the highest set bit and halves "
            "it, registering both the even normalisation and the shifted "
            "value, which then lies in [1,4).",
            "Stage 2 registers the table entry that mantissa selects.",
            "Stage 3 registers the entry and the halved exponent. The "
            "consumer applies the shift, keeping the mantissa full width.",
            "A zero input returns the saturated mantissa; RMSNorm adds an "
            "epsilon before this unit so that case does not arise.",
            "Latency from valid_in to valid_out is 3 cycles.",
        ],
    }


def render_rsqrt_testbench(spec):
    p = spec["parameters"]
    iw = p["in_width"]
    rnd = random.Random(37)
    xs = [1, 2, 3, 4, 5, (1 << 10), (1 << 10) + 7, (1 << (iw - 1)),
          (1 << iw) - 1]
    xs += [rnd.randrange(1, 1 << iw) for _ in range(120)]
    for b in range(1, iw):
        xs += [(1 << b) - 1, (1 << b), (1 << b) + 1]
    xs = [x for x in xs if 1 <= x < (1 << iw)]
    ew = max(4, iw.bit_length())
    body = "\n".join(
        "    drive(%d'd%d, %d'd%d, %d'd%d);"
        % ((iw, x, p["out_width"]) + (rsqrt_golden(x, p)[0],)
           + (ew, rsqrt_golden(x, p)[1]))
        for x in xs)
    return RSQRT_TB.format(iwm=iw - 1, owm=p["out_width"] - 1, ewm=ew - 1,
                           settle=p["pipeline_stages"] - 1,
                           cases=body, n=len(xs))


RSQRT_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the RMSNorm inverse square root. Golden
// values come from the Python fixed-point model; tests.py separately
// checks that model against true 1/sqrt.
module tb_rsqrt;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg  [{iwm}:0] x = 0;
  wire [{owm}:0] y;
  wire [{ewm}:0] e;
  wire valid_out;
  integer checks = 0;
  reg [255:0] testname;

  rsqrt dut (.clk(clk), .rst_n(rst_n), .x(x), .valid_in(valid_in),
             .y(y), .e(e), .valid_out(valid_out));
  always #5 clk = ~clk;

  task expect_idle;
    begin
      checks = checks + 1;
      if (valid_out !== 1'b0 || y !== 0 || e !== 0) begin
        $display("TB_FAIL test=reset_init expected_y=0 got_y=%0d vout=%b",
                 y, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  task drive(input [{iwm}:0] xi, input [{owm}:0] want,
             input [{ewm}:0] wante);
    begin
      @(negedge clk); x = xi; valid_in = 1;
      @(negedge clk); valid_in = 0;
      repeat ({settle}) @(negedge clk);
      checks = checks + 1;
      if (y !== want || e !== wante || valid_out !== 1'b1) begin
        $display("TB_FAIL test=%0s x=%0d expected_y=%0d got_y=%0d expected_e=%0d got_e=%0d vout=%b",
                 testname, xi, want, y, wante, e, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "reset_init";
    repeat (3) @(negedge clk);
    expect_idle;
    rst_n = 1;
    @(negedge clk);
    expect_idle;
    testname = "rsqrt";
{cases}

    $display("TB_PROFILE rows=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, {n} * (2 + {settle}), 1 + {settle});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_rsqrt(ms=None, spec_file="spec_rsqrt.json",
                   tb_file="tb_rsqrt.v"):
    """Write the derived inverse square root spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_rsqrt_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_rsqrt_testbench(spec))
    return spec


def derive_matvec_spec(ms):
    """model spec -> weight-streaming sequencer spec.

    Every block so far computes. This one sequences: it walks a weight
    matrix column by column out of memory, feeds the MAC unit, waits out
    that unit's pipeline, and signals when each column's accumulator
    holds a finished dot product. It is the first generated block whose
    correctness depends on another generated block's latency, which is
    what makes a set of verified units into something that runs.
    """
    c = derive_chiplet_spec(ms)
    dw = c["parameters"]["data_width"]
    aw = c["parameters"]["acc_width"]
    stages = c["parameters"]["pipeline_stages"]
    depth = max(ms["d_model"], ms["d_ff"])
    dep_w = max(4, depth.bit_length())
    col_w = dep_w
    addr_w = dep_w * 2
    return {
        "name": "matvec_%s" % ms["name"],
        "description": "Weight-streaming sequencer for the MAC chiplet: "
                       "walks a matrix column by column, drives the MAC, "
                       "drains its %d-stage pipeline and flags each "
                       "finished column" % stages,
        "top_module": "matvec",
        "unit": "column",
        "parameters": {
            "data_width": dw, "acc_width": aw,
            "depth_width": dep_w, "col_width": col_w, "addr_width": addr_w,
            "mac_stages": stages, "max_depth": depth,
            # Block RAM registers its read. Assuming an asynchronous one
            # is assuming a LUT RAM big enough to hold a weight matrix,
            # which does not exist on these parts.
            "mem_latency": 1,
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "reduction_depth": depth,
            "mac_pipeline_stages": stages,
            "rule": "address widths from the longest reduction; the drain "
                    "count is the MAC's own pipeline depth, so a change "
                    "there changes this block",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "begin a matrix-vector product"},
            {"name": "depth", "dir": "input", "width": dep_w,
             "desc": "reduction length, elements per column"},
            {"name": "cols", "dir": "input", "width": col_w,
             "desc": "number of output columns"},
            {"name": "a_addr", "dir": "output", "width": dep_w,
             "desc": "activation index being read"},
            {"name": "w_addr", "dir": "output", "width": addr_w,
             "desc": "weight index, column major: col*depth + row"},
            {"name": "mac_valid", "dir": "output", "width": 1,
             "desc": "drive the MAC this cycle"},
            {"name": "mac_clear", "dir": "output", "width": 1,
             "desc": "clear the MAC accumulator before the next column"},
            {"name": "col_valid", "dir": "output", "width": 1,
             "desc": "the MAC accumulator holds a finished column"},
            {"name": "col_index", "dir": "output", "width": col_w,
             "desc": "which column col_valid refers to"},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a product is in progress"},
        ],
        "behavior": [
            "start begins a product of cols columns, each a reduction of "
            "depth elements.",
            "For each column the sequencer issues depth consecutive "
            "addresses with mac_valid high, a_addr walking 0..depth-1 and "
            "w_addr walking col*depth..col*depth+depth-1.",
            "mac_valid follows the address by one cycle, because the "
            "memory registers its read. Driving it in the same cycle as "
            "the address multiplies whatever the memory held before.",
            "It then holds mac_valid low for %d cycles, the read latency "
            "plus the MAC's pipeline depth, before asserting col_valid "
            "for one cycle with col_index set." % (stages + 1),
            "mac_clear is asserted after col_valid so the next column "
            "starts from zero. Without it every column accumulates into "
            "the one before it.",
            "busy is high from start until the last column is flagged.",
            "Memory reads are synchronous: data arrives one cycle after "
            "the address, as block RAM provides.",
        ],
    }


def render_matvec_testbench(spec):
    """Integration testbench: the sequencer driving the real MAC.

    The matrix is large enough that the derived address width is
    actually used. A small one leaves the top half of w_addr always
    zero, and mutation testing showed a halved address register
    surviving every vector because of it. Memory is filled by a formula
    rather than by literals so the file stays readable at this size.
    """
    p = spec["parameters"]
    dw, aw = p["data_width"], p["acc_width"]
    dep_w, col_w, addr_w = (p["depth_width"], p["col_width"],
                            p["addr_width"])
    # The matrix has to be big enough that the top half of w_addr is
    # used, or a halved address register is invisible. That threshold
    # comes from the derived width, so the matrix size has to follow it
    # rather than being a fixed number that happens to suit one spec.
    depth = 68
    cols = (1 << (addr_w // 2)) // depth + 2
    # Test data spans the whole datapath. Capping it at a byte leaves
    # the upper half of a 16-bit word always zero, and a mutation that
    # halves the storage width is then invisible: mutation testing found
    # exactly that on the 16-bit memory.
    m = (1 << dw) - 5
    half = m // 2

    def act(i):
        return (i * 104729 + 7) % m - half

    def wt(i):
        return (i * 7919 + 13) % m - half

    golden = [sum(act(r) * wt(c * depth + r) for r in range(depth))
              for c in range(cols)]
    init = "\n".join("    expect_col[%d] = %s;" % (i, _slit(v, aw))
                      for i, v in enumerate(golden))
    return MATVEC_TB.format(
        dwm=dw - 1, awm=aw - 1, depwm=dep_w - 1, colwm=col_w - 1,
        addrwm=addr_w - 1, depth=depth, cols=cols, init=init,
        nmem=depth * cols, m=m, half=half)


MATVEC_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Integration testbench: the weight-streaming sequencer driving the real
// generated MAC. Golden dot products are computed in Python, so this
// checks the pair against the arithmetic the model needs rather than
// against either block's own idea of itself.
module tb_matvec;
  reg clk = 0, rst_n = 0, start = 0;
  reg  [{depwm}:0] depth = 0;
  reg  [{colwm}:0] cols = 0;
  wire [{depwm}:0] a_addr;
  wire [{addrwm}:0] w_addr;
  wire mac_valid, mac_clear, col_valid, busy;
  wire [{colwm}:0] col_index;

  reg signed [{dwm}:0] amem [0:{depth}-1];
  reg signed [{dwm}:0] wmem [0:{nmem}-1];
  reg signed [{awm}:0] expect_col [0:{cols}-1];

  // Block RAM: the read is registered, so data lands a cycle after the
  // address. An asynchronous model here would hide an off-by-one in the
  // sequencer's valid, which is the bug this block is most prone to.
  reg signed [{dwm}:0] a_data, w_data;
  always @(posedge clk) begin
    a_data <= amem[a_addr];
    w_data <= wmem[w_addr];
  end

  wire signed [{awm}:0] acc;
  wire mac_vout;
  integer checks = 0, seen = 0, i;
  reg [255:0] testname;

  matvec seq (.clk(clk), .rst_n(rst_n), .start(start), .depth(depth),
              .cols(cols), .a_addr(a_addr), .w_addr(w_addr),
              .mac_valid(mac_valid), .mac_clear(mac_clear),
              .col_valid(col_valid), .col_index(col_index), .busy(busy));

  mac dut (.clk(clk), .rst_n(rst_n), .clear(mac_clear),
           .a(a_data), .b(w_data), .valid_in(mac_valid),
           .acc(acc), .valid_out(mac_vout));

  always #5 clk = ~clk;

  // Every flagged column is checked the moment it is flagged.
  always @(posedge clk) begin
    if (rst_n && col_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (acc !== expect_col[col_index]) begin
        $display("TB_FAIL test=%0s col=%0d expected_acc=%0d got_acc=%0d",
                 testname, col_index, expect_col[col_index], acc);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  end

  initial begin
    // Filled by formula, matching the Python golden exactly.
    for (i = 0; i < {depth}; i = i + 1)
      amem[i] = (i * 104729 + 7) % {m} - {half};
    for (i = 0; i < {nmem}; i = i + 1)
      wmem[i] = (i * 7919 + 13) % {m} - {half};
{init}
    testname = "matvec";
    repeat (3) @(negedge clk);
    rst_n = 1;
    depth = {depth};
    cols = {cols};
    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;
    // Generous bound: depth+drain per column, plus slack.
    for (i = 0; i < {cols} * ({depth} + 10) + 60; i = i + 1)
      @(negedge clk);
    checks = checks + 1;
    if (seen !== {cols}) begin
      $display("TB_FAIL test=%0s expected_acc=%0d got_acc=%0d",
               "column_count", {cols}, seen);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    checks = checks + 1;
    if (busy !== 1'b0) begin
      $display("TB_FAIL test=%0s expected_acc=0 got_acc=1",
               "busy_deasserts");
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE columns=%0d span_cycles=%0d latency_cycles=%0d",
             {cols}, {cols} * {depth}, {depth});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_matvec(ms=None, spec_file="spec_matvec.json",
                    tb_file="tb_matvec.v"):
    """Write the derived sequencer spec and its integration testbench."""
    ms = ms or load_model_spec()
    spec = derive_matvec_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_matvec_testbench(spec))
    return spec


def derive_wmem_spec(ms):
    """model spec -> weight memory and loader spec.

    The sequencer issues addresses and expects a registered read, but
    nothing generated the memory behind them or the logic that fills it.
    This does both: a tile of weight storage with a streaming write port,
    which is how weights actually reach a device, and a registered read
    port with the timing the sequencer was built against.
    """
    c = derive_chiplet_spec(ms)
    dw = c["parameters"]["data_width"]
    cap = 1024                      # one tile, not a whole matrix
    aw = (cap - 1).bit_length()
    return {
        "name": "wmem%d_%s" % (cap, ms["name"]),
        "description": "Weight tile memory with a streaming loader: %d "
                       "entries of %d bits, sequential write port, "
                       "registered read port" % (cap, dw),
        "top_module": "wmem",
        "unit": "tile",
        "parameters": {
            "data_width": dw, "capacity": cap, "addr_width": aw,
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "rule": "one tile of %d weights at the model's datapath "
                    "width; the read port is registered because that is "
                    "what the sequencer was built against and what block "
                    "RAM provides" % cap,
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "load_start", "dir": "input", "width": 1,
             "desc": "reset the write pointer and begin a load"},
            {"name": "load_valid", "dir": "input", "width": 1,
             "desc": "write load_data at the current pointer"},
            {"name": "load_data", "dir": "input", "width": dw,
             "signed": True, "desc": "weight being written"},
            {"name": "load_count", "dir": "output", "width": aw + 1,
             "desc": "how many weights have been written"},
            {"name": "rd_addr", "dir": "input", "width": aw,
             "desc": "read address"},
            {"name": "rd_data", "dir": "output", "width": dw,
             "signed": True, "desc": "registered read data, one cycle "
                                     "after rd_addr"},
        ],
        "behavior": [
            "load_start clears the write pointer.",
            "Each cycle load_valid is high, load_data is written at the "
            "pointer and the pointer advances, saturating at capacity.",
            "load_count reports the pointer, so a loader outside can "
            "tell when the tile is full.",
            "rd_data is registered: it presents mem[rd_addr] one cycle "
            "after rd_addr. A combinational read would deliver data a "
            "cycle early and the sequencer would multiply the wrong "
            "element.",
            "A write and a read of the same address in one cycle return "
            "the old contents, which is read-first behaviour.",
        ],
    }


def render_wmem_testbench(spec):
    """System testbench: the loader and memory feeding the sequencer and
    the MAC.

    This is the whole subsystem rather than one block. Weights are
    streamed in through the load port exactly as they would reach a
    device, then the sequencer walks them and the MAC reduces them, and
    the column results are checked against Python. A block checked only
    on its own would not catch a read port that is a cycle out of step
    with the thing reading it.
    """
    p = spec["parameters"]
    ms = load_model_spec()
    c = derive_chiplet_spec(ms)
    mv = derive_matvec_spec(ms)
    dw, aw = p["data_width"], p["addr_width"]
    acc_w = c["parameters"]["acc_width"]
    depth, cols = 64, 16            # 1024 weights: exactly one tile
    # Test data spans the whole datapath. Capping it at a byte leaves
    # the upper half of a 16-bit word always zero, and a mutation that
    # halves the storage width is then invisible: mutation testing found
    # exactly that on the 16-bit memory.
    m = (1 << dw) - 5
    half = m // 2

    def act(i):
        return (i * 104729 + 7) % m - half

    def wt(i):
        return (i * 7919 + 13) % m - half

    golden = [sum(act(r) * wt(c_ * depth + r) for r in range(depth))
              for c_ in range(cols)]
    exp = "\n".join("    expect_col[%d] = %s;" % (i, _slit(v, acc_w))
                     for i, v in enumerate(golden))
    return WMEM_TB.format(
        dwm=dw - 1, awm=aw - 1, accwm=acc_w - 1,
        depwm=mv["parameters"]["depth_width"] - 1,
        colwm=mv["parameters"]["col_width"] - 1,
        mvaddrwm=mv["parameters"]["addr_width"] - 1,
        cntwm=aw, depth=depth, cols=cols, nmem=depth * cols,
        m=m, half=half, exp=exp, cap=p["capacity"])


WMEM_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// System testbench: the weight loader and memory feeding the sequencer
// and the MAC. Weights are streamed in through the load port the way
// they would reach a device, then walked and reduced, and the column
// results are checked against values computed in Python.
module tb_wmem;
  reg clk = 0, rst_n = 0;
  reg load_start = 0, load_valid = 0;
  reg signed [{dwm}:0] load_data = 0;
  wire [{cntwm}:0] load_count;
  wire signed [{dwm}:0] rd_data;

  reg start = 0;
  reg  [{depwm}:0] depth = 0;
  reg  [{colwm}:0] cols = 0;
  wire [{depwm}:0] a_addr;
  wire [{mvaddrwm}:0] w_addr;
  wire mac_valid, mac_clear, col_valid, busy;
  wire [{colwm}:0] col_index;
  wire signed [{accwm}:0] acc;
  wire mac_vout;

  reg signed [{dwm}:0] amem [0:{depth}-1];
  reg signed [{accwm}:0] expect_col [0:{cols}-1];
  reg signed [{dwm}:0] a_data;
  integer checks = 0, seen = 0, i;
  reg [255:0] testname;

  // The activation side stays a simple registered read; the weight side
  // is the generated memory.
  always @(posedge clk) a_data <= amem[a_addr];

  wmem wm (.clk(clk), .rst_n(rst_n), .load_start(load_start),
           .load_valid(load_valid), .load_data(load_data),
           .load_count(load_count), .rd_addr(w_addr[{awm}:0]),
           .rd_data(rd_data));

  matvec seq (.clk(clk), .rst_n(rst_n), .start(start), .depth(depth),
              .cols(cols), .a_addr(a_addr), .w_addr(w_addr),
              .mac_valid(mac_valid), .mac_clear(mac_clear),
              .col_valid(col_valid), .col_index(col_index), .busy(busy));

  mac dut (.clk(clk), .rst_n(rst_n), .clear(mac_clear),
           .a(a_data), .b(rd_data), .valid_in(mac_valid),
           .acc(acc), .valid_out(mac_vout));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && col_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (acc !== expect_col[col_index]) begin
        $display("TB_FAIL test=%0s col=%0d expected_acc=%0d got_acc=%0d",
                 testname, col_index, expect_col[col_index], acc);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  end

  initial begin
    for (i = 0; i < {depth}; i = i + 1)
      amem[i] = (i * 104729 + 7) % {m} - {half};
{exp}
    testname = "reset_init";
    repeat (3) @(negedge clk);
    checks = checks + 1;
    // Reset has to clear the write pointer. Without this the testbench
    // never depends on it, because load_start clears it too and every
    // load starts with one.
    if (load_count !== 0) begin
      $display("TB_FAIL test=reset_init col=0 expected_acc=0 got_acc=%0d",
               load_count);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    rst_n = 1;
    @(negedge clk);
    checks = checks + 1;
    if (load_count !== 0) begin
      $display("TB_FAIL test=reset_init col=0 expected_acc=0 got_acc=%0d",
               load_count);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    testname = "load";

    // Stream the tile in, as a host or a DMA engine would.
    load_start = 1;
    @(negedge clk);
    load_start = 0;
    for (i = 0; i < {nmem}; i = i + 1) begin
      load_data = (i * 7919 + 13) % {m} - {half};
      load_valid = 1;
      @(negedge clk);
    end
    load_valid = 0;
    @(negedge clk);
    checks = checks + 1;
    if (load_count !== {nmem}) begin
      $display("TB_FAIL test=%0s expected_acc=%0d got_acc=%0d",
               "load_count", {nmem}, load_count);
      $display("TB_RESULT: FAIL");
      $finish;
    end

    testname = "system";
    depth = {depth};
    cols = {cols};
    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;
    for (i = 0; i < {cols} * ({depth} + 10) + 60; i = i + 1)
      @(negedge clk);
    checks = checks + 1;
    if (seen !== {cols}) begin
      $display("TB_FAIL test=%0s expected_acc=%0d got_acc=%0d",
               "column_count", {cols}, seen);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE tiles=%0d span_cycles=%0d latency_cycles=%0d",
             1, {nmem}, {depth});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_wmem(ms=None, spec_file="spec_wmem.json",
                  tb_file="tb_wmem.v"):
    """Write the derived weight memory spec and its system testbench."""
    ms = ms or load_model_spec()
    spec = derive_wmem_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_wmem_testbench(spec))
    return spec


def derive_softmax_spec(ms):
    """model spec -> softmax sequencer spec.

    The exponential and the reciprocal exist as blocks, but nothing
    joined them: softmax is a max pass, an exponential pass that
    accumulates a sum, one reciprocal, and a normalising multiply. This
    sequences all four and instantiates the two transcendental units
    itself, which makes it the first generated block that contains
    others rather than sitting beside them.
    """
    e = derive_exp_spec(ms)
    r = derive_recip_spec(ms)
    cap = 256                       # one attention row tile
    nw = (cap - 1).bit_length()
    return {
        "name": "softmax_%s" % ms["name"],
        "description": "Softmax sequencer over a row of at most %d "
                       "scores: max pass, exponential pass with sum, one "
                       "reciprocal, normalising multiply" % cap,
        "top_module": "softmax",
        "unit": "row",
        "parameters": {
            "capacity": cap, "index_width": nw,
            "score_width": e["parameters"]["in_width"],
            "score_frac": e["parameters"]["in_frac"],
            "weight_width": e["parameters"]["out_width"],
            "weight_frac": e["parameters"]["out_frac"],
            "exp_stages": e["parameters"]["pipeline_stages"],
            "recip_stages": r["parameters"]["pipeline_stages"],
            "recip_in_width": r["parameters"]["in_width"],
            "recip_out_width": r["parameters"]["out_width"],
            "shift_bias": r["parameters"]["shift_bias"],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "rule": "capacity is one attention row tile; the score and "
                    "weight formats come from the exponential unit and "
                    "the normalising shift from the reciprocal, so a "
                    "change to either changes this block",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "begin a row"},
            {"name": "n", "dir": "input", "width": nw + 1,
             "desc": "number of scores in the row"},
            {"name": "s_addr", "dir": "output", "width": nw,
             "desc": "score index being read"},
            {"name": "s_data", "dir": "input",
             "width": e["parameters"]["in_width"], "signed": True,
             "desc": "score, registered read, one cycle after s_addr"},
            {"name": "w_valid", "dir": "output", "width": 1,
             "desc": "a normalised weight is on w_data"},
            {"name": "w_index", "dir": "output", "width": nw,
             "desc": "which score the weight belongs to"},
            {"name": "w_data", "dir": "output",
             "width": e["parameters"]["out_width"],
             "desc": "weight in Q0.%d" % e["parameters"]["out_frac"]},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a row is in progress"},
        ],
        "behavior": [
            "start begins a row of n scores.",
            "The first pass reads every score and keeps the maximum, "
            "because the exponential is only defined for non-positive "
            "arguments and subtracting the row maximum is what "
            "guarantees that.",
            "The second pass reads them again, feeds score minus "
            "maximum to the exponential unit, buffers each result and "
            "accumulates their sum.",
            "The sum then goes through the reciprocal unit once. A "
            "divide per weight would be absurd.",
            "The third pass multiplies each buffered exponential by "
            "that reciprocal and emits it with w_valid.",
            "Every read is registered, so addresses lead data by one "
            "cycle throughout.",
        ],
    }


def softmax_golden(scores, p):
    """Exact model of the sequencer, built from the two unit models so
    the testbench checks the composition rather than a fresh
    approximation of softmax."""
    e_p = {"in_width": p["score_width"], "in_frac": p["score_frac"],
           "out_frac": p["weight_frac"], "lut_bits": p["score_frac"]}
    r_p = {"in_width": p["recip_in_width"],
           "out_width": p["recip_out_width"],
           "lut_bits": 8, "shift_bias": p["shift_bias"]}
    mx = max(scores)
    ex = [exp_golden(s - mx, e_p) for s in scores]
    tot = min(sum(ex), (1 << r_p["in_width"]) - 1)
    m, k = recip_golden(max(1, tot), r_p)
    # recip_apply already divides, so shifting again here would scale
    # the weight down by a second factor of 2**weight_frac.
    return [min((1 << p["weight_frac"]),
                recip_apply(v << p["weight_frac"], m, k, r_p))
            for v in ex], ex, tot


def _softmax_discriminating_row(p, rnd, tries=200000):
    """Find a row where a one-count error in the normalising multiply
    changes a weight. Returns None if none is found, in which case the
    testbench simply does without it rather than pretending."""
    sw = p["score_width"]
    lo = -(1 << (sw - 1))
    hi = -lo - 1
    r_p = {"in_width": p["recip_in_width"],
           "out_width": p["recip_out_width"],
           "lut_bits": 8, "shift_bias": p["shift_bias"]}
    for _ in range(tries):
        n = rnd.randrange(2, 6)
        row = [rnd.randrange(lo // 2, hi // 2) for _ in range(n)]
        w, ex, tot = softmax_golden(row, p)
        m, k = recip_golden(max(1, tot), r_p)
        s = p["shift_bias"] - k - p["weight_frac"]
        if s <= 0:
            continue
        for v in ex:
            if ((v * m) & ((1 << s) - 1)) == (1 << s) - 1:
                return row
    return None


def render_softmax_testbench(spec):
    p = spec["parameters"]
    sw, ww, nw = p["score_width"], p["weight_width"], p["index_width"]
    rnd = random.Random(53)
    rows = []
    lo = -(1 << (sw - 1))
    hi = -lo - 1
    for n in (1, 2, 5, 16, 64):
        # Straddling zero on purpose: with every score negative the row
        # maximum is zero and subtracting it is a no-op, so a design
        # that skips that step passes.
        rows.append([rnd.randrange(hi // 4, hi // 2) for _ in range(n)])
    rows.append([1234] * 8)                    # all equal, all positive
    rows.append([hi // 2] + [hi // 8] * 7)     # one dominant score
    rows.append([lo // 4, 0, hi // 4, 7])      # mixed signs
    disc = _softmax_discriminating_row(p, rnd)
    if disc:
        # A row whose product sits one count below a shift boundary, so
        # a single LSB of error in the normalising multiply carries into
        # the weight. Without it that mutation survives every vector,
        # because the product is shifted right by about seventeen bits
        # and the chance of a random row landing on the boundary is
        # roughly one in a hundred thousand. The testbench is generated,
        # so it can go and find one.
        rows.append(disc)
    body = []
    for ri, sc in enumerate(rows):
        w, _, _ = softmax_golden(sc, p)
        body.append("    // row %d, n=%d" % (ri, len(sc)))
        for i, v in enumerate(sc):
            body.append("    smem[%d] = %s;" % (i, _slit(v, sw)))
        for i, v in enumerate(w):
            body.append("    expect_w[%d] = %d'd%d;" % (i, ww, v))
        body.append("    run_row(%d'd%d);" % (nw + 1, len(sc)))
    return SOFTMAX_TB.format(
        swm=sw - 1, wwm=ww - 1, nwm=nw - 1, nw=nw + 1, cap=p["capacity"],
        rows="\n".join(body), nrows=len(rows))


SOFTMAX_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the softmax sequencer, which instantiates
// the generated exponential and reciprocal units. Golden weights are
// built from those same unit models, so this checks the composition
// rather than a fresh approximation of softmax.
module tb_softmax;
  reg clk = 0, rst_n = 0, start = 0;
  reg  [{nw}-1:0] n = 0;
  wire [{nwm}:0] s_addr;
  wire w_valid, busy;
  wire [{nwm}:0] w_index;
  wire [{wwm}:0] w_data;

  reg signed [{swm}:0] smem [0:{cap}-1];
  reg        [{wwm}:0] expect_w [0:{cap}-1];
  reg signed [{swm}:0] s_data;
  integer checks = 0, seen = 0, i;
  reg [255:0] testname;

  always @(posedge clk) s_data <= smem[s_addr];

  softmax dut (.clk(clk), .rst_n(rst_n), .start(start), .n(n),
               .s_addr(s_addr), .s_data(s_data), .w_valid(w_valid),
               .w_index(w_index), .w_data(w_data), .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && w_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (w_data !== expect_w[w_index]) begin
        $display("TB_FAIL test=%0s idx=%0d expected_w=%0d got_w=%0d",
                 testname, w_index, expect_w[w_index], w_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  end

  task run_row(input [{nw}-1:0] cnt);
    begin
      seen = 0;
      n = cnt;
      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      while (busy) @(negedge clk);
      repeat (4) @(negedge clk);
      checks = checks + 1;
      if (seen !== cnt) begin
        $display("TB_FAIL test=%0s idx=0 expected_w=%0d got_w=%0d",
                 "weight_count", cnt, seen);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "softmax";
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
{rows}

    $display("TB_PROFILE rows=%0d span_cycles=%0d latency_cycles=%0d",
             {nrows}, {nrows} * 64, 8);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_softmax(ms=None, spec_file="spec_softmax.json",
                     tb_file="tb_softmax.v"):
    """Write the derived softmax sequencer spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_softmax_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_softmax_testbench(spec))
    return spec


def requant_golden(acc, scale, sh, out_width):
    """Scale, round to nearest, saturate. Shared by the requantizer's own
    testbench and by anything that sequences it, so the two cannot
    disagree about what requantization means."""
    hi = (1 << (out_width - 1)) - 1
    lo = -(1 << (out_width - 1))
    prod = acc * scale
    r = (prod + (1 << (sh - 1))) >> sh if sh > 0 else prod
    if r > hi:
        return hi, 1
    if r < lo:
        return lo, 1
    return r, 0


def derive_mlp_spec(ms):
    """model spec -> MLP layer sequencer spec.

    The blocks so far do one operation each. This drives two matmuls in
    order and routes the activations between them: the first result is
    requantized, rectified and written to the other bank of an
    activation buffer, which the second matmul then reads. That routing
    is the thing that was missing, and it is what makes a layer out of a
    matmul.
    """
    c = derive_chiplet_spec(ms)
    mv = derive_matvec_spec(ms)
    rq = derive_requant_spec(ms)
    dw = c["parameters"]["data_width"]
    aw = c["parameters"]["acc_width"]
    bank = 64                       # activations per bank, one tile
    bw = (bank - 1).bit_length()
    return {
        "name": "mlp_%s" % ms["name"],
        "description": "MLP layer sequencer: two matmuls with a "
                       "requantize and rectify between them, over a "
                       "two-bank activation buffer of %d each" % bank,
        "top_module": "mlp",
        "unit": "layer",
        "parameters": {
            "data_width": dw, "acc_width": aw, "bank": bank,
            "bank_width": bw,
            "depth_width": mv["parameters"]["depth_width"],
            "col_width": mv["parameters"]["col_width"],
            "addr_width": mv["parameters"]["addr_width"],
            "scale_width": rq["parameters"]["scale_width"],
            "shift_width": rq["parameters"]["shift_width"],
            "requant_stages": rq["parameters"]["pipeline_stages"],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "rule": "the activation bank is one tile; the matmul and "
                    "requantizer formats come from those blocks, so a "
                    "change to either changes this one",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "load_valid", "dir": "input", "width": 1,
             "desc": "write an input activation into bank zero"},
            {"name": "load_data", "dir": "input", "width": dw,
             "signed": True, "desc": "input activation"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "run the layer"},
            {"name": "depth1", "dir": "input",
             "width": mv["parameters"]["depth_width"],
             "desc": "reduction length of the first matmul"},
            {"name": "cols1", "dir": "input",
             "width": mv["parameters"]["col_width"],
             "desc": "outputs of the first matmul"},
            {"name": "cols2", "dir": "input",
             "width": mv["parameters"]["col_width"],
             "desc": "outputs of the second matmul"},
            {"name": "scale1", "dir": "input",
             "width": rq["parameters"]["scale_width"],
             "desc": "requantizer scale for the first matmul"},
            {"name": "shift1", "dir": "input",
             "width": rq["parameters"]["shift_width"],
             "desc": "requantizer shift for the first matmul"},
            {"name": "scale2", "dir": "input",
             "width": rq["parameters"]["scale_width"],
             "desc": "requantizer scale for the second matmul"},
            {"name": "shift2", "dir": "input",
             "width": rq["parameters"]["shift_width"],
             "desc": "requantizer shift for the second matmul"},
            {"name": "w_addr", "dir": "output",
             "width": mv["parameters"]["addr_width"],
             "desc": "weight address, into the weight memory"},
            {"name": "w_data", "dir": "input", "width": dw, "signed": True,
             "desc": "weight, registered read"},
            {"name": "o_valid", "dir": "output", "width": 1,
             "desc": "a layer output is on o_data"},
            {"name": "o_index", "dir": "output", "width": bw,
             "desc": "which output"},
            {"name": "o_data", "dir": "output", "width": dw, "signed": True,
             "desc": "layer output activation"},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a layer is in progress"},
        ],
        "behavior": [
            "load_valid writes input activations into bank zero in order.",
            "start runs the first matmul over bank zero, requantizes each "
            "column with scale1 and shift1, rectifies it and writes it to "
            "bank one.",
            "It then runs the second matmul over bank one, requantizes "
            "with scale2 and shift2, and emits each result on o_data.",
            "The second matmul's reduction length is the first one's "
            "column count, because that is what the first matmul "
            "produced. Taking it from an input instead would let the two "
            "disagree.",
            "The weight address continues across both matmuls, so the "
            "second matrix follows the first in memory.",
        ],
    }


def render_mlp_testbench(spec):
    """Testbench for the layer: the sequencer with the real matmul
    sequencer, the real MAC and the real requantizer under it."""
    p = spec["parameters"]
    dw, aw = p["data_width"], p["acc_width"]
    mw, sw = p["scale_width"], p["shift_width"]
    d1, c1, c2 = 8, 6, 5
    m = (1 << dw) - 5
    half = m // 2
    rnd = random.Random(61)
    acts = [rnd.randrange(-half, half) for _ in range(d1)]
    w1 = [rnd.randrange(-half, half) for _ in range(d1 * c1)]
    w2 = [rnd.randrange(-half, half) for _ in range(c1 * c2)]
    sh = 12
    sc = 1 << (sh - 4)              # a gentle scale, mostly in range
    h_acc = [sum(acts[r] * w1[c * d1 + r] for r in range(d1))
             for c in range(c1)]
    h = [max(0, requant_golden(v, sc, sh, dw)[0]) for v in h_acc]
    y_acc = [sum(h[r] * w2[c * c1 + r] for r in range(c1))
             for c in range(c2)]
    y = [requant_golden(v, sc, sh, dw)[0] for v in y_acc]
    init = "\n".join(
        ["    acts[%d] = %s;" % (i, _slit(v, dw))
         for i, v in enumerate(acts)]
        + ["    wmem_tb[%d] = %s;" % (i, _slit(v, dw))
           for i, v in enumerate(w1 + w2)]
        + ["    expect_y[%d] = %s;" % (i, _slit(v, dw))
           for i, v in enumerate(y)])
    return MLP_TB.format(
        dwm=dw - 1, awm=aw - 1, mwm=mw - 1, swm=sw - 1,
        bwm=p["bank_width"] - 1, depwm=p["depth_width"] - 1,
        colwm=p["col_width"] - 1, addrwm=p["addr_width"] - 1,
        d1=d1, c1=c1, c2=c2, nw=len(w1 + w2), sc=sc, sh=sh, init=init)


MLP_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the MLP layer sequencer, with the real matmul
// sequencer, the real MAC and the real requantizer under it. Golden
// values come from the shared requantization model, so the layer is
// checked against the arithmetic its parts are specified to do.
module tb_mlp;
  reg clk = 0, rst_n = 0, start = 0, load_valid = 0;
  reg signed [{dwm}:0] load_data = 0;
  reg [{depwm}:0] depth1 = 0;
  reg [{colwm}:0] cols1 = 0, cols2 = 0;
  reg [{mwm}:0] scale1 = 0, scale2 = 0;
  reg [{swm}:0] shift1 = 0, shift2 = 0;
  wire [{addrwm}:0] w_addr;
  wire o_valid, busy;
  wire [{bwm}:0] o_index;
  wire signed [{dwm}:0] o_data;

  reg signed [{dwm}:0] acts [0:{d1}-1];
  reg signed [{dwm}:0] wmem_tb [0:{nw}-1];
  reg signed [{dwm}:0] expect_y [0:{c2}-1];
  reg signed [{dwm}:0] w_data;
  integer checks = 0, seen = 0, i;
  reg [255:0] testname;

  always @(posedge clk) w_data <= wmem_tb[w_addr];

  mlp dut (.clk(clk), .rst_n(rst_n), .load_valid(load_valid),
           .load_data(load_data), .start(start), .depth1(depth1),
           .cols1(cols1), .cols2(cols2), .scale1(scale1),
           .shift1(shift1), .scale2(scale2), .shift2(shift2),
           .w_addr(w_addr), .w_data(w_data), .o_valid(o_valid),
           .o_index(o_index), .o_data(o_data), .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && o_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (o_data !== expect_y[o_index]) begin
        $display("TB_FAIL test=%0s out=%0d expected_y=%0d got_y=%0d",
                 testname, o_index, expect_y[o_index], o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  end

  initial begin
{init}
    testname = "layer";
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    for (i = 0; i < {d1}; i = i + 1) begin
      load_data = acts[i]; load_valid = 1;
      @(negedge clk);
    end
    load_valid = 0;
    depth1 = {d1}; cols1 = {c1}; cols2 = {c2};
    scale1 = {sc}; shift1 = {sh}; scale2 = {sc}; shift2 = {sh};
    @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;
    while (busy) @(negedge clk);
    repeat (8) @(negedge clk);
    checks = checks + 1;
    if (seen !== {c2}) begin
      $display("TB_FAIL test=%0s out=0 expected_y=%0d got_y=%0d",
               "output_count", {c2}, seen);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE layers=%0d span_cycles=%0d latency_cycles=%0d",
             1, {d1} * {c1} + {c1} * {c2}, 16);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_mlp(ms=None, spec_file="spec_mlp.json", tb_file="tb_mlp.v"):
    """Write the derived MLP layer spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_mlp_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_mlp_testbench(spec))
    return spec


def derive_requant_spec(ms):
    """model spec -> requantization spec.

    This is the step between two matmuls, and until now it lived in Python
    while the repo claimed to be generating the inference datapath. A wide
    signed accumulator has to come back to the next layer's operand width:
    multiply by a per-tensor scale, round, and saturate rather than wrap,
    because wrapping turns one saturated activation into a value of the
    opposite sign and the error propagates through every later layer.

    The scale is a fixed-point reciprocal applied as a multiply and an
    arithmetic shift, which is how quantized inference does it in hardware:
    a divider per activation would be absurd, and the scale is constant for
    the whole tensor so it can be a run-time input.
    """
    wb, ab = ms["weight_bits"], ms["activation_bits"]
    dw = max(wb, ab)
    depth = max(ms["d_model"], ms["d_ff"])
    aw = wb + ab + math.ceil(math.log2(depth))
    mw = 18                       # multiplier operand width for the scale
    shift_w = math.ceil(math.log2(aw + mw)) + 1
    # Partial products for the scale multiply. Depth grows with both
    # operand widths, so the split has to track the accumulator too: at a
    # fixed three the 46-bit accumulator missed its clock. Roughly six
    # multiplier bits per partial product at 28 bits of accumuland, and
    # one more split for every eight bits of accumulator beyond that.
    splits = min(mw, 3 + max(0, (aw - 28 + 7) // 8))
    stages = requant_stages(aw, mw, splits)
    return {
        "name": "requant%d_%s" % (dw, ms["name"]),
        "description": "Requantization unit derived from %s: scales a %d-bit "
                       "signed accumulator back to a %d-bit signed operand "
                       "by a fixed-point multiply, arithmetic shift with "
                       "round to nearest, and saturation"
                       % (ms["name"], aw, dw),
        "top_module": "requant",
        "unit": "activation",
        "parameters": {
            "acc_width": aw,
            "out_width": dw,
            "scale_width": mw,
            "shift_width": shift_w,
            "scale_splits": splits,
            "signed": True,
            "pipeline_stages": stages,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "weight_bits": wb,
            "activation_bits": ab,
            "reduction_depth": depth,
            "rule": "in_width = acc_width from the chiplet derivation; "
                    "out_width = max(weight_bits, activation_bits); the "
                    "scale is a %d-bit fixed-point multiplier with a "
                    "run-time arithmetic shift" % mw,
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "acc_in", "dir": "input", "width": aw, "signed": True,
             "desc": "accumulator from the MAC, signed two's complement"},
            {"name": "scale", "dir": "input", "width": mw, "signed": False,
             "desc": "fixed-point multiplier, unsigned"},
            {"name": "shift", "dir": "input", "width": shift_w,
             "desc": "arithmetic right shift applied after the multiply"},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "acc_in valid"},
            {"name": "q_out", "dir": "output", "width": dw, "signed": True,
             "desc": "requantized activation, saturated not wrapped"},
            {"name": "sat", "dir": "output", "width": 1,
             "desc": "high when this output saturated"},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "q_out updated this cycle"},
        ],
        "behavior": [
            "Stage 1 registers two half-width partial products of "
            "acc_in * scale; stage 2 recombines them into the full "
            "product.",
            "Stage 3 adds half an LSB, precomputed a stage earlier, for "
            "round to nearest.",
            "Stage 4 arithmetically shifts that sum right by shift.",
            "Stage 5 saturates to the signed %d-bit range [%d, %d] and "
            "registers the result."
            % (dw, -(1 << (dw - 1)), (1 << (dw - 1)) - 1),
            "sat is high on any output that had to be clamped.",
            "Saturating, not wrapping, is required: a wrapped overflow "
            "flips the sign of a large activation and corrupts every "
            "later layer.",
            "Latency from valid_in to valid_out is %d cycles." % stages,
        ],
    }


def render_requant_testbench(spec):
    """Testbench for the requantizer, with the golden values computed here
    in Python rather than by a mirror of the design, so a design that
    reproduces its own mistake cannot pass."""
    p = spec["parameters"]
    aw, dw, mw = p["acc_width"], p["out_width"], p["scale_width"]
    lo, hi = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    rnd = random.Random(7)

    def golden(acc, scale, sh):
        return requant_golden(acc, scale, sh, dw)

    cases = []
    # Directed: zero, the saturation edges in both directions, and the
    # rounding boundary, then random coverage.
    sh = 12
    unit = 1 << sh
    directed = [(0, unit, sh), (hi, unit, sh), (lo, unit, sh),
                (hi + 1, unit, sh), (lo - 1, unit, sh),
                ((1 << (aw - 2)), unit, sh), (-(1 << (aw - 2)), unit, sh),
                (3, unit // 2, sh), (-3, unit // 2, sh),
                (1, unit // 2, sh), (-1, unit // 2, sh)]
    for acc, sc, s_ in directed:
        cases.append(("directed", acc, sc, s_))
    # Random coverage, with the shift chosen from the magnitude of the
    # product so the result lands in the representable range. Picking the
    # shift independently makes almost every vector saturate, and a
    # saturated output hides any arithmetic error below it: mutation
    # testing caught exactly that, an inverted adder in the low half of a
    # split add surviving 120 random vectors.
    n_sat = 0
    for _ in range(140):
        acc = rnd.randrange(-(1 << (aw - 1)), 1 << (aw - 1))
        sc = rnd.randrange(1, 1 << (mw - 1))
        mag = abs(acc * sc)
        want_bits = rnd.randrange(1, dw)      # target magnitude, in bits
        s_ = max(0, min((1 << p["shift_width"]) - 1,
                        mag.bit_length() - want_bits))
        q, st = golden(acc, sc, s_)
        n_sat += st
        cases.append(("random", acc, sc, s_))
    # Small operands with a small shift. With a wide accumulator the shift
    # is always large, so the bottom of the product never reaches the
    # output and the low partial products are untested: mutation testing
    # showed an inverted adder in the low half of a split add surviving
    # every vector. These vectors keep the low bits in the result.
    for _ in range(60):
        acc = rnd.randrange(-(1 << min(aw - 1, 14)), 1 << min(aw - 1, 14))
        sc = rnd.randrange(1, 1 << min(mw - 1, 10))
        s_ = rnd.randrange(0, 6)
        q, st = golden(acc, sc, s_)
        if st:            # keep these in range, the clamp has its own cases
            continue
        cases.append(("small_operands", acc, sc, s_))

    # A testbench that never saturates would not test the clamp either.
    for _ in range(20):
        acc = rnd.randrange(-(1 << (aw - 1)), 1 << (aw - 1))
        sc = rnd.randrange(1 << (mw - 2), 1 << (mw - 1))
        cases.append(("saturating", acc, sc, rnd.randrange(0, 4)))

    body = []
    for name, acc, sc, s_ in cases:
        want, wsat = golden(acc, sc, s_)
        body.append('    testname = "%s";' % name)
        body.append("    drive(%s, %d'd%d, %d'd%d, %s, 1'b%d);"
                    % (_slit(acc, aw), mw, sc, p["shift_width"], s_,
                       _slit(want, dw), wsat))
    return REQUANT_TB.format(
        awm=aw - 1, dwm=dw - 1, mwm=mw - 1, swm=p["shift_width"] - 1,
        cases="\n".join(body), n=len(cases),
        settle=p["pipeline_stages"] - 1)


def _slit(v, w):
    """Signed Verilog literal: the sign goes outside the sized constant."""
    return "-%d'sd%d" % (w, -v) if v < 0 else "%d'sd%d" % (w, v)


REQUANT_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Self-checking testbench for the requantizer. Golden outputs are computed
// in Python from the quantization rule, not by a Verilog mirror of the
// design, so a design that reproduces its own mistake cannot pass.
module tb_requant;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg  signed [{awm}:0] acc_in = 0;
  reg         [{mwm}:0] scale = 0;
  reg         [{swm}:0] shift = 0;
  wire signed [{dwm}:0] q_out;
  wire sat, valid_out;
  integer checks = 0, i;
  reg [255:0] testname;
  reg signed [{dwm}:0] exp_q;
  reg exp_sat;

  requant dut (.clk(clk), .rst_n(rst_n), .acc_in(acc_in), .scale(scale),
               .shift(shift), .valid_in(valid_in), .q_out(q_out),
               .sat(sat), .valid_out(valid_out));
  always #5 clk = ~clk;

  // One transaction at a time: drive, wait the pipeline out, then check.
  task drive(input signed [{awm}:0] a, input [{mwm}:0] sc,
             input [{swm}:0] sh, input signed [{dwm}:0] want,
             input wsat);
    begin
      @(negedge clk); acc_in = a; scale = sc; shift = sh; valid_in = 1;
      @(negedge clk); valid_in = 0;
      // Wait the declared pipeline out. Hardcoding a depth here would make
      // a latency change look like a datapath bug.
      repeat ({settle}) @(negedge clk);
      checks = checks + 1;
      if (q_out !== want || sat !== wsat || valid_out !== 1'b1) begin
        $display("TB_FAIL test=%0s acc=%0d scale=%0d shift=%0d expected_q=%0d got_q=%0d expected_sat=%b got_sat=%b vout=%b",
                 testname, a, sc, sh, want, q_out, wsat, sat, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  // Reset has to hold the outputs quiet. Without this the testbench never
  // depends on the reset branch at all: every transaction flushes the
  // pipeline with real values before it is checked, so a dead reset passes.
  // The endpoint testbench had exactly this hole.
  task expect_quiet;
    begin
      checks = checks + 1;
      if (valid_out !== 1'b0 || q_out !== 0 || sat !== 1'b0) begin
        $display("TB_FAIL test=reset_init expected_q=0 got_q=%0d expected_sat=0 got_sat=%b vout=%b",
                 q_out, sat, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "reset_init";
    repeat (3) @(negedge clk);
    expect_quiet;
    rst_n = 1;
    @(negedge clk);
    expect_quiet;
{cases}

    $display("TB_PROFILE activations=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, {n} * (2 + {settle}), 1 + {settle});
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


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


def generate_requant(ms=None, spec_file="spec_requant.json",
                     tb_file="tb_requant.v"):
    """Write the derived requantizer spec and testbench, return the spec."""
    ms = ms or load_model_spec()
    spec = derive_requant_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_requant_testbench(spec))
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
