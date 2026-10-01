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


def acc_width(ms):
    """The datapath's accumulator width, shared by the MAC and the
    requantizer so every block that instantiates them agrees.

    The rule is weight_bits + activation_bits + ceil(log2(depth)), the
    widest reduction the model has. The attention head's weighted sum goes
    through the same requantizer, and it needs data_width + 16 bits (the
    exponential's 15 fractional bits of weight, plus sign), so that is a
    floor. Every model the sweep derives already clears it; the
    16-dimensional checkpoint does not, at 21 bits against 24.
    """
    wb, ab = ms["weight_bits"], ms["activation_bits"]
    depth = max(ms["d_model"], ms["d_ff"])
    return max(wb + ab + math.ceil(math.log2(depth)), max(wb, ab) + 16)


def derive_chiplet_spec(ms):
    """model spec -> chiplet spec. The derivation is recorded in the spec so
    the signed-off profile can carry its own provenance."""
    wb, ab = ms["weight_bits"], ms["activation_bits"]
    dw = max(wb, ab)
    assert dw >= 4, "testbench stimulus assumes at least a 4-bit datapath"
    depth = max(ms["d_model"], ms["d_ff"])
    guard = math.ceil(math.log2(depth))
    aw = acc_width(ms)
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
        ] + ([
            # The accumulator is sized for the quantized widths, not the
            # port width. The formal proof found that with int4 weights on
            # an 8-bit port, a full-range value on b overflows it, so the
            # range the rule depends on is stated rather than implied.
            "a carries %d-bit activations and b carries %d-bit weights, "
            "sign-extended to the %d-bit ports. The accumulator width "
            "depends on those ranges: a wider value on either port can "
            "overflow it." % (ab, wb, dw)
        ] if (wb < dw or ab < dw) else []),
    }


# Widest add that closes in one stage in the generic cell library, which is
# what the flow gates on. Measured: a 46-bit add leaves +0.26 ns at 100 MHz,
# a 64-bit add is 3.4 ns over, and every wide add violated, so it is a
# property of the library (no carry chain) rather than of one arrangement.
#
# It sat at 48 until a model with d_ff 4864 arrived. That gives a 29-bit
# accumulator and a 47-bit product, which fell in the gap between the
# measurement and the threshold and missed the clock. 46 is what the
# measurement actually supports: the widest add observed to close.
#
# Lowering it to 40 was tried, so that 46-bit adds split too. The generic
# library liked it (102 -> 109 MHz) and the real device did not (97.9 ->
# 93.5 MHz on an iCE40 HX8K). More pipeline stages mean more registers and
# more routing pressure, and on a real fabric this block is routing bound
# rather than logic bound. The two disagree in direction, so the cheaper
# structure wins.
WIDE_ADD_BITS = 46
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


def render_rom(name, entries, idx_bits, val_bits):
    """A constant table as its own module.

    Table contents are a generated artifact, not hand-written RTL. Every
    real flow treats ROM contents that way, and there is a concrete
    reason to here: an agent asked to emit 256 exact values of
    2**(i/256) is being asked to be a calculator, and it is not one.
    Measured, the three table blocks burned about seventy model calls
    and three hours failing on the first vector every time. Generating
    the table and letting the agent write the datapath around it puts
    each part where it belongs.
    """
    arms = "\n".join("      %d'd%d: val = %d'd%d;"
                      % (idx_bits, i, val_bits, v)
                      for i, v in enumerate(entries))
    return """// GENERATED by specgen.py: %d-entry constant table.
// Do not edit: these values are derived, and a hand edit here is a
// silent numerical change that no testbench above will attribute to it.
module %s (
  input      [%d:0] idx,
  output reg [%d:0] val
);
  always @(*) begin
    case (idx)
%s
      default: val = %d'd%d;
    endcase
  end
endmodule
""" % (len(entries), name, idx_bits - 1, val_bits - 1, arms, val_bits,
       entries[0])


def exp_rom(spec):
    p = spec["parameters"]
    return render_rom("exp_rom", exp_lut(p["lut_bits"], p["out_frac"]),
                      p["lut_bits"], p["out_width"])


def recip_rom(spec):
    p = spec["parameters"]
    return render_rom("recip_rom",
                      recip_lut(p["lut_bits"], p["out_width"]),
                      p["lut_bits"], p["out_width"])


def rsqrt_rom(spec):
    p = spec["parameters"]
    return render_rom("rsqrt_rom",
                      rsqrt_lut(p["lut_bits"], p["out_width"]),
                      p["lut_bits"], p["out_width"])


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
            "Stage 1 computes t = (x * %d) >>> 16 and registers it as a "
            "signed value. %d is log2(e) scaled by 2**16. The shift is "
            "arithmetic, so t truncates toward minus infinity, and t is a "
            "signed Q.%d number."
            % (LOG2E_Q16, LOG2E_Q16, fi),
            "Stage 2 splits t at bit %d. The integer part is n = t >>> %d, "
            "an arithmetic shift, so n is the floor and is zero or "
            "negative. The fractional part is f = t - (n << %d), which is "
            "always in [0, %d). Stage 2 registers exp_rom[f >> %d] and the "
            "shift amount sh = -n."
            % (fi, fi, fi, 1 << fi, fi - lb),
            "Stage 3 registers that table entry logically right-shifted "
            "by sh. The shift is unsigned: the table entry is a "
            "magnitude, not a signed number.",
            "When sh is greater than %d the result is zero. Do not "
            "saturate and do not let the shift wrap." % fo,
            "exp_rom is a separate module supplied as a source file, not "
            "something to write. Instantiate it. Its ports are "
            "input [%d:0] idx and output [%d:0] val, combinational, and "
            "it holds %d entries where entry i is 2**(i/%d) in Q0.%d."
            % (lb - 1, fo, 1 << lb, 1 << lb, fo),
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
                         one=1 << p["out_frac"],
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
        // e**x is below one for every negative x. Traced on the fourth
        // end-to-end run: Haiku's exponential gave 32857 for x = -1, five
        // drafts running, the table's entry 1 with no shift, and read only
        // "expected 32591 got 32857".
        if (xi < 0 && y >= {one})
          $display("TB_FAIL test=%0s x=%0d expected_y=%0d got_y=%0d vout=%b got_y_is_not_below_one_for_a_negative_x=1",
                   testname, xi, want, y, valid_out);
        else
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
            "Stage 1 computes k, the number of leading zeros of x within "
            "its %d-bit width, and registers both k and xn = x << k. "
            "After the shift bit %d of xn is set, so xn read as a Q1.%d "
            "number lies in [1,2)."
            % (iw, iw - 1, iw - 1),
            "Stage 2 registers recip_rom[idx] and k, where idx is bits "
            "[%d:%d] of xn, that is idx = (xn >> %d) and then the low %d "
            "bits of it. The leading one at bit %d is NOT part of idx: it "
            "is implicit, so the table is indexed by the fraction alone."
            % (iw - 2, iw - 1 - lb, iw - 1 - lb, lb, iw - 1),
            "Stage 3 registers those two values onto the mantissa and "
            "shift outputs. Do not apply the shift here. The consumer "
            "folds it into the multiply it was going to do anyway, which "
            "keeps the mantissa at full precision.",
            "The contract the testbench checks is that num/x equals "
            "(num * mantissa) >> (%d - shift), so mantissa must come out "
            "unshifted and shift must come out as k itself."
            % (ow + iw - 1),
            "recip_rom is a separate module supplied as a source file, "
            "not something to write. Instantiate it. Its ports are "
            "input [%d:0] idx and output [%d:0] val, combinational."
            % (lb - 1, ow - 1),
            "When x is zero the mantissa output saturates to all ones "
            "and the shift output is zero.",
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
            "Let b be the index of the highest set bit of x, counting "
            "from zero. Stage 1 computes e = b >> 1, an integer halving "
            "that rounds down, and s = %d - 2*e, and registers e together "
            "with xn = x << s. s is always even and never negative. "
            "After the shift the highest set bit of xn is at %d or %d, so "
            "xn read as a Q2.%d number lies in [1,4)."
            % (iw - 2, iw - 2, iw - 1, iw - 2),
            "Stage 2 registers rsqrt_rom[idx] where idx is the top %d "
            "bits of xn, that is idx = xn >> %d. Unlike the reciprocal "
            "unit, the leading one IS part of idx here, because the "
            "normalised range spans a factor of four and the table has "
            "to tell 1.x from 2.x."
            % (lb, iw - lb),
            "Stage 3 registers the table entry and e onto the mantissa "
            "and shift outputs. Do not apply the shift here.",
            "The contract the testbench checks is that num/sqrt(x) equals "
            "(num * mantissa) >> (%d + shift), so mantissa must come out "
            "unshifted and shift must come out as e itself." % ow,
            "rsqrt_rom is a separate module supplied as a source file, "
            "not something to write. Instantiate it. Its ports are "
            "input [%d:0] idx and output [%d:0] val, combinational."
            % (lb - 1, ow - 1),
            "A zero input returns the saturated mantissa, all ones, with "
            "a shift of zero. RMSNorm adds an epsilon before this unit "
            "so that case does not arise in practice.",
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
  integer cyc = 0, t0 = 0, load0 = 0, first_col = -1, last_col = 0;
  reg [255:0] testname;
  always @(posedge clk) cyc = cyc + 1;

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
      if (first_col < 0) first_col = cyc;
      last_col = cyc;
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
    t0 = cyc;
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
    // Measured: start to the last flagged column, and to the first.
    $display("TB_PROFILE columns=%0d span_cycles=%0d latency_cycles=%0d",
             {cols}, last_col - t0, first_col - t0);
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
  integer cyc = 0, t0 = 0, load0 = 0, first_col = -1, last_col = 0;
  reg [255:0] testname;
  always @(posedge clk) cyc = cyc + 1;

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
      if (first_col < 0) first_col = cyc;
      last_col = cyc;
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
    load0 = cyc;
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
    t0 = cyc;
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
    // Measured: the whole tile, load start to the last column, and the
    // compute latency from start to the first column.
    $display("TB_PROFILE tiles=%0d span_cycles=%0d latency_cycles=%0d",
             1, last_col - load0, first_col - t0);
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
    # One attention row tile, and never more than the context holds: a
    # 24-position model does not need 256-entry score buffers, and the
    # attention head's matvec walks cached positions as columns, so a
    # capacity past its column width cannot be addressed at all.
    cap = min(256, 1 << max(1, (ms["seq_len"] - 1).bit_length()))
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
            # Scores themselves are wider than the exponential's input. A
            # row can span far more than the exponential covers, and at
            # 13 bits a trained head's scores, 149 and 205 on the
            # Qwen-shaped checkpoint, had to be clamped to +-8 before the
            # maximum was even known, which flattened the weights.
            "score_in_width": e["parameters"]["in_width"] + 8,
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
             "width": e["parameters"]["in_width"] + 8, "signed": True,
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
            "start is a one-cycle pulse that begins a row of n scores, "
            "1 <= n <= %d, held stable for the whole row. busy must be "
            "high on the clock edge that samples start, so it already "
            "reads 1 one cycle later, and it stays high until the last "
            "weight has been emitted." % cap,
            "Scores are read through s_addr and s_data, and the read is "
            "registered: s_data carries score[a] on the cycle after "
            "s_addr = a. If s_addr is itself a register, that is two clock "
            "edges after the edge that loads a into it: one edge updates "
            "s_addr and the next one reads the memory. Count those two "
            "edges when pairing each score with its index.",
            "Pass 1 reads every score and keeps the maximum mx. The "
            "exponential is only defined for non-positive arguments, and "
            "subtracting the row maximum is what guarantees that.",
            "Pass 2 feeds e_i = expu(d_i) for every i, where d_i = "
            "max(score_i - mx, -2**%d): score_i - mx is never positive but "
            "can be far below the %d-bit signed input of expu, and below "
            "-2**%d the exponential is zero in its output format anyway, "
            "so the clamp is exact. Buffer every e_i, a %d-bit unsigned "
            "value, and accumulate their sum S. S is at most %d * 2**%d, "
            "so it fits the %d-bit input of recip without saturating."
            % (e["parameters"]["in_width"] - 1, e["parameters"]["in_width"],
               e["parameters"]["in_width"] - 1, e["parameters"]["out_width"],
               cap, e["parameters"]["out_frac"], r["parameters"]["in_width"]),
            "S then goes through recip once, giving a mantissa m and a "
            "shift k. A divide per weight would be absurd.",
            "Pass 3 emits, for every i, w_i = min(2**%d, ((e_i << %d) * m) "
            ">> (%d - k)), with w_index = i and w_valid high for exactly "
            "that one cycle. The product is unsigned and needs about %d "
            "bits; the min clamps any weight that would round above "
            "1.0."
            % (e["parameters"]["out_frac"], e["parameters"]["out_frac"],
               r["parameters"]["shift_bias"],
               e["parameters"]["out_width"] + e["parameters"]["out_frac"]
               + r["parameters"]["out_width"]),
            "Weights may be emitted in any order, but each index exactly "
            "once per row, and w_valid must be low at every other time.",
            "expu and recip are separate modules supplied as source "
            "files, not something to write. Instantiate them. expu has "
            "ports clk, rst_n, x (signed [%d:0]), valid_in, y ([%d:0]) and "
            "valid_out, with a latency of %d cycles. recip has ports clk, "
            "rst_n, x ([%d:0]), valid_in, y ([%d:0], the mantissa m), k "
            "([%d:0], the shift) and valid_out, with a latency of %d "
            "cycles. Use their valid_out rather than counting cycles."
            % (e["parameters"]["in_width"] - 1,
               e["parameters"]["out_width"] - 1,
               e["parameters"]["pipeline_stages"],
               r["parameters"]["in_width"] - 1,
               r["parameters"]["out_width"] - 1,
               (r["parameters"]["in_width"]).bit_length() - 1,
               r["parameters"]["pipeline_stages"]),
            "All state resets to zero: busy and w_valid are 0 during "
            "reset.",
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
    lo = -(1 << (p["score_width"] - 1))
    ex = [exp_golden(max(s - mx, lo), e_p) for s in scores]
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
    swi = p.get("score_in_width", sw)
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
    # Rows wider than the exponential covers: scores far apart, and
    # exactly at and one past the clamp on the difference from the max.
    top = (1 << (swi - 2)) + 12345
    rows.append([top, top - (1 << 20), -(1 << (swi - 2)), top - 3000])
    rows.append([top, top + lo, top + lo - 1, top + lo + 1, top - 5])
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
        flat = softmax_golden([0] * len(sc), p)[0][0]
        # The weights from scores with their low bits dropped, as a
        # design that takes s_data's top score_width bits gives. Traced
        # on the end-to-end run: Sonnet's and Opus's softmax took
        # s_data[20:8], two close scores became equal, and the only
        # message was that the row looked like one of equal scores.
        narrow = softmax_golden([v >> (swi - sw) for v in sc], p)[0] \
            if swi > sw else w
        body.append("    // row %d, n=%d" % (ri, len(sc)))
        body.append("    flat_w = %d'd%d;" % (ww, flat))
        for i, v in enumerate(sc):
            body.append("    smem[%d] = %s;" % (i, _slit(v, swi)))
        for i, v in enumerate(w):
            body.append("    expect_w[%d] = %d'd%d;" % (i, ww, v))
            body.append("    expect_n[%d] = %d'd%d;" % (i, ww, narrow[i]))
        body.append("    run_row(%d'd%d);" % (nw + 1, len(sc)))
    return SOFTMAX_TB.format(
        swm=swi - 1, wwm=ww - 1, nwm=nw - 1, nw=nw + 1, cap=p["capacity"],
        rows="\n".join(body), nrows=len(rows), drop=swi - sw)


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
  reg        [{wwm}:0] expect_n [0:{cap}-1];
  // The weight every index gets when a row's scores are all equal.
  reg        [{wwm}:0] flat_w = 0;
  reg signed [{swm}:0] s_data;
  integer checks = 0, seen = 0, i, j, other, best, dist, own;
  reg [255:0] testname;
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  always @(posedge clk) s_data <= smem[s_addr];

  softmax dut (.clk(clk), .rst_n(rst_n), .start(start), .n(n),
               .s_addr(s_addr), .s_data(s_data), .w_valid(w_valid),
               .w_index(w_index), .w_data(w_data), .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && w_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      // A weight for an index past the row has no expected value, and
      // comparing against one printed expected_w=x, which a traced agent
      // read as a datapath fault for three drafts running.
      if (w_index >= n) begin
        $display("TB_FAIL test=%0s idx=%0d n=%0d expected=no_weight_past_the_row got_w=%0d",
                 testname, w_index, n, w_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (w_data !== expect_w[w_index]) begin
        // A right value under the wrong index is a pipeline alignment
        // fault, not an arithmetic one. Traced, two different models
        // emitted weight 1 as weight 0 and read only "expected 14668 got
        // 18118", which says nothing about which of the two is wrong.
        // Nearest other index, reported only when the value is within
        // about 1.5% of it, which a coincidence rarely manages. Exact
        // equality missed a draft that was misaligned and also ten counts
        // off in its arithmetic.
        other = -1; best = 1 << 30;
        for (j = 0; j < n; j = j + 1) begin
          dist = (expect_w[j] > w_data) ? expect_w[j] - w_data
                                        : w_data - expect_w[j];
          if (j != w_index && dist < best) begin best = dist; other = j; end
        end
        own = (expect_w[w_index] > w_data) ? expect_w[w_index] - w_data
                                           : w_data - expect_w[w_index];
        // Every score read as one value makes every weight the same.
        // Traced, Sonnet's softmax gave 16383 for a two-score row in five
        // drafts running, the weight of a row of equal scores, and read
        // only "expected 14668 got 16383".
        if (n > 1 && w_data == flat_w && expect_w[w_index] != flat_w)
          if (w_data == expect_n[w_index] && expect_n[w_index] != expect_w[w_index])
            $display("TB_FAIL test=%0s idx=%0d n=%0d expected_w=%0d got_w=%0d got_w_is_the_weight_of_a_row_whose_scores_are_all_equal=1 got_w_is_the_weight_with_the_low_{drop}_bits_of_each_score_dropped=1",
                     testname, w_index, n, expect_w[w_index], w_data);
          else
            $display("TB_FAIL test=%0s idx=%0d n=%0d expected_w=%0d got_w=%0d got_w_is_the_weight_of_a_row_whose_scores_are_all_equal=1",
                     testname, w_index, n, expect_w[w_index], w_data);
        else if (w_data == expect_n[w_index] && expect_n[w_index] != expect_w[w_index])
          $display("TB_FAIL test=%0s idx=%0d expected_w=%0d got_w=%0d got_w_is_the_weight_with_the_low_{drop}_bits_of_each_score_dropped=1",
                   testname, w_index, expect_w[w_index], w_data);
        else if (other >= 0 && best < own && best <= expect_w[other] / 64 + 2)
          $display("TB_FAIL test=%0s idx=%0d expected_w=%0d got_w=%0d got_w_is_closest_to_the_expected_value_for_idx=%0d",
                   testname, w_index, expect_w[w_index], w_data, other);
        else
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
      t0 = cyc; first_out = -1;
      @(negedge clk); start = 0;
      while (busy) @(negedge clk);
      span = span + (cyc - t0);
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
             {nrows}, span, lat);
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



# --------------------------------------------------------------------------
# Attention head: scores against the K cache, softmax, weighted sum of V.
# --------------------------------------------------------------------------

def _round_shift(x, sh):
    """Arithmetic right shift with round to nearest, ties toward plus
    infinity: the requantizer's rounding with a scale of one."""
    return (x + (1 << (sh - 1))) >> sh if sh > 0 else x


def port_signature(spec):
    """A supplied module's exact ports, generated from its own spec so a
    composite's description cannot drift from what it instantiates.
    Traced on the attention head, a model given only prose about its
    sub-blocks guessed port names for eight drafts and never compiled."""
    parts = []
    for p in spec["ports"]:
        rng = "[%d:0] " % (p["width"] - 1) if p["width"] > 1 else ""
        parts.append("%s %s%s%s" % (p["dir"], "signed " if p.get("signed")
                                     else "", rng, p["name"]))
    return "%s (%s)" % (spec["top_module"], ", ".join(parts))


def derive_attn_spec(ms):
    """model spec -> single attention head spec, one decode step.

    The last piece of a transformer layer that was still missing. The
    query arrives as head_dim int8 values; keys and values come from a
    KV cache through registered read ports, one row per past position.
    Scores are q.k, requantized into the softmax's score format; the
    softmax block turns them into weights; the output is the
    weight-averaged value row, requantized back to int8.

    Every part is an existing generated block: matvec and the MAC for the
    scores, softmax for the weights, the requantizer for the output. The
    new logic is the routing between them and a multiply-accumulate of
    16-bit weights by 8-bit values, which the 8-bit MAC cannot do.
    """
    c = derive_chiplet_spec(ms)
    mv = derive_matvec_spec(ms)
    rq = derive_requant_spec(ms)
    sm = derive_softmax_spec(ms)
    dw, aw = c["parameters"]["data_width"], c["parameters"]["acc_width"]
    hd = ms.get("head_dim") or ms["d_model"] // ms["n_head"]
    cap = sm["parameters"]["capacity"]
    sw = sm["parameters"]["score_in_width"]
    ww = sm["parameters"]["weight_width"]
    hdw = max(1, (hd - 1).bit_length())
    addr_w = (cap * hd - 1).bit_length()
    nw = sm["parameters"]["index_width"] + 1
    # Scores saturate only at the edge of the softmax's wide input; the
    # softmax clamps each score minus the row maximum itself.
    smax = (1 << (sw - 1)) - 1
    wsum = dw + sm["parameters"]["weight_frac"] + 1
    assert wsum <= aw, "the requantizer input is narrower than the weighted sum"
    assert nw <= mv["parameters"]["col_width"], \
        "matvec cannot count as many cached positions as the head holds"
    return {
        "name": "attn_%s" % ms["name"],
        "description": "Single attention head for one decode step: q.k "
                       "scores over up to %d cached positions, softmax, "
                       "and the weighted sum of V over head_dim %d"
                       % (cap, hd),
        "top_module": "attn",
        "unit": "head",
        "parameters": {
            "data_width": dw, "acc_width": aw, "head_dim": hd,
            "head_dim_width": hdw, "capacity": cap, "n_width": nw,
            "addr_width": addr_w, "score_width": sw, "score_max": smax,
            "weight_width": ww, "weight_frac": sm["parameters"]["weight_frac"],
            # The weighted sum is bounded by the weights, not by the MAC's
            # reduction depth: |a| <= sum(w) * 2**(dw-1), and the weights
            # sum to under 2**(wf+1), so |a| < 2**(dw+wf) and dw+wf+1
            # signed bits hold it. Sizing it from the MAC gave a 46-bit add
            # at 16-bit operands that could not close timing.
            "wsum_width": wsum,
            "shift_s_width": 5,
            # Scores are (t << 4) >> shift_s, so shift_s covers x16 down to
            # /2**27. Right shifts alone forced q's scale times k's below
            # sqrt(head_dim) score counts, and a trained head needed three
            # times that: q saturated at int8 instead.
            "score_guard": 4,
            "scale_width": rq["parameters"]["scale_width"],
            "shift_width": rq["parameters"]["shift_width"],
            "requant_stages": rq["parameters"]["pipeline_stages"],
            "mv_depth_width": mv["parameters"]["depth_width"],
            "mv_col_width": mv["parameters"]["col_width"],
            "mv_addr_width": mv["parameters"]["addr_width"],
            "sm_index_width": sm["parameters"]["index_width"],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "softmax": sm["parameters"],
            "rule": "head_dim from the model; capacity and the score and "
                    "weight formats from the softmax block; the output "
                    "requantizer and accumulator widths from the MAC and "
                    "requantizer blocks",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "load_valid", "dir": "input", "width": 1,
             "desc": "write the next query element"},
            {"name": "load_data", "dir": "input", "width": dw, "signed": True,
             "desc": "query element, int8"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "run the head over n cached positions"},
            {"name": "n", "dir": "input", "width": nw,
             "desc": "number of cached positions, 1..%d" % cap},
            {"name": "shift_s", "dir": "input", "width": 5,
             "desc": "score shift: q.k to the softmax score format"},
            {"name": "scale_o", "dir": "input",
             "width": rq["parameters"]["scale_width"],
             "desc": "output requantizer scale"},
            {"name": "shift_o", "dir": "input",
             "width": rq["parameters"]["shift_width"],
             "desc": "output requantizer shift"},
            {"name": "k_addr", "dir": "output", "width": addr_w,
             "desc": "key cache address"},
            {"name": "k_data", "dir": "input", "width": dw, "signed": True,
             "desc": "key element, registered read"},
            {"name": "v_addr", "dir": "output", "width": addr_w,
             "desc": "value cache address"},
            {"name": "v_data", "dir": "input", "width": dw, "signed": True,
             "desc": "value element, registered read"},
            {"name": "o_valid", "dir": "output", "width": 1,
             "desc": "an output element is on o_data"},
            {"name": "o_index", "dir": "output", "width": hdw,
             "desc": "which element of the head's output"},
            {"name": "o_data", "dir": "output", "width": dw, "signed": True,
             "desc": "output element, int8"},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a head is being computed"},
        ],
        "behavior": [
            "Each cycle with load_valid high writes load_data into the "
            "query buffer at the next index, starting from 0 after reset "
            "and after every run, so q[0..%d] are loaded in order before "
            "start." % (hd - 1),
            "start is a one-cycle pulse, with n, shift_s, scale_o and "
            "shift_o held stable for the whole run. busy must be high on "
            "the clock edge that samples start, so it already reads 1 one "
            "cycle later, and it stays high until the last output has been "
            "emitted.",
            "The key and value caches are row-major: element d of "
            "position j is at address j*%d + d in both. k_data and v_data "
            "are registered reads: each carries the element at address a "
            "on the cycle after k_addr or v_addr = a. If the address is "
            "itself a register, that is two clock edges after the edge "
            "that loads a into it." % hd,
            "Scores: for j in 0..n-1, t_j = sum over d of q[d] * K[j][d], "
            "a signed %d-bit dot product. With u_j = t_j << 4, s_j = (u_j + "
            "2**(shift_s-1)) >>> shift_s when shift_s > 0, else u_j, then "
            "clamped to [%d, %d], the %d-bit signed score the softmax "
            "reads. The softmax clamps each score minus the row maximum "
            "to its exponential's range itself." % (aw, -(smax + 1), smax,
                                                    sw),
            "Weights: w = softmax(s) over the n scores, exactly as the "
            "supplied softmax block computes it: %d-bit unsigned Q0.%d, "
            "where %d is 1.0." % (ww, sm["parameters"]["weight_frac"],
                                   1 << sm["parameters"]["weight_frac"]),
            "Output: for d in 0..%d, a_d = sum over j of w_j * V[j][d], "
            "with w_j unsigned and V signed, then o_d = requant(a_d, "
            "scale_o, shift_o): multiply by scale_o, add 2**(shift_o-1) "
            "when shift_o > 0, arithmetic shift right by shift_o, "
            "saturate to [%d, %d]. The weights sum to under 2.0, so |a_d| "
            "stays under 2**%d: a %d-bit signed sum, sign-extended into "
            "the requantizer's %d-bit input."
            % (hd - 1, -(1 << (dw - 1)), (1 << (dw - 1)) - 1,
               dw + sm["parameters"]["weight_frac"], wsum, aw),
            "Emit each o_d with o_index = d and o_valid high for exactly "
            "that one cycle. Outputs may come in any order, but each index "
            "exactly once, and o_valid must be low at every other time.",
            "The scores and weights are held in memories named sbuf and "
            "pbuf, declared reg signed [%d:0] sbuf [0:%d] and reg [%d:0] "
            "pbuf [0:%d], with position j at index j. The testbench reads "
            "them directly to check the scores and the softmax, so these "
            "names and layouts are part of the interface."
            % (sw - 1, cap - 1, ww - 1, cap - 1),
            "matvec, mac, softmax and requant are separate modules "
            "supplied as source files, not something to write. Instantiate "
            "them with exactly these ports, connected by name: "
            + "; ".join(port_signature(x) for x in (mv, c, sm, rq)) + ".",
            "matvec walks a_addr 0..depth-1 and w_addr col*depth onward "
            "for each of cols columns, drives mac_valid one cycle behind "
            "the address, pulses mac_clear between columns, and pulses "
            "col_valid with col_index when a column's sum is on the MAC's "
            "acc. Wire mac_valid to the MAC's valid_in and mac_clear to its "
            "clear. With depth = %d and cols = n, matvec's w_addr is "
            "exactly the key address. softmax reads scores through s_addr "
            "and a registered s_data and emits w_valid, w_index and "
            "w_data. requant has a latency of exactly %d cycles."
            % (hd, rq["parameters"]["pipeline_stages"]),
            "The weighted sum multiplies a %d-bit unsigned weight by an "
            "%d-bit signed value, which the %d-bit MAC cannot do, so it "
            "is its own multiply-accumulate."
            % (ww, dw, dw),
            "All state resets to zero: busy and o_valid are 0 during "
            "reset.",
        ],
    }


def attn_golden(q, K, V, n, shift_s, scale_o, shift_o, p, sm_p):
    """Exact model of the head, built from the softmax and requantizer
    models so the testbench checks the composition."""
    smax = p["score_max"]
    t = [sum(q[d] * K[j][d] for d in range(p["head_dim"])) for j in range(n)]
    g = p.get("score_guard", 0)
    s = [max(-smax - 1, min(smax, _round_shift(v << g, shift_s))) for v in t]
    w = softmax_golden(s, sm_p)[0]
    a = [sum(w[j] * V[j][d] for j in range(n)) for d in range(p["head_dim"])]
    assert all(abs(v) < 1 << (p["wsum_width"] - 1) for v in a), \
        "weighted sum exceeds its derived accumulator"
    o = [requant_golden(v, scale_o, shift_o, p["data_width"])[0] for v in a]
    return t, s, w, a, o


def render_attn_testbench(spec, rows=((1, False), (3, False), (8, True),
                                      (40, False))):
    """Rows chosen to exercise each part: one position (a weight of 1.0),
    a few, a clamped score, and enough to stress the weight sum."""
    sm_p = spec["derivation"]["softmax"]
    p = spec["parameters"]
    dw, hd = p["data_width"], p["head_dim"]
    mw, sw_o = p["scale_width"], p["shift_width"]
    half = ((1 << dw) - 5) // 2
    rnd = random.Random(71)
    cases = []
    for n, clamp in rows:
        q = [rnd.randrange(-half, half) for _ in range(hd)]
        K = [[rnd.randrange(-half, half) for _ in range(hd)] for _ in range(n)]
        V = [[rnd.randrange(-half, half) for _ in range(hd)] for _ in range(n)]
        if n > 1:
            # One key aligned with the query, so one score dominates and
            # the softmax has something to do.
            K[1] = [half if x >= 0 else -half for x in q]
        else:
            # One position has a weight of exactly 1.0, and the most
            # negative value everywhere puts the weighted sum at the bound
            # its accumulator is sized for.
            V[0] = [-(1 << (dw - 1))] * hd
        t = [sum(q[d] * K[j][d] for d in range(hd)) for j in range(n)]
        big = max(abs(v) for v in t) << p.get("score_guard", 0)
        sh = 0
        while (big >> sh) > 1800:
            sh += 1
        if clamp:
            # Spread the row past the exponential's range, so the
            # softmax's clamp on score minus maximum has work to do.
            sh = max(0, sh - 4)
        _, s, w, a, _ = attn_golden(q, K, V, n, sh, 1, 0, p, sm_p)
        sc, so = _mlp_scale(a, dw, mw, sw_o)
        t, s, w, a, o = attn_golden(q, K, V, n, sh, sc, so, p, sm_p)
        if clamp:
            e_lo = 1 << (sm_p["score_width"] - 1)
            assert max(s) - min(s) > e_lo, "spread case did not spread"
        cases.append((n, q, K, V, sh, sc, so, s, w, o))
    body = []
    for ci, (n, q, K, V, sh, sc, so, s, w, o) in enumerate(cases):
        body.append("    // case %d, n=%d" % (ci, n))
        for d in range(hd):
            body.append("    qvec[%d] = %s;" % (d, _slit(q[d], dw)))
        for j in range(n):
            for d in range(hd):
                body.append("    kmem[%d] = %s; vmem[%d] = %s;"
                            % (j * hd + d, _slit(K[j][d], dw),
                               j * hd + d, _slit(V[j][d], dw)))
            body.append("    expect_s[%d] = %s; expect_p[%d] = %d'd%d;"
                        % (j, _slit(s[j], p["score_width"]), j,
                           p["weight_width"], w[j]))
        for d in range(hd):
            body.append("    expect_o[%d] = %s;" % (d, _slit(o[d], dw)))
        body.append("    run_head(%d, %d, %d, %d);" % (n, sh, sc, so))
    return ATTN_TB.format(
        dwm=dw - 1, hd=hd, hdwm=p["head_dim_width"] - 1,
        cap=p["capacity"], mem=p["capacity"] * hd,
        awm=p["addr_width"] - 1, nwm=p["n_width"] - 1,
        swm=p["score_width"] - 1, wwm=p["weight_width"] - 1,
        mwm=mw - 1, sowm=sw_o - 1, cases="\n".join(body),
        ncases=len(cases))


ATTN_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for a single attention head over a KV cache. Golden values
// come from the softmax and requantizer models, so the head is checked
// against the arithmetic its parts are specified to do. The scores and
// weights are checked directly, before the outputs, because they are
// upstream of them.
module tb_attn;
  reg clk = 0, rst_n = 0, start = 0, load_valid = 0;
  reg signed [{dwm}:0] load_data = 0;
  reg [{nwm}:0] n = 0;
  reg [4:0] shift_s = 0;
  reg [{mwm}:0] scale_o = 0;
  reg [{sowm}:0] shift_o = 0;
  wire [{awm}:0] k_addr, v_addr;
  wire o_valid, busy;
  wire [{hdwm}:0] o_index;
  wire signed [{dwm}:0] o_data;

  reg signed [{dwm}:0] qvec [0:{hd}-1];
  reg signed [{dwm}:0] kmem [0:{mem}-1];
  reg signed [{dwm}:0] vmem [0:{mem}-1];
  reg signed [{swm}:0] expect_s [0:{cap}-1];
  reg        [{wwm}:0] expect_p [0:{cap}-1];
  reg signed [{dwm}:0] expect_o [0:{hd}-1];
  reg signed [{dwm}:0] k_data, v_data;
  integer checks = 0, seen = 0, i, bad, nbad;
  reg [255:0] testname;
  reg [15:0] bad_idx [0:7];
  reg signed [{dwm}:0] bad_exp [0:7], bad_got [0:7];
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  always @(posedge clk) begin
    k_data <= kmem[k_addr];
    v_data <= vmem[v_addr];
  end

  attn dut (.clk(clk), .rst_n(rst_n), .load_valid(load_valid),
            .load_data(load_data), .start(start), .n(n),
            .shift_s(shift_s), .scale_o(scale_o), .shift_o(shift_o),
            .k_addr(k_addr), .k_data(k_data), .v_addr(v_addr),
            .v_data(v_data), .o_valid(o_valid), .o_index(o_index),
            .o_data(o_data), .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && o_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      if (o_index >= {hd}) begin
        $display("TB_FAIL test=%0s out=%0d head_dim={hd} expected=no_output_past_head_dim got_o=%0d",
                 testname, o_index, o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (o_data !== expect_o[o_index]) begin
        // Held: a wrong score or weight makes every output wrong, and the
        // checks after the run name the upstream cause first.
        if (nbad < 8) begin
          bad_idx[nbad] = o_index; bad_exp[nbad] = expect_o[o_index];
          bad_got[nbad] = o_data;
        end
        nbad = nbad + 1;
      end
    end
  end

  task run_head(input integer cnt, input integer sh, input integer sc,
                input integer so);
    begin
      seen = 0; nbad = 0; bad = 0;
      for (i = 0; i < {hd}; i = i + 1) begin
        @(negedge clk); load_data = qvec[i]; load_valid = 1;
      end
      @(negedge clk); load_valid = 0;
      n = cnt; shift_s = sh; scale_o = sc; shift_o = so;
      @(negedge clk); start = 1;
      t0 = cyc; first_out = -1;
      @(negedge clk); start = 0;
      while (busy) @(negedge clk);
      span = span + (cyc - t0);
      repeat (8) @(negedge clk);
      for (i = 0; i < cnt; i = i + 1) begin
        checks = checks + 1;
        if (dut.sbuf[i] !== expect_s[i]) begin
          $display("TB_FAIL test=%0s score_index=%0d expected_s=%0d got_s=%0d",
                   testname, i, expect_s[i], dut.sbuf[i]);
          bad = bad + 1;
        end
      end
      for (i = 0; i < cnt; i = i + 1) begin
        checks = checks + 1;
        if (dut.pbuf[i] !== expect_p[i]) begin
          $display("TB_FAIL test=%0s weight_index=%0d expected_p=%0d got_p=%0d",
                   testname, i, expect_p[i], dut.pbuf[i]);
          bad = bad + 1;
        end
      end
      for (i = 0; i < nbad && i < 8; i = i + 1)
        $display("TB_FAIL test=%0s out=%0d expected_o=%0d got_o=%0d",
                 testname, bad_idx[i], bad_exp[i], bad_got[i]);
      checks = checks + 1;
      if (!bad && !nbad && seen !== {hd})
        $display("TB_FAIL test=%0s out=0 expected_o_count={hd} got_o_count=%0d",
                 "output_count", seen);
      if (bad || nbad || seen !== {hd}) begin
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "attention";
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
{cases}

    $display("TB_PROFILE heads=%0d span_cycles=%0d latency_cycles=%0d",
             {ncases}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def derive_score_mac_spec(ms):
    """The MAC a head's score lanes run: q times k over head_dim, both
    16-bit activations at a16, where the model's MAC multiplies an int8
    weight by an activation over d_ff. Its accumulator follows the same
    rule for that reduction, so no score can overflow it: at Qwen3-0.6B's
    head_dim of 128 that is 39 bits, where the model's MAC has 36."""
    ab = ms["activation_bits"]
    hd = ms.get("head_dim") or ms["d_model"] // ms["n_head"]
    return derive_chiplet_spec(dict(ms, weight_bits=ab, d_model=hd, d_ff=hd,
                                    name=ms["name"] + "_qk"))


def derive_attnn_spec(ms, board=None):
    """model spec -> multi-lane attention head spec.

    With the projection widened to the DDR, the single-lane head was the
    slowest thing in a Qwen token: 14 heads by 24 layers, each reading
    1024 cached positions twice, one multiply a cycle, about 44 million
    cycles against the projections' 15. This head runs L lanes: for the
    scores each lane takes one cached position, from a key word holding
    element d of L consecutive positions; for the weighted sum each lane
    takes one output dimension, from a value word holding L elements of
    one position's row. The softmax between them stays one lane: it is
    3n cycles against the 2*n*hd it used to wait behind.
    """
    base = derive_attn_spec(ms)
    p = dict(base["parameters"])
    hd, cap, dw = p["head_dim"], p["capacity"], p["data_width"]
    La = min(derive_projn_spec(ms, board)["parameters"]["lanes"], hd)
    kwords = -(-cap // La) * hd
    vwords = cap * (hd // La)
    p.update(lanes=La, word_width=La * dw,
             k_addr_width=max(1, (kwords - 1).bit_length()),
             v_addr_width=max(1, (vwords - 1).bit_length()),
             mac_stages=derive_chiplet_spec(ms)["parameters"]
             ["pipeline_stages"],
             # The score lanes' accumulator: the model's MAC's, or the q.k
             # rule's when that is wider (16-bit activations).
             score_acc_width=max(p["acc_width"], derive_score_mac_spec(ms)
                                 ["parameters"]["acc_width"]))
    spec = dict(base)
    spec.update(name="attnn%d_%s" % (La, ms["name"]), top_module="attnn",
                description="Multi-lane attention head, %d lanes: scores "
                            "for %d cached positions at once, the weighted "
                            "sum for %d dimensions at once" % (La, La, La),
                parameters=p)
    ports = []
    for q in base["ports"]:
        q = dict(q)
        if q["name"] == "k_addr":
            q.update(width=p["k_addr_width"],
                     desc="key word: element d of positions g*%d.. at "
                          "g*%d + d" % (La, hd))
        elif q["name"] == "v_addr":
            q.update(width=p["v_addr_width"],
                     desc="value word: row j, elements dg*%d.. at "
                          "j*%d + dg" % (La, hd // La))
        elif q["name"] in ("k_data", "v_data"):
            q.update(width=La * dw, signed=False,
                     desc="lane l in bits [%d*l+%d:%d*l], registered read"
                          % (dw, dw - 1, dw))
        ports.append(q)
    spec["ports"] = ports
    # The one-lane head's prose, rewritten where lanes change it. It was
    # appended to instead, so the spec said the caches were row-major and
    # then that they were laid out by lane, and that matvec with cols = n
    # gave the key address, which is true of one lane only. Sonnet spent
    # its whole 32,000 token answer reconciling the two and never wrote a
    # line; the rules agent, which reads no prose, never noticed.
    ngr = "ceil(n / %d)" % La
    beh = []
    for b_ in base["behavior"]:
        if b_.startswith("The key and value caches are row-major"):
            b_ = ("The key and value caches are read a word at a time, %d "
                  "lanes of %d bits. Key words: word g*%d + d holds element "
                  "d of positions g*%d + l in lane l, so one word gives "
                  "dimension d of %d consecutive positions. Value words: "
                  "word j*%d + dg holds V[j][dg*%d + l] in lane l, so one "
                  "word gives %d dimensions of position j. k_data and v_data "
                  "are registered reads: each carries the word at address a "
                  "on the cycle after k_addr or v_addr = a. If the address is "
                  "itself a register, that is two clock edges after the edge "
                  "that loads a into it."
                  % (La, dw, hd, La, La, hd // La, La, La))
        elif b_.startswith("matvec, mac, softmax and requant are separate"):
            b_ = b_.replace(
                "matvec, mac, softmax and requant are separate modules "
                "supplied as source files, not something to write. "
                "Instantiate them",
                "mac, softmax and requant are separate modules supplied as "
                "source files, not something to write, and matvec is "
                "supplied too, for the score pass if you want it. "
                "Instantiate the ones you use")
            assert "Instantiate the ones you use" in b_
        elif b_.startswith("matvec walks a_addr"):
            old = ("Wire mac_valid to the MAC's valid_in and mac_clear to its "
                   "clear. With depth = %d and cols = n, matvec's w_addr is "
                   "exactly the key address." % hd)
            assert old in b_
            b_ = b_.replace(old, (
                "If it runs the score pass, it does so with depth = %d and "
                "cols = %s, the number of %d-position groups: its w_addr is "
                "then exactly the key word address g*%d + d, col_index is the "
                "group g, and mac_valid and mac_clear go to every score "
                "lane. At col_valid, score lane l holds t_j for j = g*%d + l. "
                "In the last group, lanes with j >= n are not scores: write "
                "only j < n into sbuf." % (hd, ngr, La, hd, La)))
        beh.append(b_)
    spec["behavior"] = beh
    if p["score_acc_width"] != p["acc_width"]:
        spec["behavior"].append(
            "Each score lane is an instance of mac_s (smac_dep.v), the MAC "
            "derived for q times k over head_dim, whose acc output is %d "
            "bits: the model's mac, at %d, can overflow on a score. The "
            "score path runs at %d bits up to the score quantizer. Its "
            "ports, connected by name: %s."
            % (p["score_acc_width"], p["acc_width"], p["score_acc_width"],
               port_signature(dict(derive_score_mac_spec(ms), top_module="mac_s"))))
    return spec


def render_attnn_testbench(spec, rows=None):
    """The head's cases, its golden and its direct score and weight
    checks, with wide key and value words built from the same memories."""
    p = spec["parameters"]
    La, dw, hd = p["lanes"], p["data_width"], p["head_dim"]
    tb = (render_attn_testbench(spec, rows) if rows
          else render_attn_testbench(spec))
    rep = [
        ("module tb_attn;", "module tb_attnn;"),
        ("  wire [%d:0] k_addr, v_addr;" % (p["addr_width"] - 1),
         "  wire [%d:0] k_addr;\n  wire [%d:0] v_addr;"
         % (p["k_addr_width"] - 1, p["v_addr_width"] - 1)),
        ("  reg signed [%d:0] k_data, v_data;" % (dw - 1),
         "  reg [%d:0] k_data, v_data;\n  integer wl, ki;" % (La * dw - 1)),
        ("    k_data <= kmem[k_addr];\n    v_data <= vmem[v_addr];",
         "    for (wl = 0; wl < %d; wl = wl + 1) begin\n"
         "      ki = ((k_addr / %d) * %d + wl) * %d + k_addr %% %d;\n"
         "      k_data[wl * %d +: %d] <= (ki < %d) ? kmem[ki] : 0;\n"
         "      v_data[wl * %d +: %d] <= vmem[v_addr * %d + wl];\n"
         "    end" % (La, hd, La, hd, hd, dw, dw, p["capacity"] * hd,
                      dw, dw, La)),
        ("  attn dut (", "  attnn dut ("),
    ]
    for a_, b_ in rep:
        assert a_ in tb, a_
        tb = tb.replace(a_, b_)
    return tb


def generate_attnn(ms=None, spec_file="spec_attnn.json",
                   tb_file="tb_attnn.v"):
    ms = ms or load_model_spec()
    spec = derive_attnn_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_attnn_testbench(spec))
    return spec


def generate_attn(ms=None, spec_file="spec_attn.json", tb_file="tb_attn.v"):
    """Write the derived attention head spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_attn_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_attn_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# RMSNorm: sum of squares, one inverse square root, a scaled product each.
# --------------------------------------------------------------------------

def derive_rmsnorm_spec(ms, row=None):
    """model spec -> RMSNorm spec for one d_model activation row, or for a
    row of the given length (Qwen3's q and k norms, over one head).

    Qwen normalises before attention and before the MLP. The inverse
    square root has been generated for this since it was added; this is
    the sequencer that uses it. The mean's 1/D and the quantization
    scales are folded into the runtime output scale, so the block
    computes t_i = (x_i * g_i * m) >> (e + k) with (m, e) the rsqrt of
    the sum of squares, and requantizes t_i to int8.
    """
    c = derive_chiplet_spec(ms)
    rs = derive_rsqrt_spec(ms)
    rq = derive_requant_spec(ms)
    dw, aw = c["parameters"]["data_width"], c["parameters"]["acc_width"]
    wb = ms["weight_bits"]
    D = row or ms["d_model"]
    iw, ow = rs["parameters"]["in_width"], rs["parameters"]["out_width"]
    xaw = max(1, (D - 1).bit_length())
    # |x_i| <= sqrt(ssq) and 2**e > sqrt(ssq)/2, and the gain is a
    # data_width value, so |t| < 2**(dw+ow-k): k is the extra shift that
    # brings that inside the requantizer input.
    k = max(0, dw + ow + 1 - aw)
    ew = [q["width"] for q in rs["ports"] if q["name"] == "e"][0]
    assert D * (1 << (2 * dw - 2)) < (1 << iw), "rsqrt input too narrow"
    return {
        "name": "rmsnorm_%s" % ms["name"],
        "description": "RMSNorm over a row of %d activations: sum of "
                       "squares, one inverse square root, and a scaled "
                       "product per element, requantized to int8" % D,
        "top_module": "rmsnorm",
        "unit": "norm",
        "parameters": {
            "data_width": dw, "acc_width": aw, "d_model": D,
            "addr_width": xaw, "rsqrt_in_width": iw, "rsqrt_out_width": ow,
            "rsqrt_e_width": ew, "norm_shift": k,
            "scale_width": rq["parameters"]["scale_width"],
            "shift_width": rq["parameters"]["shift_width"],
            "requant_stages": rq["parameters"]["pipeline_stages"],
            "rsqrt_stages": rs["parameters"]["pipeline_stages"],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            # The golden model needs this variant's rsqrt, not the base
            # model's: re-deriving from load_model_spec() sized the 16-bit
            # variant's 46-bit sum of squares for a 30-bit unit.
            "rsqrt": rs["parameters"],
            "rule": "row length from d_model; the rsqrt input and shift "
                    "widths from that block's ports; the extra shift k from "
                    "the bound |t| < 2**(data_width+ow-k) against the "
                    "requantizer input",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "normalise the row"},
            {"name": "eps", "dir": "input", "width": iw,
             "desc": "epsilon, in units of the sum of squares"},
            {"name": "scale_o", "dir": "input",
             "width": rq["parameters"]["scale_width"],
             "desc": "output requantizer scale"},
            {"name": "shift_o", "dir": "input",
             "width": rq["parameters"]["shift_width"],
             "desc": "output requantizer shift"},
            {"name": "x_addr", "dir": "output", "width": xaw,
             "desc": "activation address"},
            {"name": "x_data", "dir": "input", "width": dw, "signed": True,
             "desc": "activation, registered read"},
            {"name": "g_addr", "dir": "output", "width": xaw,
             "desc": "gain (gamma) address"},
            {"name": "g_data", "dir": "input", "width": dw, "signed": True,
             "desc": "gain, registered read"},
            {"name": "o_valid", "dir": "output", "width": 1,
             "desc": "a normalised element is on o_data"},
            {"name": "o_index", "dir": "output", "width": xaw,
             "desc": "which element"},
            {"name": "o_data", "dir": "output", "width": dw, "signed": True,
             "desc": "normalised element, int8"},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a row is being normalised"},
        ],
        "behavior": [
            "start is a one-cycle pulse, with eps, scale_o and shift_o "
            "held stable for the whole row. busy must be high on the clock "
            "edge that samples start, so it already reads 1 one cycle "
            "later, and it stays high until the last output is emitted.",
            "x_data and g_data are registered reads: each carries the "
            "element at address a on the cycle after x_addr or g_addr = a. "
            "If the address is itself a register, that is two clock edges "
            "after the edge that loads a into it.",
            "Pass 1: ssq = eps + sum over i in 0..%d of x_i * x_i, an "
            "unsigned %d-bit value held in a register named ssq. Start "
            "the sum from eps, not from zero." % (D - 1, iw),
            "Then (m, e) = rsqrt(ssq) from the supplied rsqrt block, held "
            "in registers named rs_m ([%d:0]) and rs_e ([%d:0])."
            % (ow - 1, ew - 1),
            "Pass 2: for i in 0..%d, t_i = (x_i * g_i * m) >>> (e + %d), "
            "an arithmetic shift of the signed product, which floors. Then "
            "o_i = requant(t_i, scale_o, shift_o): multiply by scale_o, add "
            "2**(shift_o-1) when shift_o > 0, arithmetic shift right by "
            "shift_o, saturate to [%d, %d]. |t_i| stays under 2**%d, so it "
            "fits the requantizer's %d-bit input."
            % (D - 1, k, -(1 << (dw - 1)), (1 << (dw - 1)) - 1,
               dw + ow - k, aw),
            "Emit each o_i with o_index = i and o_valid high for exactly "
            "that one cycle. Outputs may come in any order, but each index "
            "exactly once, and o_valid must be low at every other time.",
            "The testbench reads ssq, rs_m and rs_e directly to check pass "
            "1 and the inverse square root, so those names and widths are "
            "part of the interface.",
            "rsqrt and requant are separate modules supplied as source "
            "files, not something to write. Instantiate them with exactly "
            "these ports, connected by name: %s; %s. rsqrt has a latency of "
            "%d cycles and requant of exactly %d."
            % (port_signature(rs), port_signature(rq),
               rs["parameters"]["pipeline_stages"],
               rq["parameters"]["pipeline_stages"]),
            "All state resets to zero: busy and o_valid are 0 during "
            "reset.",
        ],
    }


def rmsnorm_golden(x, g, eps, scale_o, shift_o, p, rs_p):
    """Exact model of the norm, built from the rsqrt and requantizer
    models."""
    ssq = eps + sum(v * v for v in x)
    assert ssq < 1 << p["rsqrt_in_width"]
    m, e = rsqrt_golden(ssq, rs_p)
    t = [(x[i] * g[i] * m) >> (e + p["norm_shift"]) for i in range(len(x))]
    assert all(abs(v) < 1 << (p["acc_width"] - 1) for v in t), \
        "t exceeds the requantizer input"
    o = [requant_golden(v, scale_o, shift_o, p["data_width"])[0] for v in t]
    return ssq, m, e, t, o


def render_rmsnorm_testbench(spec):
    """Rows for each part: typical values, all zeros (the sum is eps
    alone, the smallest rsqrt input), one full-scale spike that puts t at
    its bound, and full-range values."""
    rs_p = spec["derivation"]["rsqrt"]
    p = spec["parameters"]
    dw, D = p["data_width"], p["d_model"]
    mw, sw_o = p["scale_width"], p["shift_width"]
    top = (1 << (dw - 1)) - 1
    rnd = random.Random(83)
    cases = []
    for kind in ("typical", "zeros", "spike", "full"):
        if kind == "typical":
            x = [rnd.randrange(-top // 2, top // 2) for _ in range(D)]
            g = [rnd.randrange(-top * 3 // 4, top * 3 // 4) for _ in range(D)]
            eps = D
        elif kind == "zeros":
            x = [0] * D
            g = [rnd.randrange(-top, top) for _ in range(D)]
            eps = 5
        elif kind == "spike":
            x = [rnd.randrange(-3, 4) for _ in range(D)]
            x[0] = -(top + 1)
            g = [rnd.randrange(-top, top) for _ in range(D)]
            g[0] = -(top + 1)
            eps = 1
        else:
            x = [rnd.randrange(-top - 1, top + 1) for _ in range(D)]
            g = [rnd.randrange(-top - 1, top + 1) for _ in range(D)]
            eps = 0
        _, _, _, t, _ = rmsnorm_golden(x, g, eps, 1, 0, p, rs_p)
        sc, so = _mlp_scale(t, dw, mw, sw_o)
        ssq, m, e, t, o = rmsnorm_golden(x, g, eps, sc, so, p, rs_p)
        cases.append((kind, x, g, eps, sc, so, ssq, m, e, o))
    body = []
    for kind, x, g, eps, sc, so, ssq, m, e, o in cases:
        body.append("    // %s" % kind)
        body.append("    testname = \"%s\";" % kind)
        for i in range(D):
            body.append("    xmem[%d] = %s; gmem[%d] = %s; expect_o[%d] = %s;"
                        % (i, _slit(x[i], dw), i, _slit(g[i], dw), i,
                           _slit(o[i], dw)))
        # A read one cycle off counts an end element twice and the other
        # never. Traced on Sonnet's RMSNorm: the sum was exactly x[0]
        # twice and x[63] not at all, and it read only "expected_ssq got
        # 6275206260" for three drafts.
        sq = [v * v for v in x]
        mask = (1 << p["rsqrt_in_width"]) - 1
        # And a sum that reads x as unsigned: traced on Haiku's RMSNorm,
        # ssq <= ssq + x * x with ssq unsigned, 19 times the sum for
        # eight drafts, each read as only "expected_ssq got_ssq".
        uns = eps + sum((v & ((1 << dw) - 1)) ** 2 for v in x)
        body.append("    run_row(%d, %d, %d, %d, %d, %d, %d, %d, %d);"
                    % (eps, sc, so, ssq, m, e, (ssq - sq[-1] + sq[0]) & mask,
                       (ssq - sq[0] + sq[-1]) & mask, uns & mask))
    return RMSNORM_TB.format(
        dwm=dw - 1, D=D, awm=p["addr_width"] - 1,
        iwm=p["rsqrt_in_width"] - 1, owm=p["rsqrt_out_width"] - 1,
        ewm=p["rsqrt_e_width"] - 1, mwm=mw - 1, sowm=sw_o - 1,
        cases="\n".join(body), ncases=len(cases))


RMSNORM_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for RMSNorm over one row. Golden values come from the rsqrt
// and requantizer models. The sum of squares and the inverse square root
// are checked directly, before the outputs, because both are upstream.
module tb_rmsnorm;
  reg clk = 0, rst_n = 0, start = 0;
  reg [{iwm}:0] eps = 0;
  reg [{mwm}:0] scale_o = 0;
  reg [{sowm}:0] shift_o = 0;
  wire [{awm}:0] x_addr, g_addr, o_index;
  wire o_valid, busy;
  wire signed [{dwm}:0] o_data;
  reg signed [{dwm}:0] xmem [0:{D}-1];
  reg signed [{dwm}:0] gmem [0:{D}-1];
  reg signed [{dwm}:0] expect_o [0:{D}-1];
  reg signed [{dwm}:0] x_data, g_data;
  integer checks = 0, seen = 0, i, bad, nbad;
  reg [255:0] testname;
  reg [15:0] bad_idx [0:7];
  reg signed [{dwm}:0] bad_exp [0:7], bad_got [0:7];
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  always @(posedge clk) begin
    x_data <= xmem[x_addr];
    g_data <= gmem[g_addr];
  end

  // What the design saw, cycle by cycle, for a failing sum: the first
  // cycles after start and the four up to the sum's last change. Traced:
  // "x0 counted twice" for five drafts running named the fault but not
  // the cycle it was on, which is what an engineer reads off a waveform.
  integer trk = 99, q;
  reg [{awm}:0] tr_a [0:7], ha [0:3], ea [0:3];
  reg signed [{dwm}:0] tr_d [0:7], hd [0:3], ed [0:3];
  reg [{iwm}:0] tr_s [0:7], hs [0:3], es [0:3], prev_ssq;
  always @(posedge clk) begin
    if (trk < 8) begin
      tr_a[trk] = x_addr; tr_d[trk] = x_data; tr_s[trk] = dut.ssq;
      trk = trk + 1;
    end
    if (busy) begin
      for (q = 0; q < 3; q = q + 1) begin
        ha[q] = ha[q + 1]; hd[q] = hd[q + 1]; hs[q] = hs[q + 1];
      end
      ha[3] = x_addr; hd[3] = x_data; hs[3] = dut.ssq;
      if (dut.ssq !== prev_ssq)
        for (q = 0; q < 4; q = q + 1) begin
          ea[q] = ha[q]; ed[q] = hd[q]; es[q] = hs[q];
        end
      prev_ssq = dut.ssq;
    end
  end

  rmsnorm dut (.clk(clk), .rst_n(rst_n), .start(start), .eps(eps),
               .scale_o(scale_o), .shift_o(shift_o), .x_addr(x_addr),
               .x_data(x_data), .g_addr(g_addr), .g_data(g_data),
               .o_valid(o_valid), .o_index(o_index), .o_data(o_data),
               .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && o_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      if (o_index >= {D}) begin
        $display("TB_FAIL test=%0s out=%0d d_model={D} expected=no_output_past_the_row got_norm=%0d",
                 testname, o_index, o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (o_data !== expect_o[o_index]) begin
        if (nbad < 8) begin
          bad_idx[nbad] = o_index; bad_exp[nbad] = expect_o[o_index];
          bad_got[nbad] = o_data;
        end
        nbad = nbad + 1;
      end
    end
  end

  task run_row(input integer ep, input integer sc, input integer so,
               input [{iwm}:0] want_ssq, input [{owm}:0] want_m,
               input [{ewm}:0] want_e, input [{iwm}:0] ssq_first_twice,
               input [{iwm}:0] ssq_last_twice, input [{iwm}:0] ssq_unsigned);
    begin
      seen = 0; nbad = 0; bad = 0;
      eps = ep; scale_o = sc; shift_o = so;
      @(negedge clk); start = 1; trk = 0;
      t0 = cyc; first_out = -1;
      @(negedge clk); start = 0;
      while (busy) @(negedge clk);
      span = span + (cyc - t0);
      repeat (8) @(negedge clk);
      checks = checks + 3;
      if (dut.ssq !== want_ssq) begin
        if (dut.ssq == ssq_first_twice && ssq_first_twice != want_ssq)
          $write("TB_FAIL test=%0s expected_ssq=%0d got_ssq=%0d got_ssq_counts_x0_twice_and_never_the_last_element=1",
                   testname, want_ssq, dut.ssq);
        else if (dut.ssq == ssq_last_twice && ssq_last_twice != want_ssq)
          $write("TB_FAIL test=%0s expected_ssq=%0d got_ssq=%0d got_ssq_counts_the_last_element_twice_and_never_x0=1",
                   testname, want_ssq, dut.ssq);
        else if (dut.ssq == ssq_unsigned && ssq_unsigned != want_ssq)
          $write("TB_FAIL test=%0s expected_ssq=%0d got_ssq=%0d got_ssq_is_the_sum_with_x_read_as_unsigned=1",
                   testname, want_ssq, dut.ssq);
        else
          $write("TB_FAIL test=%0s expected_ssq=%0d got_ssq=%0d",
                   testname, want_ssq, dut.ssq);
        $write(" cycles_after_start_as_cycle_x_addr_x_data_ssq=");
        for (i = 0; i < 8; i = i + 1)
          $write("%0d:%0d,%0d,%0d;", i, tr_a[i], tr_d[i], tr_s[i]);
        $write(" four_cycles_to_the_last_change_of_ssq_as_x_addr_x_data_ssq=");
        for (i = 0; i < 4; i = i + 1)
          $write("%0d,%0d,%0d;", ea[i], ed[i], es[i]);
        $display("");
        bad = bad + 1;
      end
      if (dut.rs_m !== want_m || dut.rs_e !== want_e) begin
        $display("TB_FAIL test=%0s expected_rsm=%0d got_rsm=%0d expected_rse=%0d got_rse=%0d",
                 testname, want_m, dut.rs_m, want_e, dut.rs_e);
        bad = bad + 1;
      end
      // An output that is its neighbour's expected value is a pipeline
      // alignment fault, not an arithmetic one; traced, Sonnet's outputs
      // were each the next index's for two drafts.
      for (i = 0; i < nbad && i < 8; i = i + 1)
        if (bad_idx[i] + 1 < {D} && bad_got[i] == expect_o[bad_idx[i] + 1])
          $display("TB_FAIL test=%0s out=%0d expected_norm=%0d got_norm=%0d got_norm_is_the_expected_value_for_out=%0d",
                   testname, bad_idx[i], bad_exp[i], bad_got[i], bad_idx[i] + 1);
        else if (bad_idx[i] > 0 && bad_got[i] == expect_o[bad_idx[i] - 1])
          $display("TB_FAIL test=%0s out=%0d expected_norm=%0d got_norm=%0d got_norm_is_the_expected_value_for_out=%0d",
                   testname, bad_idx[i], bad_exp[i], bad_got[i], bad_idx[i] - 1);
        else
          $display("TB_FAIL test=%0s out=%0d expected_norm=%0d got_norm=%0d",
                   testname, bad_idx[i], bad_exp[i], bad_got[i]);
      if (!bad && !nbad && seen !== {D})
        $display("TB_FAIL test=%0s out=0 expected_count={D} got_count=%0d",
                 testname, seen);
      if (bad || nbad || seen !== {D}) begin
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

    $display("TB_PROFILE rows=%0d span_cycles=%0d latency_cycles=%0d",
             {ncases}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_rmsnorm(ms=None, spec_file="spec_rmsnorm.json",
                     tb_file="tb_rmsnorm.v"):
    """Write the derived RMSNorm spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_rmsnorm_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_rmsnorm_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# SiLU: x * sigmoid(x), from the exponential and the reciprocal.
# --------------------------------------------------------------------------

def derive_silu_spec(ms):
    """model spec -> SiLU unit spec.

    Qwen's MLP is gated, down(SiLU(gate(x)) * up(x)), and SiLU is the one
    nonlinearity in it. It needs nothing new: with a = |x| and
    e = exp(-a), which the exponential unit handles because its argument
    is non-positive, sigmoid(x) is 1/(1+e) for x >= 0 and e/(1+e) for
    x < 0, and the reciprocal unit supplies the division.
    """
    e = derive_exp_spec(ms)
    r = derive_recip_spec(ms)
    iw, fi = e["parameters"]["in_width"], e["parameters"]["in_frac"]
    wf = e["parameters"]["out_frac"]
    return {
        "name": "silu_%s" % ms["name"],
        "description": "Streaming SiLU, x * sigmoid(x), on Q%d.%d in and out, "
                       "built from the exponential and reciprocal units"
                       % (iw - 1 - fi, fi),
        "top_module": "silu",
        "unit": "element",
        "parameters": {
            "width": iw, "frac": fi, "weight_frac": wf,
            "recip_in_width": r["parameters"]["in_width"],
            "recip_out_width": r["parameters"]["out_width"],
            "shift_bias": r["parameters"]["shift_bias"],
            "recip_k_width": [q["width"] for q in r["ports"]
                              if q["name"] == "k"][0],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "exp": e["parameters"], "recip": r["parameters"],
            "rule": "input and output format from the exponential unit's "
                    "score format; sigmoid through that unit and the "
                    "reciprocal",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "x", "dir": "input", "width": iw, "signed": True,
             "desc": "input, Q%d.%d" % (iw - 1 - fi, fi)},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "x valid"},
            {"name": "y", "dir": "output", "width": iw, "signed": True,
             "desc": "SiLU(x), Q%d.%d" % (iw - 1 - fi, fi)},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y valid"},
        ],
        "behavior": [
            "Fully pipelined: a new x may arrive on every cycle, and each "
            "valid_in produces exactly one valid_out, in the same order. "
            "The latency is fixed but is the design's choice.",
            "a = |x|, clamped to %d so that -a fits the exponential's "
            "%d-bit signed input (x = %d is the one value that needs it)."
            % ((1 << (iw - 1)) - 1, iw, -(1 << (iw - 1))),
            "e = expu(-a), a Q0.%d value from the supplied exponential "
            "unit." % wf,
            "d = 2**%d + e, then (m, k) = recip(d) from the supplied "
            "reciprocal unit, d zero-extended to its %d-bit input."
            % (wf, r["parameters"]["in_width"]),
            "num = 2**%d when x >= 0 and num = e when x < 0: sigmoid(x) is "
            "1/(1+e) for non-negative x and e/(1+e) for negative x. "
            "sig = min(2**%d, ((num << %d) * m) >> (%d - k)), a Q0.%d "
            "sigmoid." % (wf, wf, wf, r["parameters"]["shift_bias"], wf),
            "y = (x * sig) >>> %d, an arithmetic shift of the signed "
            "product, which floors." % wf,
            "expu and recip are separate modules supplied as source files, "
            "not something to write. Instantiate them with exactly these "
            "ports, connected by name: %s; %s. Each has a latency of %d "
            "cycles." % (port_signature(e), port_signature(r),
                         e["parameters"]["pipeline_stages"]),
            "All state resets to zero: valid_out is 0 during reset.",
        ],
    }


def silu_golden(x, p):
    """Exact model of the unit, from the exponential and reciprocal
    models."""
    e_p, r_p = p["exp"], p["recip"]
    iw, wf = e_p["in_width"], e_p["out_frac"]
    a = min(abs(x), (1 << (iw - 1)) - 1)
    e = exp_golden(-a, e_p)
    m, k = recip_golden((1 << wf) + e, r_p)
    num = (1 << wf) if x >= 0 else e
    sig = min(1 << wf, recip_apply(num << wf, m, k, r_p))
    return (x * sig) >> wf


def render_silu_testbench(spec):
    """Inputs across the whole range, streamed back to back with gaps, and
    checked in order off valid_out."""
    p = spec["parameters"]
    d = spec["derivation"]
    iw, fi = p["width"], p["frac"]
    lo, hi = -(1 << (iw - 1)), (1 << (iw - 1)) - 1
    rnd = random.Random(97)
    xs = [0, 1, -1, 1 << fi, -(1 << fi), 2 << fi, -(2 << fi), 5 << fi,
          -(5 << fi), lo, hi, lo + 1, -(1 << (fi - 1)), 1 << (fi - 1)]
    xs += [rnd.randrange(lo, hi + 1) for _ in range(300)]
    body = []
    for i, x in enumerate(xs):
        body.append("    xs[%d] = %s; ys[%d] = %s;"
                    % (i, _slit(x, iw), i, _slit(silu_golden(x, d), iw)))
    return SILU_TB.format(iwm=iw - 1, n=len(xs), cases="\n".join(body))


SILU_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the streaming SiLU unit. Golden values come from the
// exponential and reciprocal models. Inputs stream one per cycle with
// occasional gaps, and outputs are checked in order off valid_out, so the
// latency is the design's choice but the order and the count are not.
module tb_silu;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg signed [{iwm}:0] x = 0;
  wire signed [{iwm}:0] y;
  wire valid_out;
  reg signed [{iwm}:0] xs [0:{n}-1];
  reg signed [{iwm}:0] ys [0:{n}-1];
  integer checks = 0, got = 0, i;
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  silu dut (.clk(clk), .rst_n(rst_n), .x(x), .valid_in(valid_in),
            .y(y), .valid_out(valid_out));
  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && valid_out) begin
      checks = checks + 1;
      if (got >= {n}) begin
        $display("TB_FAIL test=silu idx=%0d expected=no_more_outputs got_silu=%0d",
                 got, y);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (y !== ys[got]) begin
        $display("TB_FAIL test=silu idx=%0d x=%0d expected_silu=%0d got_silu=%0d",
                 got, xs[got], ys[got], y);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (got == 0) lat = cyc - t0;
      got = got + 1;
      span = cyc - t0 + 1;
    end
  end

  initial begin
{cases}
    // valid_out must be low from the first cycle of reset, not merely
    // after enough idle cycles to flush the pipeline: checked only after
    // three, a shallow design with no reset at all passed.
    for (i = 0; i < 3; i = i + 1) begin
      @(negedge clk);
      checks = checks + 1;
      if (valid_out !== 1'b0) begin
        $display("TB_FAIL test=reset_init cycle=%0d expected_vout=0 got_vout=%b",
                 i, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
    rst_n = 1;
    t0 = cyc + 1;
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
      x = xs[i]; valid_in = 1;
      if (i % 37 == 36) begin
        @(negedge clk); valid_in = 0;
      end
    end
    @(negedge clk); valid_in = 0;
    repeat (40) @(negedge clk);
    checks = checks + 1;
    if (got !== {n}) begin
      $display("TB_FAIL test=silu idx=0 expected_count={n} got_count=%0d", got);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE elements=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_silu(ms=None, spec_file="spec_silu.json", tb_file="tb_silu.v"):
    """Write the derived SiLU spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_silu_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_silu_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# Rotary position embedding: the rotation Qwen applies to q and k.
# --------------------------------------------------------------------------

def derive_rope_spec(ms):
    """model spec -> rotary position embedding spec.

    Qwen encodes position by rotating q and k before the scores, not by
    adding a learned vector: each pair (x[i], x[i + d/2]) of a head turns
    by pos * theta_i with theta_i = base**(-2i/d). This is Qwen's rotate
    half pairing, so both halves of a pair share one angle.

    The angle is carried in turns, as a fixed-point fraction, so wrapping
    past a full turn is free: pos * F_i is taken modulo 2**phase_bits.
    It has pos_width + 2 bits more than the table index, so the rounding
    in F_i cannot walk the angle off by a table step anywhere in the
    context. The sine table has 2**lut_bits entries over a full turn, and
    cosine is the same table a quarter turn on. 4096 entries puts the
    angle within pi/4096 of exact, a quarter of an int8 step on the
    largest pair; the table's own values carry data_width + 6 fraction
    bits so their rounding is smaller than that again.
    """
    ab = ms["activation_bits"]
    dw = ab
    hd = ms.get("head_dim") or ms["d_model"] // ms["n_head"]
    assert hd % 2 == 0, "rotary pairs need an even head dimension"
    npair = hd // 2
    iw = max(1, (npair - 1).bit_length())
    pw = max(1, (ms["seq_len"] - 1).bit_length())
    lb = 12
    ph = lb + pw + 2
    cf = dw + 6
    base = float(ms.get("rope_theta", 10000.0))
    freqs = [int(round(base ** (-2.0 * i / hd) / (2 * math.pi) * (1 << ph)))
             for i in range(npair)]
    fw = max(f.bit_length() for f in freqs)
    return {
        "name": "rope_%s" % ms["name"],
        "description": "Streaming rotary position embedding for a %d-wide "
                       "head: one (x[i], x[i+%d]) pair per cycle, rotated "
                       "by pos * theta_i, theta base %g"
                       % (hd, npair, base),
        "top_module": "rope",
        "unit": "pair",
        "parameters": {
            "data_width": dw, "head_dim": hd, "pairs": npair,
            "index_width": iw, "pos_width": pw, "lut_bits": lb,
            "phase_bits": ph, "freq_width": fw, "coef_frac": cf,
            "coef_width": cf + 2, "rope_theta": base,
            "signed": True, "pipeline_stages": 1, "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "freqs": freqs,
            "rule": "pairs from head_dim; phase bits from the table index "
                    "plus the position width plus two guard bits; table "
                    "fraction bits from the activation width",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "x1", "dir": "input", "width": dw, "signed": True,
             "desc": "x[i]"},
            {"name": "x2", "dir": "input", "width": dw, "signed": True,
             "desc": "x[i + %d]" % npair},
            {"name": "idx", "dir": "input", "width": iw,
             "desc": "the pair index i"},
            {"name": "pos", "dir": "input", "width": pw,
             "desc": "the token's position"},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "x1, x2, idx, pos valid"},
            {"name": "y1", "dir": "output", "width": dw, "signed": True,
             "desc": "x1 cos - x2 sin"},
            {"name": "y2", "dir": "output", "width": dw, "signed": True,
             "desc": "x2 cos + x1 sin"},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y1, y2 valid"},
        ],
        "behavior": [
            "Fully pipelined: a new pair may arrive on every cycle, and each "
            "valid_in produces exactly one valid_out, in the same order. "
            "The latency is fixed but is the design's choice.",
            "phase = (pos * F[idx]) mod 2**%d, where F comes from the "
            "supplied rope_freq table: rope_freq (input [%d:0] idx, output "
            "[%d:0] val), combinational." % (ph, iw - 1, fw - 1),
            "t = ((phase + 2**%d) >> %d) mod 2**%d, the nearest table entry; "
            "s = SIN[t] and c = SIN[(t + %d) mod 2**%d], from the supplied "
            "rope_sin table: rope_sin (input [%d:0] idx, output [%d:0] "
            "val), combinational, val a %d-bit two's complement value "
            "with %d fraction bits. Instantiate it twice."
            % (ph - lb - 1, ph - lb, lb, 1 << (lb - 2), lb, lb - 1,
               cf + 1, cf + 2, cf),
            "y1 = sat((x1 * c - x2 * s + 2**%d) >>> %d) and "
            "y2 = sat((x2 * c + x1 * s + 2**%d) >>> %d): signed products, "
            "round half up, an arithmetic shift, saturate to %d bits."
            % (cf - 1, cf, cf - 1, cf, dw),
            "All state resets to zero: valid_out is 0 during reset.",
        ],
    }


def rope_sin_table(p):
    lb, cf = p["lut_bits"], p["coef_frac"]
    return [int(round(math.sin(2 * math.pi * k / (1 << lb)) * (1 << cf)))
            for k in range(1 << lb)]


def rope_golden(x1, x2, i, pos, p, freqs, dphase=0):
    """Exact model of the unit. dphase offsets the angle by that many
    counts, which is how the testbench finds cases that tell an angle one
    count off from the right one."""
    lb, ph, cf, dw = (p["lut_bits"], p["phase_bits"], p["coef_frac"],
                      p["data_width"])
    tab = rope_sin_table(p)
    phase = (pos * freqs[i] + dphase) & ((1 << ph) - 1)
    k = ((phase + (1 << (ph - lb - 1))) >> (ph - lb)) & ((1 << lb) - 1)
    s = tab[k]
    c = tab[(k + (1 << (lb - 2))) & ((1 << lb) - 1)]
    hi, lo = (1 << (dw - 1)) - 1, -(1 << (dw - 1))
    sat = lambda v: max(lo, min(hi, v))
    y1 = sat((x1 * c - x2 * s + (1 << (cf - 1))) >> cf)
    y2 = sat((x2 * c + x1 * s + (1 << (cf - 1))) >> cf)
    return y1, y2


def rope_roms(spec):
    p = spec["parameters"]
    mask = (1 << p["coef_width"]) - 1
    return (render_rom("rope_freq", spec["derivation"]["freqs"],
                       p["index_width"], p["freq_width"])
            + render_rom("rope_sin", [v & mask for v in rope_sin_table(p)],
                         p["lut_bits"], p["coef_width"]))


def _rope_cases(p, freqs, rnd):
    """Pairs that pin down the rotation, not only its magnitude."""
    dw, npair = p["data_width"], p["pairs"]
    pmax = (1 << p["pos_width"]) - 1
    lo, hi = -(1 << (dw - 1)), (1 << (dw - 1)) - 1
    cases = []
    # Position 0 is the identity: sin is 0, cos is one.
    for i in range(npair):
        cases.append((rnd.randrange(lo, hi + 1), rnd.randrange(lo, hi + 1),
                      i, 0))
    # The fastest pair over many positions: the direction of the turn.
    for pos in range(1, 40):
        cases.append((hi // 2, 0, 0, pos))
        cases.append((0, hi // 2, 0, pos))
    # The slowest pairs at the end of the context, where phase error in
    # F would have accumulated.
    for i in (npair - 1, npair - 2, npair // 2):
        cases.append((hi, lo, i, pmax))
    # Saturation: both halves at full scale, near an eighth of a turn.
    best = None
    for i in range(npair):
        for pos in range(1, pmax + 1, 7):
            y1, y2 = rope_golden(hi, hi, i, pos, p, freqs)
            if y2 == hi and (best is None or y1 > best[0]):
                best = (y1, i, pos)
    if best:
        cases.append((hi, hi, best[1], best[2]))
        cases.append((lo, lo, best[1], best[2]))
    # An angle one count off moves the table index only when pos * F_i
    # sits one count below a rounding boundary, about one position in
    # 4096, so random pairs never find it: mutation testing showed a
    # phase off by one surviving on every int8 variant. Search the grid
    # for such a point and values that make the neighbouring entry show.
    ph, lb = p["phase_bits"], p["lut_bits"]
    edge = (1 << (ph - lb - 1)) - 1
    mask = (1 << (ph - lb)) - 1
    found = 0
    for i in range(npair):
        for pos in range(pmax + 1):
            if (pos * freqs[i]) & mask != edge:
                continue
            for a, b in ((hi, hi), (hi, lo), (lo, hi), (hi, 0), (0, hi)):
                if (rope_golden(a, b, i, pos, p, freqs)
                        != rope_golden(a, b, i, pos, p, freqs, 1)):
                    cases.append((a, b, i, pos))
                    found += 1
                    break
            if found >= 3:
                break
        if found >= 3:
            break
    for _ in range(400):
        cases.append((rnd.randrange(lo, hi + 1), rnd.randrange(lo, hi + 1),
                      rnd.randrange(npair), rnd.randrange(pmax + 1)))
    return cases


def render_rope_testbench(spec):
    p = spec["parameters"]
    freqs = spec["derivation"]["freqs"]
    dw = p["data_width"]
    cases = _rope_cases(p, freqs, random.Random(131))
    body = []
    for n_, (a, b, i, pos) in enumerate(cases):
        y1, y2 = rope_golden(a, b, i, pos, p, freqs)
        body.append("    x1s[%d] = %s; x2s[%d] = %s; ids[%d] = %d; ps[%d] = %d;"
                    " y1s[%d] = %s; y2s[%d] = %s;"
                    % (n_, _slit(a, dw), n_, _slit(b, dw), n_, i, n_, pos,
                       n_, _slit(y1, dw), n_, _slit(y2, dw)))
    return ROPE_TB.format(dwm=dw - 1, iwm=p["index_width"] - 1,
                          pwm=p["pos_width"] - 1, n=len(cases),
                          cases="\n".join(body))


ROPE_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the rotary position embedding. Golden values come from
// specgen.rope_golden. Pairs stream one per cycle with occasional gaps
// and are checked in order off valid_out.
module tb_rope;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg signed [{dwm}:0] x1 = 0, x2 = 0;
  reg [{iwm}:0] idx = 0;
  reg [{pwm}:0] pos = 0;
  wire signed [{dwm}:0] y1, y2;
  wire valid_out;
  reg signed [{dwm}:0] x1s [0:{n}-1];
  reg signed [{dwm}:0] x2s [0:{n}-1];
  reg [{iwm}:0] ids [0:{n}-1];
  reg [{pwm}:0] ps [0:{n}-1];
  reg signed [{dwm}:0] y1s [0:{n}-1];
  reg signed [{dwm}:0] y2s [0:{n}-1];
  integer checks = 0, got = 0, i;
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  rope dut (.clk(clk), .rst_n(rst_n), .x1(x1), .x2(x2), .idx(idx),
            .pos(pos), .valid_in(valid_in), .y1(y1), .y2(y2),
            .valid_out(valid_out));
  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && valid_out) begin
      checks = checks + 2;
      if (got >= {n}) begin
        $display("TB_FAIL test=rope idx=%0d expected=no_more_outputs got_rope=%0d",
                 got, y1);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (y1 !== y1s[got] || y2 !== y2s[got]) begin
        $display("TB_FAIL test=rope n=%0d pair=%0d pos=%0d x1=%0d x2=%0d expected_rope=%0d,%0d got_rope=%0d,%0d",
                 got, ids[got], ps[got], x1s[got], x2s[got], y1s[got], y2s[got], y1, y2);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (got == 0) lat = cyc - t0;
      got = got + 1;
      span = cyc - t0 + 1;
    end
  end

  initial begin
{cases}
    for (i = 0; i < 3; i = i + 1) begin
      @(negedge clk);
      checks = checks + 1;
      if (valid_out !== 1'b0) begin
        $display("TB_FAIL test=reset_init cycle=%0d expected_vout=0 got_vout=%b",
                 i, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
    rst_n = 1;
    t0 = cyc + 1;
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
      x1 = x1s[i]; x2 = x2s[i]; idx = ids[i]; pos = ps[i]; valid_in = 1;
      if (i % 41 == 40) begin
        @(negedge clk); valid_in = 0;
      end
    end
    @(negedge clk); valid_in = 0;
    repeat (40) @(negedge clk);
    checks = checks + 1;
    if (got !== {n}) begin
      $display("TB_FAIL test=rope idx=0 expected_count={n} got_count=%0d", got);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE pairs=%0d span_cycles=%0d latency_cycles=%0d",
             {n}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_rope(ms=None, spec_file="spec_rope.json", tb_file="tb_rope.v"):
    """Write the derived RoPE spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_rope_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_rope_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# Gated MLP: down(SiLU(gate(x)) * up(x)), Qwen's MLP.
# --------------------------------------------------------------------------

def derive_gmlp_spec(ms):
    """model spec -> gated MLP sequencer spec.

    Qwen's MLP has three projections, not two: gate and up from the input,
    their elementwise product with SiLU on the gate, and down from that.
    This runs them through one matmul sequencer, the SiLU unit and one
    shared requantizer, over one weight memory holding the three matrices
    back to back.
    """
    c = derive_chiplet_spec(ms)
    mv = derive_matvec_spec(ms)
    rq = derive_requant_spec(ms)
    si = derive_silu_spec(ms)
    dw, aw = c["parameters"]["data_width"], c["parameters"]["acc_width"]
    bank = 64
    bw = (bank - 1).bit_length()
    gw = si["parameters"]["width"]
    addr_w = (3 * bank * bank - 1).bit_length()
    mw, shw = rq["parameters"]["scale_width"], rq["parameters"]["shift_width"]
    return {
        "name": "gmlp_%s" % ms["name"],
        "description": "Gated MLP, Qwen's: down(SiLU(gate(x)) * up(x)) over "
                       "a %d-entry activation tile, three projections from "
                       "one weight memory" % bank,
        "top_module": "gmlp",
        "unit": "layer",
        "parameters": {
            "data_width": dw, "acc_width": aw, "bank": bank,
            "bank_width": bw, "gate_width": gw,
            "gate_frac": si["parameters"]["frac"],
            "depth_width": mv["parameters"]["depth_width"],
            "col_width": mv["parameters"]["col_width"],
            "mv_addr_width": mv["parameters"]["addr_width"],
            "addr_width": addr_w, "scale_width": mw, "shift_width": shw,
            "requant_stages": rq["parameters"]["pipeline_stages"],
            "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "silu": si["derivation"],
            "rule": "tile and address widths as the MLP layer, with room "
                    "for three matrices; the gate format from the SiLU "
                    "unit; the requantizer from that block",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "load_valid", "dir": "input", "width": 1,
             "desc": "write the next input activation"},
            {"name": "load_data", "dir": "input", "width": dw,
             "signed": True, "desc": "input activation"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "run the layer"},
            {"name": "depth", "dir": "input",
             "width": mv["parameters"]["depth_width"],
             "desc": "input width, the gate and up reduction length"},
            {"name": "cols_ff", "dir": "input",
             "width": mv["parameters"]["col_width"],
             "desc": "hidden width: gate and up outputs, down reduction"},
            {"name": "cols_out", "dir": "input",
             "width": mv["parameters"]["col_width"],
             "desc": "outputs of the down projection"},
            {"name": "shift_g", "dir": "input", "width": 5,
             "desc": "gate accumulator shift into the SiLU format"},
            {"name": "scale_u", "dir": "input", "width": mw,
             "desc": "up projection requantizer scale"},
            {"name": "shift_u", "dir": "input", "width": shw,
             "desc": "up projection requantizer shift"},
            {"name": "scale_h", "dir": "input", "width": mw,
             "desc": "SiLU(gate) * up requantizer scale"},
            {"name": "shift_h", "dir": "input", "width": shw,
             "desc": "SiLU(gate) * up requantizer shift"},
            {"name": "scale_d", "dir": "input", "width": mw,
             "desc": "down projection requantizer scale"},
            {"name": "shift_d", "dir": "input", "width": shw,
             "desc": "down projection requantizer shift"},
            {"name": "w_addr", "dir": "output", "width": addr_w,
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
            "Each cycle with load_valid high writes load_data into the "
            "input buffer at the next index, starting from 0 after reset "
            "and after every run, so x[0..depth-1] are loaded in order "
            "before start.",
            "start is a one-cycle pulse, with every width, scale and shift "
            "input held stable for the whole layer. busy must be high on "
            "the clock edge that samples start, so it already reads 1 one "
            "cycle later, and it stays high until the last output has been "
            "emitted.",
            "Weights are column-major and packed back to back in one "
            "memory: the gate matrix's weight for input r of hidden unit c "
            "at c*depth + r; the up matrix's after it, at depth*cols_ff + "
            "c*depth + r; the down matrix's after both, at 2*depth*cols_ff "
            "+ c*cols_ff + r for hidden unit r of output c.",
            "w_data is a registered read: it carries the weight at address "
            "a on the cycle after w_addr = a. If w_addr is itself a "
            "register, that is two clock edges after the edge that loads a "
            "into it.",
            "Gate: for c in 0..cols_ff-1, t_c = sum over r of x[r] * "
            "Wgate[c][r]; g_c = (t_c + 2**(shift_g-1)) >>> shift_g when "
            "shift_g > 0, else t_c, clamped to [%d, %d], the SiLU unit's "
            "Q%d.%d input. gbuf[c] = silu(g_c), from the supplied SiLU unit."
            % (-(1 << (gw - 1)), (1 << (gw - 1)) - 1,
               gw - 1 - si["parameters"]["frac"], si["parameters"]["frac"]),
            "Up: for c in 0..cols_ff-1, ubuf[c] = requant(sum over r of "
            "x[r] * Wup[c][r], scale_u, shift_u).",
            "Product: for c in 0..cols_ff-1, hbuf[c] = requant(gbuf[c] * "
            "ubuf[c], scale_h, shift_h), the signed product of the Q%d.%d "
            "SiLU output and the int8 up value."
            % (gw - 1 - si["parameters"]["frac"], si["parameters"]["frac"]),
            "Output: for c in 0..cols_out-1, y[c] = requant(sum over r of "
            "hbuf[r] * Wdown[c][r], scale_d, shift_d), with reduction "
            "length cols_ff.",
            "requant is exactly the supplied requantizer: multiply by "
            "scale, add 2**(shift-1) when shift > 0, arithmetic shift "
            "right by shift, then saturate to [%d, %d]."
            % (-(1 << (dw - 1)), (1 << (dw - 1)) - 1),
            "Emit each y[c] with o_index = c and o_valid high for exactly "
            "that one cycle. Outputs may come in any order, but each index "
            "exactly once, and o_valid must be low at every other time.",
            "The buffers are memories named gbuf, ubuf and hbuf, declared "
            "reg signed [%d:0] gbuf [0:%d], reg signed [%d:0] ubuf "
            "[0:%d] and reg signed [%d:0] hbuf [0:%d], with hidden unit c "
            "at index c. The testbench reads them directly to check each "
            "stage before the outputs, so these names and layouts are part "
            "of the interface."
            % (gw - 1, bank - 1, dw - 1, bank - 1, dw - 1, bank - 1),
            "matvec, mac, silu and requant are separate modules supplied "
            "as source files, not something to write. Instantiate them "
            "with exactly these ports, connected by name: "
            + "; ".join(port_signature(x) for x in (mv, c, si, rq)) + ".",
            "matvec walks a_addr 0..depth-1 and w_addr col*depth onward "
            "for each of cols columns, drives mac_valid one cycle behind "
            "the address, pulses mac_clear between columns, and pulses "
            "col_valid with col_index when a column's sum is on the MAC's "
            "acc. Its w_addr starts from 0, so each projection adds its own "
            "base. silu streams one value per cycle and returns results in "
            "order, with a latency of its own. requant has a latency of "
            "exactly %d cycles." % rq["parameters"]["pipeline_stages"],
            "All state resets to zero: busy and o_valid are 0 during "
            "reset.",
        ],
    }


def gmlp_golden(x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, scd, shd, p):
    """Exact model of the layer, from the SiLU and requantizer models."""
    dw, gw = p["data_width"], p["gate_width"]
    d, ff, out = len(x), len(Wg), len(Wd)
    lo, hi = -(1 << (gw - 1)), (1 << (gw - 1)) - 1
    tg = [sum(x[r] * Wg[c][r] for r in range(d)) for c in range(ff)]
    g = [max(lo, min(hi, _round_shift(v, sh_g))) for v in tg]
    gs = [silu_golden(v, p["silu"]) for v in g]
    tu = [sum(x[r] * Wu[c][r] for r in range(d)) for c in range(ff)]
    u = [requant_golden(v, scu, shu, dw)[0] for v in tu]
    hp = [gs[c] * u[c] for c in range(ff)]
    h = [requant_golden(v, sch, shh, dw)[0] for v in hp]
    ty = [sum(h[r] * Wd[c][r] for r in range(ff)) for c in range(out)]
    y = [requant_golden(v, scd, shd, dw)[0] for v in ty]
    return {"tg": tg, "g": g, "gs": gs, "tu": tu, "u": u, "hp": hp,
            "h": h, "ty": ty, "y": y}


def _gmlp_case(p, gp, seed):
    """One set of inputs and weights, with every scale derived from its own
    stage's values."""
    dw, gw, mw, sw_o = (p["data_width"], p["gate_width"], p["scale_width"],
                        p["shift_width"])
    d, ff, out = 8, 6, 5
    half = ((1 << dw) - 5) // 2
    rnd = random.Random(seed)
    x = [rnd.randrange(-half, half) for _ in range(d)]
    Wg = [[rnd.randrange(-half, half) for _ in range(d)] for _ in range(ff)]
    Wu = [[rnd.randrange(-half, half) for _ in range(d)] for _ in range(ff)]
    Wd = [[rnd.randrange(-half, half) for _ in range(ff)] for _ in range(out)]
    # One column aligned per projection, so each accumulator is large by
    # construction and a truncated one shows (as in the MLP testbench).
    Wg[0] = [half if v >= 0 else -half for v in x]
    Wu[1] = [half if v >= 0 else -half for v in x]
    tg = [sum(x[r] * Wg[c][r] for r in range(d)) for c in range(ff)]
    sh_g = 0
    while max(abs(v) for v in tg) >> sh_g > (1 << (gw - 2)):
        sh_g += 1
    g0 = gmlp_golden(x, Wg, Wu, Wd, sh_g, 1, 0, 1, 0, 1, 0, gp)
    scu, shu = _mlp_scale(g0["tu"], dw, mw, sw_o)
    g1 = gmlp_golden(x, Wg, Wu, Wd, sh_g, scu, shu, 1, 0, 1, 0, gp)
    sch, shh = _mlp_scale(g1["hp"], dw, mw, sw_o)
    g2 = gmlp_golden(x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, 1, 0, gp)
    Wd[0] = [half if v >= 0 else -half for v in g2["h"]]
    g2 = gmlp_golden(x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, 1, 0, gp)
    scd, shd = _mlp_scale(g2["ty"], dw, mw, sw_o)
    G = gmlp_golden(x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, scd, shd, gp)
    return (d, ff, out, x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, scd, shd, G)


def render_gmlp_testbench(spec):
    """Small dimensions, so the simulation stays fast, with scales derived
    per stage from the stage's own values as the MLP testbench does.

    The seed is searched until one hidden product sits on a rounding
    boundary of its requantizer, so that an off-by-one in that product
    changes a hidden value. Without it mutation testing showed the
    mutant surviving: the product is shifted right about twenty bits and
    six random hidden units almost never land on the boundary.
    """
    p = spec["parameters"]
    gp = dict(p, silu=spec["derivation"]["silu"])
    dw, gw, mw, sw_o = (p["data_width"], p["gate_width"], p["scale_width"],
                        p["shift_width"])
    for trial in range(5000):
        (d, ff, out, x, Wg, Wu, Wd, sh_g, scu, shu, sch, shh, scd, shd,
         G) = _gmlp_case(p, gp, 101 + trial)
        if len(set(G["gs"])) <= 2:
            continue
        if any(requant_golden(G["hp"][c] + 1, sch, shh, dw)[0] != G["h"][c]
               for c in range(ff)):
            break
    else:
        raise AssertionError("no gated MLP case with a product on a "
                             "rounding boundary")
    wm = ([Wg[c][r] for c in range(ff) for r in range(d)]
          + [Wu[c][r] for c in range(ff) for r in range(d)]
          + [Wd[c][r] for c in range(out) for r in range(ff)])
    init = "\n".join(
        ["    acts[%d] = %s;" % (i, _slit(v, dw)) for i, v in enumerate(x)]
        + ["    wmem_tb[%d] = %s;" % (i, _slit(v, dw)) for i, v in enumerate(wm)]
        + ["    expect_g[%d] = %s; expect_u[%d] = %s; expect_h[%d] = %s;"
           % (c, _slit(G["gs"][c], gw), c, _slit(G["u"][c], dw), c,
              _slit(G["h"][c], dw)) for c in range(ff)]
        + ["    expect_y[%d] = %s;" % (c, _slit(v, dw))
           for c, v in enumerate(G["y"])])
    return GMLP_TB.format(
        dwm=dw - 1, gwm=gw - 1, mwm=mw - 1, swm=sw_o - 1,
        bwm=p["bank_width"] - 1, depwm=p["depth_width"] - 1,
        colwm=p["col_width"] - 1, addrwm=p["addr_width"] - 1,
        d=d, ff=ff, out=out, nw=len(wm), shg=sh_g, scu=scu, shu=shu,
        sch=sch, shh=shh, scd=scd, shd=shd, init=init)


GMLP_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the gated MLP, with the real matmul sequencer, MAC, SiLU
// unit and requantizer under it. Golden values come from their models.
// The SiLU, up and product buffers are checked directly, in that order,
// before the outputs, because each is upstream of the next.
module tb_gmlp;
  reg clk = 0, rst_n = 0, start = 0, load_valid = 0;
  reg signed [{dwm}:0] load_data = 0;
  reg [{depwm}:0] depth = 0;
  reg [{colwm}:0] cols_ff = 0, cols_out = 0;
  reg [4:0] shift_g = 0;
  reg [{mwm}:0] scale_u = 0, scale_h = 0, scale_d = 0;
  reg [{swm}:0] shift_u = 0, shift_h = 0, shift_d = 0;
  wire [{addrwm}:0] w_addr;
  wire o_valid, busy;
  wire [{bwm}:0] o_index;
  wire signed [{dwm}:0] o_data;

  reg signed [{dwm}:0] acts [0:{d}-1];
  reg signed [{dwm}:0] wmem_tb [0:{nw}-1];
  reg signed [{gwm}:0] expect_g [0:{ff}-1];
  reg signed [{dwm}:0] expect_u [0:{ff}-1];
  reg signed [{dwm}:0] expect_h [0:{ff}-1];
  reg signed [{dwm}:0] expect_y [0:{out}-1];
  reg signed [{dwm}:0] w_data;
  integer checks = 0, seen = 0, i, nbad = 0, bad = 0;
  reg [255:0] testname;
  reg [7:0] bad_idx [0:7];
  reg signed [{dwm}:0] bad_exp [0:7], bad_got [0:7];
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  always @(posedge clk) w_data <= wmem_tb[w_addr];

  gmlp dut (.clk(clk), .rst_n(rst_n), .load_valid(load_valid),
            .load_data(load_data), .start(start), .depth(depth),
            .cols_ff(cols_ff), .cols_out(cols_out), .shift_g(shift_g),
            .scale_u(scale_u), .shift_u(shift_u), .scale_h(scale_h),
            .shift_h(shift_h), .scale_d(scale_d), .shift_d(shift_d),
            .w_addr(w_addr), .w_data(w_data), .o_valid(o_valid),
            .o_index(o_index), .o_data(o_data), .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && o_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      if (o_index >= {out}) begin
        $display("TB_FAIL test=%0s out=%0d cols_out={out} expected=no_output_past_cols_out got_gy=%0d",
                 testname, o_index, o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (o_data !== expect_y[o_index]) begin
        if (nbad < 8) begin
          bad_idx[nbad] = o_index; bad_exp[nbad] = expect_y[o_index];
          bad_got[nbad] = o_data;
        end
        nbad = nbad + 1;
      end
    end
  end

  initial begin
{init}
    testname = "gated_layer";
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    for (i = 0; i < {d}; i = i + 1) begin
      load_data = acts[i]; load_valid = 1;
      @(negedge clk);
    end
    load_valid = 0;
    depth = {d}; cols_ff = {ff}; cols_out = {out};
    shift_g = {shg}; scale_u = {scu}; shift_u = {shu};
    scale_h = {sch}; shift_h = {shh}; scale_d = {scd}; shift_d = {shd};
    @(negedge clk);
    start = 1;
      t0 = cyc; first_out = -1;
    @(negedge clk);
    start = 0;
    while (busy) @(negedge clk);
      span = span + (cyc - t0);
    repeat (8) @(negedge clk);
    for (i = 0; i < {ff}; i = i + 1) begin
      checks = checks + 1;
      if (dut.gbuf[i] !== expect_g[i]) begin
        $display("TB_FAIL test=%0s hidden=%0d expected_gs=%0d got_gs=%0d",
                 testname, i, expect_g[i], dut.gbuf[i]);
        bad = bad + 1;
      end
    end
    for (i = 0; i < {ff}; i = i + 1) begin
      checks = checks + 1;
      if (dut.ubuf[i] !== expect_u[i]) begin
        $display("TB_FAIL test=%0s hidden=%0d expected_u=%0d got_u=%0d",
                 testname, i, expect_u[i], dut.ubuf[i]);
        bad = bad + 1;
      end
    end
    for (i = 0; i < {ff}; i = i + 1) begin
      checks = checks + 1;
      if (dut.hbuf[i] !== expect_h[i]) begin
        $display("TB_FAIL test=%0s hidden=%0d expected_gh=%0d got_gh=%0d",
                 testname, i, expect_h[i], dut.hbuf[i]);
        bad = bad + 1;
      end
    end
    for (i = 0; i < nbad && i < 8; i = i + 1)
      $display("TB_FAIL test=%0s out=%0d expected_gy=%0d got_gy=%0d",
               testname, bad_idx[i], bad_exp[i], bad_got[i]);
    checks = checks + 1;
    if (!bad && !nbad && seen !== {out})
      $display("TB_FAIL test=output_count out=0 expected_gy_count={out} got_gy_count=%0d",
               seen);
    if (bad || nbad || seen !== {out}) begin
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE layers=%0d span_cycles=%0d latency_cycles=%0d",
             1, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_gmlp(ms=None, spec_file="spec_gmlp.json", tb_file="tb_gmlp.v"):
    """Write the derived gated MLP spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_gmlp_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_gmlp_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# Residual add: two int8 tensors at different scales, into a third.
# --------------------------------------------------------------------------

def derive_resadd_spec(ms):
    """model spec -> residual add spec.

    Each layer adds its attention output back onto its input and its MLP
    output back onto that. The two operands carry different quantization
    scales, so the add is y = sat(round((a*scale_a + b*scale_b) >> shift)),
    the requantizer's arithmetic over two terms.
    """
    c = derive_chiplet_spec(ms)
    rq = derive_requant_spec(ms)
    dw = c["parameters"]["data_width"]
    mw, shw = rq["parameters"]["scale_width"], rq["parameters"]["shift_width"]
    sw = dw + mw + 2
    return {
        "name": "resadd_%s" % ms["name"],
        "description": "Streaming residual add of two %d-bit tensors at "
                       "different scales, rounded and saturated" % dw,
        "top_module": "resadd",
        "unit": "element",
        "parameters": {
            "data_width": dw, "scale_width": mw, "shift_width": shw,
            "sum_width": sw, "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "rule": "operand width from the MAC; scale and shift widths "
                    "from the requantizer; the sum is two scaled operands, "
                    "dw+mw+1 bits each, plus a carry",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "a", "dir": "input", "width": dw, "signed": True,
             "desc": "residual stream element"},
            {"name": "b", "dir": "input", "width": dw, "signed": True,
             "desc": "block output element"},
            {"name": "scale_a", "dir": "input", "width": mw,
             "desc": "scale for a"},
            {"name": "scale_b", "dir": "input", "width": mw,
             "desc": "scale for b"},
            {"name": "shift", "dir": "input", "width": shw,
             "desc": "shift after the scaled sum"},
            {"name": "valid_in", "dir": "input", "width": 1,
             "desc": "a and b valid"},
            {"name": "y", "dir": "output", "width": dw, "signed": True,
             "desc": "sum, saturated"},
            {"name": "valid_out", "dir": "output", "width": 1,
             "desc": "y valid"},
        ],
        "behavior": [
            "Fully pipelined: a new pair may arrive on every cycle, and "
            "each valid_in produces exactly one valid_out, in the same "
            "order. The latency is fixed but is the design's choice. "
            "scale_a, scale_b and shift are held stable while a stream "
            "runs.",
            "v = a * scale_a + b * scale_b. scale_a and scale_b are "
            "unsigned magnitudes: every one of their %d bits, the top one "
            "included, is magnitude, so they run up to %d. Each product "
            "has the sign of its operand, and v is a signed %d-bit value."
            % (mw, (1 << mw) - 1, sw),
            "If shift > 0, r = (v + 2**(shift-1)) >>> shift, an arithmetic "
            "shift, which rounds to nearest with ties toward plus infinity. "
            "If shift is 0, r = v with no rounding term.",
            "y = r clamped to [%d, %d]." % (-(1 << (dw - 1)),
                                            (1 << (dw - 1)) - 1),
            "All state resets to zero: valid_out is 0 during reset.",
        ],
    }


def resadd_golden(a, b, sa, sb, sh, dw):
    v = a * sa + b * sb
    r = _round_shift(v, sh)
    return max(-(1 << (dw - 1)), min((1 << (dw - 1)) - 1, r))


def render_resadd_testbench(spec):
    """Streams with gaps, checked in order off valid_out, at several held
    scales, the largest and a zero shift among them. Includes both
    saturation rails and exact rounding ties. Truncation is caught by any
    value whose dropped fraction is at least a half; the ties are there
    for the direction a tie rounds, toward plus infinity, which a design
    rounding half away from zero gets wrong only on negative ties."""
    p = spec["parameters"]
    dw, mw = p["data_width"], p["scale_width"]
    top = (1 << (dw - 1)) - 1
    rnd = random.Random(113)
    sh = 12
    sa, sb = rnd.randrange(1 << 9, 1 << 11), rnd.randrange(1 << 9, 1 << 11)
    pairs = [(0, 0), (top, top), (-top - 1, -top - 1), (top, -top - 1),
             (1, 0), (-1, 0), (0, 1), (0, -1)]
    # Rounding ties: v an odd multiple of 2**(sh-1).
    # A random pair is a tie about once in 2**sh, so the search runs far
    # enough to find six.
    found = 0
    for _ in range(400000):
        a, b = rnd.randrange(-top - 1, top + 1), rnd.randrange(-top - 1, top + 1)
        if (a * sa + b * sb) % (1 << sh) == 1 << (sh - 1):
            pairs.append((a, b))
            found += 1
            if found >= 6:
                break
    pairs += [(rnd.randrange(-top - 1, top + 1), rnd.randrange(-top - 1, top + 1))
              for _ in range(200)]
    ties = sum(1 for a, b in pairs
               if (a * sa + b * sb) % (1 << sh) == 1 << (sh - 1))
    assert ties >= 3, "no rounding ties in the residual testbench"
    n1 = len(pairs)
    # The first stream is the profiled one, and its scales sit well inside
    # the scale range. The real design's do not: a scale with its top bit
    # set is large, not negative, and a design that read it as signed
    # passed the first stream alone and got every such add wrong in the
    # decode step. So more streams follow, each at its own held scales:
    # the largest scale and one with only the top bit set, both scales at
    # the largest with the sum saturating, and a shift of zero.
    hi = (1 << mw) - 1
    extra = random.Random(114)
    edge = [(0, 0), (top, top), (-top - 1, -top - 1), (top, -top - 1),
            (-top - 1, top), (1, 0), (-1, 0), (0, 1), (0, -1)]
    streams = [(sa, sb, sh, pairs)]
    for esa, esb, esh, k in ((hi, 1 << (mw - 1), 20, 32), (hi, hi, 16, 12),
                             (1, 2, 0, 16)):
        streams.append((esa, esb, esh, edge + [
            (extra.randrange(-top - 1, top + 1), extra.randrange(-top - 1, top + 1))
            for _ in range(k)]))
    body, segs, i = [], [], 0
    for k, (ssa, ssb, ssh, ps) in enumerate(streams):
        start = i
        for a, b in ps:
            body.append("    as_[%d] = %s; bs_[%d] = %s; ys[%d] = %s;"
                        % (i, _slit(a, dw), i, _slit(b, dw), i,
                           _slit(resadd_golden(a, b, ssa, ssb, ssh, dw), dw)))
            i += 1
        if k:
            segs.append(RESADD_STREAM.format(sa=ssa, sb=ssb, sh=ssh, lo=start, hi=i))
    return RESADD_TB.format(dwm=dw - 1, mwm=mw - 1,
                            swm=p["shift_width"] - 1, n=i, n1=n1,
                            sa=sa, sb=sb, sh=sh, cases="\n".join(body),
                            streams="".join(segs))


# One more stream at its own held scales, started once the last has drained.
RESADD_STREAM = """    scale_a = {sa}; scale_b = {sb}; shift = {sh};
    repeat (3) @(negedge clk);
    for (i = {lo}; i < {hi}; i = i + 1) begin
      @(negedge clk);
      a = as_[i]; b = bs_[i]; valid_in = 1;
    end
    @(negedge clk); valid_in = 0;
    repeat (20) @(negedge clk);
"""


RESADD_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the residual add. Pairs stream one per cycle with gaps
// and are checked in order off valid_out.
module tb_resadd;
  reg clk = 0, rst_n = 0, valid_in = 0;
  reg signed [{dwm}:0] a = 0, b = 0;
  reg [{mwm}:0] scale_a = {sa}, scale_b = {sb};
  reg [{swm}:0] shift = {sh};
  wire signed [{dwm}:0] y;
  wire valid_out;
  reg signed [{dwm}:0] as_ [0:{n}-1];
  reg signed [{dwm}:0] bs_ [0:{n}-1];
  reg signed [{dwm}:0] ys [0:{n}-1];
  integer checks = 0, got = 0, i;
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  resadd dut (.clk(clk), .rst_n(rst_n), .a(a), .b(b), .scale_a(scale_a),
              .scale_b(scale_b), .shift(shift), .valid_in(valid_in),
              .y(y), .valid_out(valid_out));
  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && valid_out) begin
      checks = checks + 1;
      if (got >= {n}) begin
        $display("TB_FAIL test=resadd idx=%0d expected=no_more_outputs got_res=%0d", got, y);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (y !== ys[got]) begin
        $display("TB_FAIL test=resadd idx=%0d a=%0d b=%0d scale_a=%0d scale_b=%0d shift=%0d expected_res=%0d got_res=%0d",
                 got, as_[got], bs_[got], scale_a, scale_b, shift, ys[got], y);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (got == 0) lat = cyc - t0;
      // the profile is the first stream's alone
      if (got < {n1}) span = cyc - t0 + 1;
      got = got + 1;
    end
  end

  initial begin
{cases}
    // valid_out must be low from the first cycle of reset, not merely
    // after enough idle cycles to flush the pipeline: checked only after
    // three, a shallow design with no reset at all passed.
    for (i = 0; i < 3; i = i + 1) begin
      @(negedge clk);
      checks = checks + 1;
      if (valid_out !== 1'b0) begin
        $display("TB_FAIL test=reset_init cycle=%0d expected_vout=0 got_vout=%b",
                 i, valid_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
    rst_n = 1;
    t0 = cyc + 1;
    for (i = 0; i < {n1}; i = i + 1) begin
      @(negedge clk);
      a = as_[i]; b = bs_[i]; valid_in = 1;
      if (i % 29 == 28) begin
        @(negedge clk); valid_in = 0;
      end
    end
    @(negedge clk); valid_in = 0;
    repeat (20) @(negedge clk);
{streams}    checks = checks + 1;
    if (got !== {n}) begin
      $display("TB_FAIL test=resadd idx=0 expected_count={n} got_count=%0d", got);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE elements=%0d span_cycles=%0d latency_cycles=%0d",
             {n1}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def generate_resadd(ms=None, spec_file="spec_resadd.json",
                    tb_file="tb_resadd.v"):
    """Write the derived residual add spec and testbench."""
    ms = ms or load_model_spec()
    spec = derive_resadd_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_resadd_testbench(spec))
    return spec


# --------------------------------------------------------------------------
# Projection: y = requant(W x) at full model size, over external memories.
# --------------------------------------------------------------------------

def derive_proj_spec(ms):
    """model spec -> full-size projection spec.

    The composite layers hold their activations in 64-entry banks, and a
    Qwen layer's rows run to d_ff = 4864. Growing those banks is not the
    answer: a real accelerator keeps activations in SRAM outside the
    compute and streams them in, exactly as the weights already are here.
    This block does that. It holds no activation buffer: it reads x and W
    through registered memory ports sized from the model, so a projection
    of any size the model has runs through it.
    """
    c = derive_chiplet_spec(ms)
    mv = derive_matvec_spec(ms)
    rq = derive_requant_spec(ms)
    dw, aw = c["parameters"]["data_width"], c["parameters"]["acc_width"]
    dep_w, col_w = mv["parameters"]["depth_width"], mv["parameters"]["col_width"]
    mva = mv["parameters"]["addr_width"]
    mw, shw = rq["parameters"]["scale_width"], rq["parameters"]["shift_width"]
    top = max(ms["d_model"], ms["d_ff"])
    assert top < 1 << dep_w and top * top < 1 << mva, "projection too large"
    return {
        "name": "proj_%s" % ms["name"],
        "description": "Full-size projection y = requant(W x) over external "
                       "activation and weight memories, up to %d by %d"
                       % (top, top),
        "top_module": "proj",
        "unit": "projection",
        "parameters": {
            "data_width": dw, "acc_width": aw, "depth_width": dep_w,
            "col_width": col_w, "addr_width": mva, "scale_width": mw,
            "shift_width": shw,
            # The accumulator is sized for weights of this many bits on
            # the data_width port, as the model's int8 weights arrive.
            "weight_bits": min(ms["weight_bits"], dw),
            "requant_stages": rq["parameters"]["pipeline_stages"],
            "max_dim": top, "signed": True, "pipeline_stages": 1,
            "target_clock_mhz": 100,
        },
        "derivation": {
            "model": ms["name"],
            "rule": "depth, column and address widths from the matmul "
                    "sequencer, which is sized from the model's largest "
                    "dimension; the requantizer from that block",
        },
        "ports": [
            {"name": "clk", "dir": "input", "width": 1,
             "desc": "clock, rising edge"},
            {"name": "rst_n", "dir": "input", "width": 1,
             "desc": "active-low synchronous reset"},
            {"name": "start", "dir": "input", "width": 1,
             "desc": "run one projection"},
            {"name": "depth", "dir": "input", "width": dep_w,
             "desc": "input length, the reduction depth"},
            {"name": "cols", "dir": "input", "width": col_w,
             "desc": "outputs"},
            {"name": "scale", "dir": "input", "width": mw,
             "desc": "requantizer scale"},
            {"name": "shift", "dir": "input", "width": shw,
             "desc": "requantizer shift"},
            {"name": "a_addr", "dir": "output", "width": dep_w,
             "desc": "activation address"},
            {"name": "a_data", "dir": "input", "width": dw, "signed": True,
             "desc": "activation, registered read"},
            {"name": "w_addr", "dir": "output", "width": mva,
             "desc": "weight address"},
            {"name": "w_data", "dir": "input", "width": dw, "signed": True,
             "desc": "weight, registered read"},
            {"name": "o_valid", "dir": "output", "width": 1,
             "desc": "an output is on o_data"},
            {"name": "o_index", "dir": "output", "width": col_w,
             "desc": "which output"},
            {"name": "o_data", "dir": "output", "width": dw, "signed": True,
             "desc": "output activation"},
            {"name": "busy", "dir": "output", "width": 1,
             "desc": "a projection is in progress"},
        ],
        "behavior": [
            "start is a one-cycle pulse, with depth, cols, scale and shift "
            "held stable for the whole projection. busy must be high on the "
            "clock edge that samples start, so it already reads 1 one cycle "
            "later, and it stays high until the last output is emitted.",
            "Activations and weights live outside the block. a_data and "
            "w_data are registered reads: each carries the element at "
            "address a on the cycle after a_addr or w_addr = a. Weights are "
            "column-major: W[c][r] is at c*depth + r.",
            "y[c] = requant(sum over r in 0..depth-1 of x[r] * W[c][r], "
            "scale, shift) for c in 0..cols-1. requant is exactly the "
            "supplied requantizer: multiply by scale, add 2**(shift-1) when "
            "shift > 0, arithmetic shift right by shift, saturate to [%d, %d]."
            % (-(1 << (dw - 1)), (1 << (dw - 1)) - 1),
            "Emit each y[c] with o_index = c and o_valid high for exactly "
            "that one cycle. Outputs may come in any order, but each index "
            "exactly once, and o_valid must be low at every other time.",
            "matvec, mac and requant are separate modules supplied as source "
            "files, not something to write. Instantiate them with exactly "
            "these ports, connected by name: "
            + "; ".join(port_signature(x) for x in (mv, c, rq)) + ".",
            "matvec walks a_addr 0..depth-1 and w_addr c*depth onward for "
            "each of cols columns, drives mac_valid one cycle behind the "
            "address, pulses mac_clear between columns, and pulses "
            "col_valid with col_index when a column's sum is on the MAC's "
            "acc: its a_addr and w_addr are exactly this block's. requant "
            "has a latency of exactly %d cycles."
            % rq["parameters"]["pipeline_stages"],
            "All state resets to zero: busy and o_valid are 0 during reset.",
        ],
    }


_HMASK = (1 << 32) - 1


def proj_x(i, seed, dw):
    """The activation at address i, a hash the testbench computes the same
    way, so a full-size test needs no initializer per element."""
    v = ((i * 2246822519 + seed * 97 + 7) & _HMASK) >> 16
    v &= (1 << dw) - 1
    return v - (1 << dw) if v >> (dw - 1) else v


def proj_w(i, seed, dw, wb=None):
    """A weight of wb bits (dw when not given), as the testbench's hw()
    makes it: past the accumulator's derivation, a wider one could
    overflow a sum no model's weights can reach."""
    wb = wb or dw
    v = ((i * 2654435761 + seed * 131 + 12345) & _HMASK) >> 13
    v &= (1 << wb) - 1
    return v - (1 << wb) if v >> (wb - 1) else v


def proj_golden(depth, cols, seed, dw, scale, shift, wb=None):
    x = [proj_x(r, seed, dw) for r in range(depth)]
    acc = [sum(x[r] * proj_w(c * depth + r, seed, dw, wb) for r in range(depth))
           for c in range(cols)]
    return acc, [requant_golden(a, scale, shift, dw)[0] for a in acc]


def render_proj_testbench(spec, cases=None, colfn=None):
    """Cases are (depth, cols, seed). The default is small, for the flow;
    tests.py renders one at the model's full size.

    The depth-3 case is there because a column takes depth + 3 cycles, and
    only when that is shorter than the requantizer's latency does an output
    index taken at the wrong moment land under a different column. Without
    it a design with exactly that bug passed every case.

    The last case reaches past half the weight address width. The small
    ones stop at address 4000, so a design that dropped the address's top
    bits passed them: mutation testing found it on the int4 variant, where
    the 26-bit address is the widest vector and halving it went unseen.
    """
    p = spec["parameters"]
    if cases is None:
        dwide = min(p["max_dim"], 128)
        cwide = min(p["max_dim"], (1 << (p["addr_width"] // 2 + 1)) // dwide + 1)
        cases = ((16, 8, 1), (100, 40, 2), (7, 130, 3), (3, 60, 4),
                 (dwide, cwide, 5))
    dw, mw, sw_o = p["data_width"], p["scale_width"], p["shift_width"]
    # Weights span the width the accumulator was derived for. At 16-bit
    # activations they used to span all 16 bits: a 100-deep sum then
    # overflowed a 256-wide model's 32-bit accumulator, which int8
    # weights cannot, and the testbench failed a correct design.
    wb = p.get("weight_bits", dw)
    hwx = ("h[13 + %d - 1:13]" % dw if wb == dw else
           "{{%d{h[%d]}}, h[%d:13]}" % (dw - wb, 12 + wb, 12 + wb))
    body, maxc = [], max(c for _, c, _ in cases)
    for depth, cols, seed in cases:
        acc, _ = proj_golden(depth, cols, seed, dw, 1, 0, wb)
        sc, sh = _mlp_scale(acc, dw, mw, sw_o)
        _, y = proj_golden(depth, cols, seed, dw, sc, sh, wb)
        body.append("    // depth %d, cols %d" % (depth, cols))
        if colfn:
            # Each column its own bias, scale and shift, from a small
            # memory, as a model with per-channel weight scales needs.
            y, words = [], list(colfn(acc, seed))
            for c, (b, csc, csh) in enumerate(words):
                y.append(requant_golden(acc[c] + b, csc, csh, dw)[0])
                # This column's sum with the previous column's word: a
                # column word read one edge early. Traced on the end-to-end
                # run, where Haiku and Sonnet both captured c_data the edge
                # after loading c_addr, and column 0 came out right only
                # because c_addr already held 0.
                pb, psc, psh = words[c - 1] if c else (b, csc, csh)
                body.append("    expect_p[%d] = %s;" % (c, _slit(
                    requant_golden(acc[c] + pb, psc, psh, dw)[0], dw)))
                # The sum without its last row: the rows fed to the MACs
                # one cycle early, as Haiku's projection did for eight
                # drafts of the end-to-end run, 12414 for 12375.
                last = proj_x(depth - 1, seed, dw) * proj_w(
                    c * depth + depth - 1, seed, dw, wb)
                body.append("    expect_l[%d] = %s;" % (c, _slit(
                    requant_golden(acc[c] - last + b, csc, csh, dw)[0], dw)))
                word = (((b & ((1 << p["acc_width"]) - 1)) << (sw_o + mw))
                        | (csh << mw) | csc)
                body.append("    cmem[%d] = %d'h%x;"
                            % (c, p["acc_width"] + sw_o + mw, word))
        for c, v in enumerate(y):
            body.append("    expect_y[%d] = %s;" % (c, _slit(v, dw)))
        body.append("    run_proj(%d, %d, %d, %d, %d);"
                    % (depth, cols, seed, sc, sh))
    return PROJ_TB.format(
        dwm=dw - 1, depwm=p["depth_width"] - 1, colwm=p["col_width"] - 1,
        addrwm=p["addr_width"] - 1, mwm=mw - 1, swm=sw_o - 1, maxc=maxc,
        dw=dw, cases="\n".join(body), ncases=len(cases), hwx=hwx)


PROJ_TB = """`timescale 1ns/1ps
// GENERATED by specgen.py: do not edit by hand.
// Testbench for the full-size projection. Activations and weights are a
// hash of their address, computed here and in specgen.proj_x/proj_w the
// same way, so a projection of any size needs no memory initializer.
module tb_proj;
  reg clk = 0, rst_n = 0, start = 0;
  reg [{depwm}:0] depth = 0;
  reg [{colwm}:0] cols = 0;
  reg [{mwm}:0] scale = 0;
  reg [{swm}:0] shift = 0;
  reg [31:0] seed = 0;
  wire [{depwm}:0] a_addr;
  wire [{addrwm}:0] w_addr;
  wire o_valid, busy;
  wire [{colwm}:0] o_index;
  wire signed [{dwm}:0] o_data;
  reg signed [{dwm}:0] a_data, w_data;
  reg signed [{dwm}:0] expect_y [0:{maxc}-1];
  reg [{maxc}-1:0] seen_bits;
  integer checks = 0, seen = 0, nbad = 0;
  reg [255:0] testname;
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;

  function signed [{dwm}:0] hx(input [31:0] i, input [31:0] s);
    reg [31:0] h;
    begin
      h = i * 32'd2246822519 + s * 32'd97 + 32'd7;
      hx = h[16 + {dw} - 1:16];
    end
  endfunction
  function signed [{dwm}:0] hw(input [31:0] i, input [31:0] s);
    reg [31:0] h;
    begin
      h = i * 32'd2654435761 + s * 32'd131 + 32'd12345;
      hw = {hwx};
    end
  endfunction

  always @(posedge clk) begin
    a_data <= hx(a_addr, seed);
    w_data <= hw(w_addr, seed);
  end

  proj dut (.clk(clk), .rst_n(rst_n), .start(start), .depth(depth),
            .cols(cols), .scale(scale), .shift(shift), .a_addr(a_addr),
            .a_data(a_data), .w_addr(w_addr), .w_data(w_data),
            .o_valid(o_valid), .o_index(o_index), .o_data(o_data),
            .busy(busy));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    if (rst_n && o_valid) begin
      checks = checks + 1;
      seen = seen + 1;
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      if (o_index >= cols || seen_bits[o_index]) begin
        $display("TB_FAIL test=%0s out=%0d cols=%0d expected=each_index_once got_proj=%0d",
                 testname, o_index, cols, o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      seen_bits[o_index] = 1'b1;
      if (o_data !== expect_y[o_index] && nbad < 4) begin
        $display("TB_FAIL test=%0s out=%0d expected_proj=%0d got_proj=%0d",
                 testname, o_index, expect_y[o_index], o_data);
        nbad = nbad + 1;
      end
    end
  end

  task run_proj(input integer d, input integer c, input integer s,
                input integer sc, input integer sh);
    begin
      seen = 0; nbad = 0; seen_bits = 0;
      depth = d; cols = c; seed = s; scale = sc; shift = sh;
      @(negedge clk); start = 1;
      t0 = cyc; first_out = -1;
      @(negedge clk); start = 0;
      while (busy) @(negedge clk);
      span = span + (cyc - t0);
      repeat (8) @(negedge clk);
      checks = checks + 1;
      if (!nbad && seen !== c)
        $display("TB_FAIL test=%0s out=0 expected_count=%0d got_count=%0d",
                 testname, c, seen);
      if (nbad || seen !== c) begin
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "projection";
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
{cases}

    $display("TB_PROFILE projections=%0d span_cycles=%0d latency_cycles=%0d",
             {ncases}, span, lat);
    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
"""


def derive_projn_spec(ms, board=None, per_column=False):
    """model spec -> multi-lane projection spec.

    Batch-1 decode reads every weight once per token, so throughput is
    set by how many weight bytes arrive per cycle, not by the arithmetic.
    The single-lane projection does one multiply-accumulate a cycle: at
    Qwen2.5-0.5B's 494 million per token that is under a quarter of a
    token a second, far below what the board's DDR could feed. This block
    has as many lanes as it takes to consume the board's sustained DDR
    bandwidth at the fabric clock, rounded up to a power of two, and each
    lane takes its own byte of one wide weight word: N columns computed
    at once against a shared activation.
    """
    import boards
    b = boards.BOARDS[board or ms.get("board", "zybo_z7_20")]
    base = derive_proj_spec(ms)
    p = dict(base["parameters"])
    clock = p["target_clock_mhz"]
    need = b["mem_gbytes_per_s"] * 1e3 / clock / (p["data_width"] / 8.0)
    lanes = 1
    while lanes < need:
        lanes *= 2
    # A build may ask for a width of its own: the rule above counts the
    # activation's bytes, where the DDR carries int8 weights, so on a board
    # with DSPs to spare twice the lanes still fit its bandwidth.
    lanes = ms.get("lanes") or lanes
    top = p["max_dim"]
    words = -(-top // lanes) * top
    p.update(per_column=bool(per_column),
             col_word_width=p["acc_width"] + p["shift_width"]
             + p["scale_width"])
    p.update(lanes=lanes, word_width=lanes * p["data_width"],
             word_addr_width=max(1, (words - 1).bit_length()),
             mac_stages=derive_chiplet_spec(ms)["parameters"]
             ["pipeline_stages"])
    spec = dict(base)
    spec.update(
        name="projn%d%s_%s" % (lanes, "c" if per_column else "",
                                ms["name"]),
        description="Multi-lane projection y = requant(W x): %d columns at "
                    "once, each lane taking its own byte of a %d-bit weight "
                    "word, sized to consume %s's sustained DDR bandwidth"
                    % (lanes, lanes * p["data_width"], b["name"]),
        top_module="projn", parameters=p)
    spec["derivation"] = dict(base["derivation"], board=b["name"],
                              mem_gbytes_per_s=b["mem_gbytes_per_s"],
                              lanes_rule="smallest power of two with lanes "
                              "* clock * data bytes >= sustained DDR "
                              "bandwidth")
    ports = []
    for q in base["ports"]:
        q = dict(q)
        if q["name"] == "w_addr":
            q.update(width=p["word_addr_width"],
                     desc="weight word address: group g, row r at "
                          "g * depth + r")
        elif q["name"] == "w_data":
            q.update(width=p["word_width"], signed=False,
                     desc="lane j's weight in bits [%d*j+%d:%d*j], "
                          "registered read" % (p["data_width"],
                                               p["data_width"] - 1,
                                               p["data_width"]))
        ports.append(q)
    if per_column:
        ports.insert(-4, {"name": "c_addr", "dir": "output",
                          "width": p["col_width"],
                          "desc": "column parameter address"})
        ports.insert(-4, {"name": "c_data", "dir": "input",
                          "width": p["col_word_width"],
                          "desc": "{bias, shift, scale} for column c_addr, "
                                  "registered read"})
    spec["ports"] = ports
    spec["behavior"] = [
        "Columns are processed in groups of %d. Group g covers columns "
        "g*%d .. g*%d+%d; in the last group, lanes past cols produce "
        "nothing." % (lanes, lanes, lanes, lanes - 1),
        "For each group, rows r = 0..depth-1 are read: a_addr = r and "
        "w_addr = g*depth + r, both registered reads whose data arrives "
        "the cycle after the address; with the address a register, that is "
        "two clock edges after the edge that loads r into it. Every row "
        "r = 0..depth-1 is summed exactly once. Lane j of the word is the "
        "weight of column g*%d + j at row r." % lanes,
        "Each lane accumulates a[r] * w_j over the rows in its own MAC, "
        "the supplied mac module, one instance per lane, cleared between "
        "groups.",
        "Each column's sum is requantized by the supplied requant module "
        "with scale and shift, and emitted with o_valid high for one cycle "
        "and o_index its column. Each column exactly once, in order.",
        "busy is high from start until the last output has been emitted.",
        "All state resets to zero.",
    ]
    if per_column:
        # Traced, run 6: the spec had the c_data port and nothing on what
        # was in it or how it was used, and Haiku and Sonnet, guessing,
        # both gave 14918 for a column whose answer is 12375.
        aw, mw_, sw_ = p["acc_width"], p["scale_width"], p["shift_width"]
        spec["behavior"][3] = (
            "Each column's sum acc is requantized by the supplied requant "
            "module, as requant(acc + bias) with that column's own scale "
            "and shift, and emitted with o_valid high for one cycle and "
            "o_index its column. Each column exactly once, in order.")
        spec["behavior"].append(
            "Each column's bias, scale and shift are in one word at c_addr = "
            "its column index, a registered read like the others: c_data "
            "carries the word for c_addr = c on the cycle after, and with "
            "c_addr a register that is two clock edges after the edge that "
            "loads c into it. c_data = "
            "{bias, shift, scale}, with bias the top %d bits, a signed value "
            "at the accumulator's scale, shift the next %d bits and scale the "
            "low %d bits. The bias is added to the column's sum before the "
            "requantizer, and the requantizer's scale and shift inputs take "
            "that column's scale and shift from the word. This block's own "
            "scale and shift ports are not used; tied to 0, the "
            "requantizer's would make every output 0."
            % (aw, sw_, mw_))
    # Traced, run 5: told only "the supplied mac module", Haiku and Sonnet
    # guessed its ports (enable, p, result, o, q; acc, sum, in for the
    # requantizer) in all thirteen drafts, and none compiled.
    mac_, rq_ = derive_chiplet_spec(ms), derive_requant_spec(ms)
    spec["behavior"].append(
        "mac and requant are separate modules supplied as source files, "
        "not something to write. Instantiate them with exactly these ports, "
        "connected by name: %s; %s. mac's valid_out follows its valid_in by "
        "%d cycles, with acc then holding the running sum, and clear zeroes "
        "it; requant has a latency of exactly %d cycles."
        % (port_signature(mac_), port_signature(rq_),
           mac_["parameters"]["pipeline_stages"],
           rq_["parameters"]["pipeline_stages"]))
    return spec


def _colparams(p):
    """Per-column bias, scale and shift for the testbench: every column
    different, most outputs in range, a few saturated."""
    dw, mw, sw_o = p["data_width"], p["scale_width"], p["shift_width"]

    def fn(acc, seed):
        rnd = random.Random(1000 + seed)
        m = max(1, max(abs(a) for a in acc))
        sc, sh = _mlp_scale(acc, dw, mw, sw_o)
        out = []
        for _ in acc:
            b = rnd.randrange(-m // 6, m // 6 + 1)
            csh = min((1 << sw_o) - 1, sh + rnd.randrange(0, 2))
            csc = max(1, rnd.randrange(sc // 2, max(sc // 2 + 1, sc * 5 // 6)))
            out.append((b, csc, csh))
        return out
    return fn


FAIL_LANE = (
    '        $display("TB_FAIL test=%0s out=%0d expected_lane=%0d got_lane=%0d",\n'
    '                 testname, o_index, expect_y[o_index], o_data);')
FAIL_LANE_PREV = (
    '        if (o_data === expect_p[o_index] && expect_p[o_index] !== expect_y[o_index])\n'
    '          $display("TB_FAIL test=%0s out=%0d expected_lane=%0d got_lane=%0d '
    'got_lane_is_this_columns_sum_with_the_previous_columns_word=1",\n'
    '                   testname, o_index, expect_y[o_index], o_data);\n'
    '        else if (o_data === expect_l[o_index] && expect_l[o_index] !== expect_y[o_index])\n'
    '          $display("TB_FAIL test=%0s out=%0d expected_lane=%0d got_lane=%0d '
    'got_lane_is_this_columns_sum_without_its_last_row=1",\n'
    '                   testname, o_index, expect_y[o_index], o_data);\n'
    '        else\n'
    '          $display("TB_FAIL test=%0s out=%0d expected_lane=%0d got_lane=%0d",\n'
    '                   testname, o_index, expect_y[o_index], o_data);')


def render_projn_testbench(spec, cases=None):
    """The projection's cases and golden, with a wide weight word: lane j
    of word g*depth + r is column g*lanes + j's weight at row r."""
    p = spec["parameters"]
    if cases is None:
        dwide = min(p["max_dim"], 128)
        cwide = min(p["max_dim"], (1 << (p["addr_width"] // 2 + 1)) // dwide + 1)
        cases = ((16, 8, 1), (100, 40, 2), (7, 130, 3), (3, 60, 4),
                 (dwide, cwide, 5))
    pc = p.get("per_column")
    tb = render_proj_testbench(spec, cases, _colparams(p) if pc else None)
    maxc = max(c for _, c, _ in cases)
    N, dw = p["lanes"], p["data_width"]
    rep = [
        ("module tb_proj;", "module tb_projn;"),
        ("  wire [%d:0] w_addr;" % (p["addr_width"] - 1),
         "  wire [%d:0] w_addr;" % (p["word_addr_width"] - 1)),
        ("  reg signed [%d:0] a_data, w_data;" % (dw - 1),
         "  reg signed [%d:0] a_data;\n  reg [%d:0] w_data;\n"
         "  integer wj, wc, wr;" % (dw - 1, N * dw - 1)),
        ("    w_data <= hw(w_addr, seed);",
         "    for (wj = 0; wj < %d; wj = wj + 1) begin\n"
         "      wc = (w_addr / depth) * %d + wj;\n"
         "      wr = w_addr %% depth;\n"
         "      w_data[wj * %d +: %d] <= (wc < cols) ? hw(wc * depth + wr, seed)"
         " : 0;\n    end" % (N, N, dw, dw)),
        ("  proj dut (", "  projn dut ("),
        ("expected_proj", "expected_lane"),
        ("got_proj", "got_lane"),
    ]
    if pc:
        cw = p["col_word_width"]
        rep += [
            ("  integer wj, wc, wr;",
             "  integer wj, wc, wr;\n  wire [%d:0] c_addr;\n"
             "  reg [%d:0] cmem [0:%d];\n  reg [%d:0] c_data;\n"
             "  always @(posedge clk) c_data <= cmem[c_addr];"
             % (p["col_width"] - 1, cw - 1,
                max(c for _, c, _ in (cases or ((1, 260, 0),))) - 1
                if cases else (1 << p["col_width"]) - 1, cw - 1)),
            (".w_data(w_data),", ".w_data(w_data), .c_addr(c_addr), .c_data(c_data),"),
            ("  reg [%d-1:0] seen_bits;" % maxc,
             "  reg [%d-1:0] seen_bits;\n  reg signed [%d:0] expect_p [0:%d];\n"
             "  reg signed [%d:0] expect_l [0:%d];"
             % (maxc, dw - 1, maxc - 1, dw - 1, maxc - 1)),
            (FAIL_LANE, FAIL_LANE_PREV),
        ]
    for a_, b_ in rep:
        assert a_ in tb, a_
        tb = tb.replace(a_, b_)
    return tb


def generate_projn(ms=None, spec_file="spec_projn.json",
                   tb_file="tb_projn.v"):
    ms = ms or load_model_spec()
    spec = derive_projn_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_projn_testbench(spec))
    return spec


def generate_proj(ms=None, spec_file="spec_proj.json", tb_file="tb_proj.v"):
    """Write the derived projection spec and its small testbench."""
    ms = ms or load_model_spec()
    spec = derive_proj_spec(ms)
    with open(os.path.join(ROOT, spec_file), "w") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(ROOT, tb_file), "w") as f:
        f.write(render_proj_testbench(spec))
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
            # Addressed from what this block can reach, not from the
            # model's whole matrix. Both matmuls read a bank, so the most
            # weights a layer can touch is two full bank by bank
            # matrices. Taking the sequencer's width here gave a port
            # whose top half nothing could ever drive, which is dead
            # silicon and, worse, unverifiable: mutation testing halved
            # that register and no vector could tell.
            "addr_width": (2 * bank * bank - 1).bit_length(),
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
            # The port the writer reads, sized like the parameter above.
            # Fixing only the parameter left a 26-bit port in the spec
            # against a 13-bit wire in the testbench.
            {"name": "w_addr", "dir": "output",
             "width": (2 * bank * bank - 1).bit_length(),
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
            "Both banks are one memory named act, declared reg signed "
            "[%d:0] act [0:%d]. Bank zero is act[0..%d] and bank one is "
            "act[%d..%d]. The testbench reads act directly to check the "
            "hidden layer, so this name and layout are part of the "
            "interface." % (dw - 1, 2 * bank - 1, bank - 1, bank,
                            2 * bank - 1),
            "Each cycle with load_valid high writes load_data into bank "
            "zero at the next index, starting from 0 after reset, so "
            "x[0..depth1-1] are loaded in order before start.",
            "start is a one-cycle pulse, with depth1, cols1, cols2, "
            "scale1, shift1, scale2 and shift2 held stable for the whole "
            "layer. busy must be high on the clock edge that samples "
            "start, so it already reads 1 one cycle later, and it stays "
            "high until the last output has been emitted.",
            "Weights are column-major and packed back to back in one "
            "memory. The first matrix's weight for input r of hidden "
            "unit c is at address c*depth1 + r. The second matrix "
            "follows it: its weight for hidden unit r of output c is at "
            "depth1*cols1 + c*cols1 + r.",
            "w_data is a registered read: it carries the weight at "
            "address a on the cycle after w_addr = a. If w_addr is itself "
            "a register, that is two clock edges after the edge that loads "
            "a into it.",
            "Hidden layer: h[c] = max(0, requant(sum over r of x[r] * "
            "W1[c][r], scale1, shift1)) for c in 0..cols1-1, written to "
            "act[%d + c]. Requantize first, then rectify." % bank,
            "Output: y[c] = requant(sum over r of h[r] * W2[c][r], "
            "scale2, shift2) for c in 0..cols2-1, with no rectifier. The "
            "second matmul's reduction length is cols1, because that is "
            "what the first produced; there is no separate input for it, "
            "so the two cannot disagree.",
            "requant is exactly the supplied requantizer: multiply by "
            "scale, add 2**(shift-1) when shift > 0, arithmetic shift "
            "right by shift, then saturate to [%d, %d]."
            % (-(1 << (dw - 1)), (1 << (dw - 1)) - 1),
            "Emit each y[c] with o_index = c and o_valid high for exactly "
            "that one cycle. Outputs may come in any order, but each "
            "index exactly once, and o_valid must be low at every other "
            "time.",
            "matvec, mac and requant are separate modules supplied as "
            "source files, not something to write. Instantiate them. "
            "matvec sequences one matrix: after a start pulse with depth "
            "and cols it walks a_addr 0..depth-1 and w_addr col*depth "
            "onward for each column, drives mac_valid one cycle behind "
            "the address, pulses col_valid with col_index when a column's "
            "sum is ready in the MAC, and pulses mac_clear after it. Its "
            "w_addr starts from 0, so the second matrix needs a base "
            "offset of depth1*cols1 added. mac has ports clk, rst_n, "
            "clear, a, b, valid_in, acc ([%d:0] signed) and valid_out. "
            "requant has ports clk, rst_n, acc_in, scale, shift, "
            "valid_in, q_out, sat and valid_out, with a latency of "
            "exactly %d cycles."
            % (aw - 1, rq["parameters"]["pipeline_stages"]),
            "All state resets to zero: busy and o_valid are 0 during "
            "reset.",
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
    # One column per layer has its weights aligned in sign with its
    # inputs, so that accumulator is large by construction. Without it
    # nothing reached the top half of the accumulator, and a design that
    # kept only the low half of it passed: an earlier version killed that
    # mutant through one value that happened to cross the line.
    for r in range(d1):
        w1[r] = half if acts[r] >= 0 else -half
    for r in range(c1):
        w2[r] = half                  # hidden values are rectified, >= 0
    # Each layer's scale is derived from its own accumulators. A fixed
    # scale used to put every output on a rail, and a layer whose outputs
    # all clamp only checks the sign of the arithmetic: the hidden layer
    # came out as two values pinned at the maximum and the rest zero.
    h_acc = [sum(acts[r] * w1[c * d1 + r] for r in range(d1))
             for c in range(c1)]
    sc1, sh1 = _mlp_scale(h_acc, dw, mw, sw)
    h = [max(0, requant_golden(v, sc1, sh1, dw)[0]) for v in h_acc]
    y_acc = [sum(h[r] * w2[c * c1 + r] for r in range(c1))
             for c in range(c2)]
    sc2, sh2 = _mlp_scale(y_acc, dw, mw, sw)
    y = [requant_golden(v, sc2, sh2, dw)[0] for v in y_acc]
    # The two layers use different scales, and a design that applies the
    # first one to both has to fail. Check the vectors can see that.
    assert [requant_golden(v, sc1, sh1, dw)[0] for v in y_acc] != y, \
        "mlp testbench cannot tell scale1 from scale2"
    for name, accs in (("first", h_acc), ("second", y_acc)):
        assert max(abs(v) for v in accs) >= 1 << (aw // 2), \
            "mlp %s layer never reaches the top half of the accumulator" % name
    init = "\n".join(
        ["    acts[%d] = %s;" % (i, _slit(v, dw))
         for i, v in enumerate(acts)]
        + ["    wmem_tb[%d] = %s;" % (i, _slit(v, dw))
           for i, v in enumerate(w1 + w2)]
        + ["    expect_y[%d] = %s;" % (i, _slit(v, dw))
           for i, v in enumerate(y)]
        + ["    expect_h[%d] = %s;" % (i, _slit(v, dw))
           for i, v in enumerate(h)])
    return MLP_TB.format(
        dwm=dw - 1, awm=aw - 1, mwm=mw - 1, swm=sw - 1,
        bwm=p["bank_width"] - 1, depwm=p["depth_width"] - 1,
        colwm=p["col_width"] - 1, addrwm=p["addr_width"] - 1,
        d1=d1, c1=c1, c2=c2, bank=p["bank"], nw=len(w1 + w2), sc1=sc1, sh1=sh1,
        sc2=sc2, sh2=sh2, init=init)


def _mlp_scale(accs, dw, mw, sw):
    """A (scale, shift) that maps the largest accumulator just inside the
    output range, so every output lands in range and its value, not just
    its sign, is checked. That includes the large column, which is the
    one that exercises the accumulator's top bits: a clamped output
    there would hide exactly the error it exists to catch. Saturation
    itself is the requantizer's to test, and its own testbench does."""
    hi = (1 << (dw - 1)) - 2
    ref = max(1, max(abs(v) for v in accs))
    sh = 1
    while (hi << sh) // ref < (1 << (mw - 2)) and sh < (1 << sw) - 1:
        sh += 1
    return max(1, min((1 << mw) - 1, (hi << sh) // ref)), sh


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
  reg signed [{dwm}:0] expect_h [0:{c1}-1];
  integer nbad = 0, hbad = 0;
  reg [7:0] bad_idx [0:7];
  reg signed [{dwm}:0] bad_exp [0:7], bad_got [0:7];
  // Cycles measured, not computed: the profile's cycles_per_unit and
  // latency come from here.
  integer cyc = 0, span = 0, t0 = 0, first_out = -1, lat = 0;
  always @(posedge clk) cyc = cyc + 1;
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
      if (first_out < 0) begin first_out = cyc; if (cyc - t0 > lat) lat = cyc - t0; end
      if (o_index >= {c2}) begin
        $display("TB_FAIL test=%0s out=%0d cols2=%0d expected=no_output_past_cols2 got_y=%0d",
                 testname, o_index, {c2}, o_data);
        $display("TB_RESULT: FAIL");
        $finish;
      end
      if (o_data !== expect_y[o_index]) begin
        // Held, not reported yet: a wrong hidden layer makes every output
        // wrong, and the hidden check below names the cause.
        if (nbad < 8) begin
          bad_idx[nbad] = o_index; bad_exp[nbad] = expect_y[o_index];
          bad_got[nbad] = o_data;
        end
        nbad = nbad + 1;
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
    scale1 = {sc1}; shift1 = {sh1}; scale2 = {sc2}; shift2 = {sh2};
    @(negedge clk);
    start = 1;
      t0 = cyc; first_out = -1;
    @(negedge clk);
    start = 0;
    while (busy) @(negedge clk);
      span = span + (cyc - t0);
    repeat (8) @(negedge clk);
    // Whitebox check of the hidden layer, which the spec places in act at
    // {bank}..{bank}+cols1-1. Reported before the outputs because it is
    // upstream of them: traced, a draft whose writeback was wrong sat for
    // six iterations on "output 0 wrong" and never looked at layer one.
    for (i = 0; i < {c1}; i = i + 1) begin
      checks = checks + 1;
      if (dut.act[{bank} + i] !== expect_h[i]) begin
        $display("TB_FAIL test=hidden_layer h_index=%0d expected_h=%0d got_h=%0d",
                 i, expect_h[i], dut.act[{bank} + i]);
        hbad = hbad + 1;
      end
    end
    for (i = 0; i < nbad && i < 8; i = i + 1)
      $display("TB_FAIL test=%0s out=%0d expected_y=%0d got_y=%0d",
               testname, bad_idx[i], bad_exp[i], bad_got[i]);
    if (hbad || nbad) begin
      $display("TB_RESULT: FAIL");
      $finish;
    end
    checks = checks + 1;
    if (seen !== {c2}) begin
      $display("TB_FAIL test=%0s out=0 expected_y=%0d got_y=%0d",
               "output_count", {c2}, seen);
      $display("TB_RESULT: FAIL");
      $finish;
    end
    $display("TB_PROFILE layers=%0d span_cycles=%0d latency_cycles=%0d",
             1, span, lat);
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
    aw = acc_width(ms)
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
            "Let p = acc_in * scale. scale is a non-negative magnitude, "
            "so p is a signed %d-bit product with the sign of acc_in."
            % (aw + mw),
            "If shift is greater than zero, r = (p + 2**(shift-1)) >>> "
            "shift, an arithmetic right shift. That rounds to nearest "
            "with ties toward plus infinity: 2.5 becomes 3 and -2.5 "
            "becomes -2. If shift is zero, r = p and no rounding term "
            "is added. shift is always less than %d." % (aw + mw),
            "q_out is r clamped to the signed %d-bit range [%d, %d]. sat "
            "is 1 exactly when the clamp changed the value, and 0 "
            "otherwise."
            % (dw, -(1 << (dw - 1)), (1 << (dw - 1)) - 1),
            "Latency from valid_in to valid_out is exactly %d cycles, "
            "and q_out, sat and valid_out are sampled at exactly that "
            "cycle. If a design needs fewer register stages it must add "
            "delay registers to reach %d; more is also a failure."
            % (stages, stages),
            "The multiply is too wide for one cycle at the target "
            "clock. The reference splits scale into %d slices, "
            "registers the %d partial products, sums them in a "
            "registered adder tree, then registers the rounding add, "
            "the shift and the saturate. Any structure that meets both "
            "timing and the exact latency is acceptable."
            % (splits, requant_terms(aw, splits)),
            "Every pipeline register resets to zero, so valid_out, "
            "q_out and sat all read 0 during reset and on the first "
            "cycle after it.",
            "Saturating, not wrapping, is required: a wrapped overflow "
            "flips the sign of a large activation and corrupts every "
            "later layer.",
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

    # One vector per scale bit, each with a shift that brings the result
    # back into range. The scale is sliced across several partial
    # products, and without this the slices above the lowest are never
    # exercised on their own: mutation testing showed an inverted adder
    # inside one of them surviving every other vector. This says nothing
    # about how the multiply is built, only that every bit of the scale
    # has to matter.
    for b in range(mw):
        cases.append(("scale_bit", 100, 1 << b, b))
        cases.append(("scale_bit", -100, 1 << b, b))
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
