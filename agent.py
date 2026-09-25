"""Design agent. Rule-based stand-in for an LLM: propose(spec, feedback_history)
returns Verilog source derived from parsed tool feedback, never from an
iteration counter. To upgrade, replace RuleBasedAgent with a class whose
propose() sends spec plus feedback to an LLM and returns its Verilog.

Two block types are handled, dispatched on spec name: the fabric endpoint
(crc32_endpoint) and the compute chiplet (any MAC spec, including the ones
specgen.py derives from the model, at whatever data and accumulator widths
the derivation chose). Each carries its own seeded first-cut bugs so the demo
shows the feedback loop converging on both."""

import specgen

FIX_WIDTH = "widen_product_register"
FIX_CLEAR = "implement_sync_clear"
FIX_XOR = "apply_final_inversion"
FIX_SATURATE = "saturate_instead_of_wrap"
FIX_LUT = "interpolate_the_fractional_part"
# One definition, in specgen, so the declared depth and the generated
# depth cannot disagree.
WIDE_ADD_BITS = specgen.WIDE_ADD_BITS
ACC_SLICE_BITS = specgen.ACC_SLICE_BITS


class RuleBasedAgent:
    def propose(self, spec, feedback_history):
        fixes = self.diagnose(spec, feedback_history)
        # Dispatch on the block's top module, not its name: specgen derives
        # names that carry the parameters (mac8_gpt2_124m, crc32_endpoint_16B)
        # and those change with the model and the link rate.
        if spec["top_module"] == "crc32":
            return self.render_crc(spec, fixes), sorted(fixes)
        if spec["top_module"] == "requant":
            return self.render_requant(spec, fixes), sorted(fixes)
        if spec["top_module"] == "expu":
            return self.render_exp(spec, fixes), sorted(fixes)
        return self.render_mac(spec, fixes), sorted(fixes)

    def diagnose(self, spec, history):
        """Map parsed feedback to fix intents, mimicking how an LLM would read
        tool output. MAC: clear-test mismatches implicate the clear path,
        other accumulator mismatches implicate datapath width. CRC: a wrong
        checksum on every frame with correct relative behavior implicates the
        standard final inversion."""
        fixes = set()
        for fb in history:
            if fb.get("stage") != "sim":
                continue
            for m in fb.get("mismatches", []):
                if "expected_crc" in m:
                    fixes.add(FIX_XOR)
                elif "clear" in m.get("test", "").lower():
                    fixes.add(FIX_CLEAR)
                elif "expected_y" in m:
                    # The exponential's only seeded bug: the fractional
                    # part of the exponent was thrown away.
                    fixes.add(FIX_LUT)
                elif "expected_q" in m:
                    # The requantizer's only seeded bug: it wrapped where
                    # it had to saturate.
                    fixes.add(FIX_SATURATE)
                elif m.get("got_acc") != m.get("expected_acc"):
                    fixes.add(FIX_WIDTH)
        return fixes

    def render_exp(self, spec, fixes):
        """Fixed-point exponential by table and shift.

        exp(x) = 2**(x*log2(e)). Split x*log2(e) into an integer part and
        a fraction: the fraction indexes a table of 2**f and the integer
        part, which is non-positive because softmax subtracts the row
        maximum first, is a right shift.

        The seeded first-cut bug is dropping the fractional part, so the
        unit returns a plain power of two and every value between them is
        wrong by up to a factor of two.

        A logical shift where the integer part wants an arithmetic one was
        tried as the seeded bug first and turned out to be unobservable
        here: the two differ only in the high bits, and only the low bits
        of the shift amount survive into sh, so they agree modulo its
        width. The same reasoning showed a separate flush-to-zero flag was
        dead logic, because clamping the shift already drives the output
        to zero. Both are gone; the shift is arithmetic because that is
        what it means, not because the testbench can tell.
        """
        p = spec["parameters"]
        iw, fi = p["in_width"], p["in_frac"]
        ow, fo, lb = p["out_width"], p["out_frac"], p["lut_bits"]
        tw = iw + 18
        shw = max(4, (tw).bit_length())
        lut = specgen.exp_lut(lb, fo)
        arms = "\n".join(
            "      %d'd%d: lut = %d'd%d;" % (lb, i, ow, v)
            for i, v in enumerate(lut))
        return """module expu (
  input                     clk,
  input                     rst_n,
  input      signed [{iwm}:0] x,
  input                     valid_in,
  output reg        [{owm}:0] y,
  output reg                valid_out
);
  // exp(x) = 2**(x*log2(e)); the fraction of x*log2(e) indexes a table of
  // 2**f and its integer part is a right shift. x is non-positive, which
  // softmax guarantees by subtracting the row maximum, so the shift only
  // ever goes right.
  function [{owm}:0] lut;
    input [{lbm}:0] idx;
    case (idx)
{arms}
      default: lut = {ow}'d{one};
    endcase
  endfunction

  reg signed [{twm}:0] t;
  reg        [{owm}:0] m;
  reg        [{shm}:0] sh;
  reg                  vpipe, vpipe2;
  wire signed [{twm}:0] prod = x * $signed({{1'b0, 18'd{log2e}}});
  wire signed [{twm}:0] tt   = prod >>> 16;
  wire signed [{twm}:0] n    = t >>> {fi};
  always @(posedge clk) begin
    if (!rst_n) begin
      t         <= 0;
      m         <= 0;
      sh        <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      y         <= 0;
      valid_out <= 1'b0;
    end else begin
      t         <= tt;
      vpipe     <= valid_in;
      m         <= {mexpr};
      // Clamping the shift is what drives an underflowing input to zero:
      // the table entry is {ow} bits, so a shift of {fo1} empties it. A
      // separate flush flag was redundant.
      sh        <= (-n) > {fo} ? {fo1} : (-n);
      vpipe2    <= vpipe;
      y         <= m >> sh;
      valid_out <= vpipe2;
    end
  end
endmodule
""".format(iwm=iw - 1, owm=ow - 1, twm=tw - 1, shm=shw - 1, lbm=lb - 1,
           fim=fi - 1, fi=fi, fo=fo, fo1=fo + 1, ow=ow, arms=arms,
           one=1 << fo, log2e=specgen.LOG2E_Q16,
           mexpr=("lut(t[%d:0])" % (fi - 1)) if FIX_LUT in fixes
                 else ("%d'd%d" % (ow, 1 << fo)))

    def render_requant(self, spec, fixes):
        """Requantizer: scale, round, saturate.

        The structure is generated, not written, because every structure
        guessed at was wrong and OpenSTA named a different critical path
        each time: the rounding shifters, then the scale multiply, then the
        adder recombination. The final constraint is blunt. In this generic
        cell library, with no carry chain, a 64-bit add is about 13.4 ns
        against a 10 ns budget, and every wide add in the block violated,
        so no amount of rearranging helps. Wide adds are therefore split
        across two stages with the carry registered between them.

        Latency is the currency being spent, and it is nearly free here: a
        requantization happens once per dot product, so once per
        reduction_depth MACs, which is at least 64 and usually thousands.

        The seeded first-cut bug is wrapping instead of saturating, the
        mistake this block exists to prevent: a wrapped overflow flips the
        sign of a large activation and corrupts every later layer.
        """
        p = spec["parameters"]
        aw, dw = p["acc_width"], p["out_width"]
        mw, sw = p["scale_width"], p["shift_width"]
        n = p.get("scale_splits", 3)
        pw = aw + mw
        wide = pw > WIDE_ADD_BITS
        half_pt = pw // 2
        lo_s, hi_s = -(1 << (dw - 1)), (1 << (dw - 1)) - 1

        decls, reset = [], []
        stages = []          # each entry: list of "dst <= expr" strings

        def reg(name, width=None, init="0"):
            decls.append("  reg  signed [%d:0] %s;" % ((width or pw) - 1, name)
                         if width != 1 else "  reg  %s;" % name)
            reset.append("      %-10s <= %s;" % (name, init))

        def add_stage(pairs):
            stages.append(pairs)

        def emit_add(dst, x, y, into):
            """One add, as one stage when it fits and two when it does not.
            The split keeps the carry in a register between halves."""
            if not wide:
                reg(dst)
                into.append([(dst, "%s + %s" % (x, y))])
                return
            lo, hh, yh = dst + "_lo", dst + "_xh", dst + "_yh"
            decls.append("  reg  [%d:0] %s;" % (half_pt, lo))
            reset.append("      %-10s <= 0;" % lo)
            reg(hh)
            reg(yh)
            reg(dst)
            into.append([(lo, "{1'b0, %s[%d:0]} + {1'b0, %s[%d:0]}"
                          % (x, half_pt - 1, y, half_pt - 1)),
                         (hh, x), (yh, y)])
            into.append([(dst, "{%s[%d:%d] + %s[%d:%d] + %s[%d], %s[%d:0]}"
                          % (hh, pw - 1, half_pt, yh, pw - 1, half_pt,
                             lo, half_pt, lo, half_pt - 1))])

        # Stage 0: partial products, each already weighted so the tree
        # never carries a shift.
        #
        # Both operands are sliced, not just the scale. Slicing only the
        # scale left the critical path at acc_in to a partial product:
        # multiplying a 46-bit accumulator by even a 3-bit slice is two
        # wide adds of shifted copies, so the multiplicand's width is on
        # the path whatever the multiplier does. The high slice of acc_in
        # keeps the sign; the low slice is unsigned.
        an = 2 if aw > ACC_SLICE_BITS else 1
        sb = [(i * mw // n, ((i + 1) * mw // n) - 1) for i in range(n)]
        sb[-1] = (sb[-1][0], mw - 1)
        ab = [(i * aw // an, ((i + 1) * aw // an) - 1) for i in range(an)]
        ab[-1] = (ab[-1][0], aw - 1)
        cur, s0 = [], []
        for ai, (alo, ahi) in enumerate(ab):
            atop = (ai == an - 1)
            aterm = ("$signed(acc_in[%d:%d])" % (ahi, alo) if atop
                     else "$signed({1'b0, acc_in[%d:%d]})" % (ahi, alo))
            for si, (slo, shi) in enumerate(sb):
                nm = "pp%d_%d" % (ai, si)
                cur.append(nm)
                reg(nm)
                e = "%s * $signed({1'b0, scale[%d:%d]})" % (aterm, shi, slo)
                w = alo + slo
                s0.append((nm, "(%s) <<< %d" % (e, w) if w else e))
        add_stage(s0)

        # Pairwise reduction.
        lvl = 0
        per_add = 2 if wide else 1
        while len(cur) > 1:
            lvl += 1
            nxt, groups = [], []
            for i in range(0, len(cur) - 1, 2):
                dst = "s%d_%d" % (lvl, i // 2)
                nxt.append(dst)
                own = []                     # this add's own stage list
                emit_add(dst, cur[i], cur[i + 1], own)
                groups.append(own)
            if len(cur) % 2:
                # An odd term is carried forward, and must be delayed by
                # exactly as many stages as the adds beside it or it
                # arrives at the next level a cycle early.
                dst = "s%d_%d" % (lvl, len(cur) // 2)
                nxt.append(dst)
                own, src = [], cur[-1]
                for k in range(per_add):
                    nm = dst if k == per_add - 1 else "%s_d%d" % (dst, k)
                    reg(nm)
                    own.append([(nm, src)])
                    src = nm
                groups.append(own)
            # Every add in a level takes the same number of stages, so the
            # level's stages are the per-add stages merged position by
            # position.
            for k in range(per_add):
                merged = []
                for grp in groups:
                    if k < len(grp):
                        merged.extend(grp[k])
                add_stage(merged)
            cur = nxt
        rdepth = len(stages) - 1

        # Rounding add, then the output shift, then the saturate.
        built = []
        emit_add("summed", cur[0], "hf_last", built)
        for b in built:
            add_stage(b)
        # The saturate stays in the last stage. Precomputing the two
        # range flags beside the shift was tried, so the last stage would
        # be a small mux: it duplicates the barrel shifter three times and
        # took the real device from 97.88 MHz to 58.78. Measured, reverted.
        reg("shifted")
        add_stage([("shifted", "summed >>> sh_last")])
        nstage = len(stages) + 1           # + the saturate stage

        # Sideband pipelines, each exactly as deep as the point it is used.
        hf_depth = 1 + rdepth              # hf is consumed by the first
        sh_depth = len(stages) - 1         # sh by the shift stage
        for nm, depth, src in (("hf", hf_depth, "half_w"),
                               ("sh", sh_depth, "shift")):
            w = pw if nm == "hf" else sw
            names = ["%s%d" % (nm, i) for i in range(depth)]
            decls.append("  reg  %s[%d:0] %s;"
                         % ("signed " if nm == "hf" else "", w - 1,
                            ", ".join(names)))
            for x in names:
                reset.append("      %-10s <= 0;" % x)
            stages[0].append((names[0], src))
            for i in range(1, depth):
                stages[i].append((names[i], names[i - 1]))
        body = "\n".join(
            "\n".join("      %-10s <= %s;" % (d, e) for d, e in st)
            for st in stages)
        body = body.replace("hf_last", "hf%d" % (hf_depth - 1))
        body = body.replace("sh_last", "sh%d" % (sh_depth - 1))

        vnames = ["v%d" % i for i in range(nstage - 1)]
        decls.append("  reg  %s;" % ", ".join(vnames))
        for x in vnames:
            reset.append("      %-10s <= 1'b0;" % x)
        vlines = ["      %-10s <= valid_in;" % vnames[0]]
        vlines += ["      %-10s <= %s;" % (vnames[i], vnames[i - 1])
                   for i in range(1, len(vnames))]

        if FIX_SATURATE in fixes:
            clamp = """      if (shifted > %d'sd%d) begin
        q_out <= %d'sd%d;
        sat   <= 1'b1;
      end else if (shifted < -%d'sd%d) begin
        q_out <= -%d'sd%d;
        sat   <= 1'b1;
      end else begin
        q_out <= shifted[%d:0];
        sat   <= 1'b0;
      end""" % (pw, hi_s, dw, hi_s, pw, -lo_s, dw, -lo_s, dw - 1)
        else:   # first cut truncates, which wraps on overflow
            clamp = ("      q_out <= shifted[%d:0];\n"
                     "      sat   <= 1'b0;" % (dw - 1))

        return """module requant (
  input                     clk,
  input                     rst_n,
  input      signed [{awm}:0] acc_in,
  input             [{mwm}:0] scale,
  input             [{swm}:0] shift,
  input                     valid_in,
  output reg signed [{dwm}:0] q_out,
  output reg                sat,
  output reg                valid_out
);
  // scale is unsigned, so it is zero-extended before the signed multiply:
  // mixing a signed and an unsigned operand makes the whole expression
  // unsigned in Verilog and silently breaks every negative accumulator.
  // half depends only on shift, so it is built before the tree and never
  // sits on the rounding path.
{decls}
  wire signed [{pwm}:0] half_w = (shift == 0)
        ? {pw}'sd0 : ({pw}'sd1 <<< (shift - 1));
  always @(posedge clk) begin
    if (!rst_n) begin
{reset}
      q_out      <= 0;
      sat        <= 1'b0;
      valid_out  <= 1'b0;
    end else begin
{body}
{vlines}
{clamp}
      valid_out  <= {vlast};
    end
  end
endmodule
""".format(awm=aw - 1, mwm=mw - 1, swm=sw - 1, dwm=dw - 1, pwm=pw - 1,
           pw=pw, decls="\n".join(decls), reset="\n".join(reset),
           body=body, vlines="\n".join(vlines), clamp=clamp,
           vlast=vnames[-1])

    def render_mac(self, spec, fixes):
        p = spec["parameters"]
        dw, aw = p["data_width"], p["acc_width"]
        pw = 2 * dw if FIX_WIDTH in fixes else dw  # first cut truncates the product
        # Quantized weights are two's complement, so the operands, the
        # product and the accumulator are all signed and the product is
        # sign-extended on the way in. Declaring only some of them signed
        # is worse than declaring none: Verilog makes the whole expression
        # unsigned if any operand is, so a half-signed datapath silently
        # computes the unsigned answer.
        sg = "signed " if p.get("signed", True) else ""
        if FIX_CLEAR in fixes:
            acc_logic = """      if (clear) begin
        acc <= {aw}'d0;
        valid_out <= 1'b0;
      end else begin
        if (vpipe) acc <= acc + prod;
        valid_out <= vpipe;
      end""".format(aw=aw)
        else:  # first cut forgets the synchronous clear
            acc_logic = """      if (vpipe) acc <= acc + prod;
      valid_out <= vpipe;"""
        stages = p.get("pipeline_stages", 2)
        if stages >= 3:
            # Split the multiply. b's high half is signed, its low half is
            # not, so the two partial products are formed separately and
            # recombined a cycle later. Each multiplier is half as deep as
            # the full one, which is where the clock comes back.
            h = dw // 2
            decl = ("  reg {sg}[{p}:0] p_lo, p_hi;\n"
                    "  reg {sg}[{p}:0] prod;\n"
                    "  reg         vpipe, vpipe2;").format(sg=sg, p=pw - 1)
            reset = ("      p_lo      <= 0;\n"
                     "      p_hi      <= 0;\n"
                     "      prod      <= 0;\n"
                     "      vpipe     <= 1'b0;\n"
                     "      vpipe2    <= 1'b0;")
            drive = ("      p_lo   <= a * $signed({{1'b0, b[{hm}:0]}});\n"
                     "      p_hi   <= a * $signed(b[{d}:{h}]);\n"
                     "      prod   <= p_lo + (p_hi <<< {h});\n"
                     "      vpipe  <= valid_in;\n"
                     "      vpipe2 <= vpipe;").format(hm=h - 1, d=dw - 1, h=h)
            vq = "vpipe2"
        else:
            decl = ("  reg {sg}[{p}:0] prod;\n"
                    "  reg         vpipe;").format(sg=sg, p=pw - 1)
            reset = ("      prod      <= 0;\n"
                     "      vpipe     <= 1'b0;")
            drive = ("      prod  <= a * b;\n"
                     "      vpipe <= valid_in;")
            vq = "vpipe"
        acc_logic = acc_logic.replace("vpipe)", vq + ")").replace(
            "<= vpipe;", "<= " + vq + ";")
        return """module mac (
  input                     clk,
  input                     rst_n,
  input                     clear,
  input      {sg}[{d}:0] a,
  input      {sg}[{d}:0] b,
  input                     valid_in,
  output reg {sg}[{a}:0] acc,
  output reg                valid_out
);
{decl}
  always @(posedge clk) begin
    if (!rst_n) begin
{reset}
      acc       <= 0;
      valid_out <= 1'b0;
    end else begin
{drive}
{logic}
    end
  end
endmodule
""".format(d=dw - 1, a=aw - 1, logic=acc_logic, sg=sg,
           decl=decl, reset=reset, drive=drive)

    def render_crc_matrix(self, spec, fixes):
        """CRC32 next state as one XOR reduction per output bit.

        The step function is linear over GF(2), so the next state is the XOR
        of a fixed set of current-state and input bits. Written this way the
        combinational depth is logarithmic in the datapath width rather than
        linear in it, which is the difference between an endpoint that meets
        its clock at 32 bytes per cycle and one that misses by 11 ns. The
        selection matrices come from specgen, which derives them from the
        same polynomial the golden vectors use.
        """
        w = spec["parameters"]["bytes_per_cycle"]
        A, B = specgen.crc_matrix(w)
        out_expr = "nxt ^ 32'hFFFFFFFF" if FIX_XOR in fixes else "nxt"
        lines = []
        for i in range(32):
            terms = ["state[%d]" % j for j in range(32) if (A[j] >> i) & 1]
            terms += ["data[%d]" % k for k in range(8 * w) if (B[k] >> i) & 1]
            if not terms:
                lines.append("  assign nxt[%d] = 1'b0;" % i)
                continue
            body, cur = [], "  assign nxt[%d] = ^{" % i
            for t in terms:
                piece = t + ", "
                if len(cur) + len(piece) > 76:
                    body.append(cur)
                    cur = "      "
                cur += piece
            body.append(cur.rstrip(", ") + "};")
            lines.extend(body)
        return """module crc32 (
  input              clk,
  input              rst_n,
  input              clear,
  input      [{dm}:0] data,
  input              valid_in,
  output reg [31:0]  crc_out,
  output reg         valid_out
);
  // zlib/Ethernet CRC32, reflected polynomial 0xEDB88320, {w} byte(s) per
  // cycle, bytes consumed LSB-first. Next state is a GF(2) linear function
  // of the state and the input word, so each bit is a single XOR reduction
  // and the depth is logarithmic in the width rather than linear in it.
  reg  [31:0] state;
  wire [31:0] nxt;
{assigns}

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= 32'hFFFFFFFF;
      crc_out   <= 32'd0;
      valid_out <= 1'b0;
    end else if (clear) begin
      state     <= 32'hFFFFFFFF;
      valid_out <= 1'b0;
    end else if (valid_in) begin
      state     <= nxt;
      crc_out   <= {out};
      valid_out <= 1'b1;
    end else begin
      valid_out <= 1'b0;
    end
  end
endmodule
""".format(out=out_expr, w=w, dm=8 * w - 1, assigns="\n".join(lines))

    def render_crc(self, spec, fixes):
        if spec["parameters"].get("architecture") == "matrix":
            return self.render_crc_matrix(spec, fixes)
        # Width-generic: the endpoint consumes bytes_per_cycle bytes per
        # clock, derived from the link rate it has to keep up with, so the
        # same agent covers a 1-byte 1GbE endpoint and a 16-byte 25GbE one.
        # First cut forgets the standard final inversion (crc_out = state
        # instead of state ^ 0xFFFFFFFF), a classic CRC32 bring-up bug: every
        # frame checksum is wrong by the same transformation, which the
        # golden-vector testbench catches on the first frame.
        w = spec["parameters"]["bytes_per_cycle"]
        out_expr = "nxt ^ 32'hFFFFFFFF" if FIX_XOR in fixes else "nxt"
        return """module crc32 (
  input              clk,
  input              rst_n,
  input              clear,
  input      [{dm}:0] data,
  input              valid_in,
  output reg [31:0]  crc_out,
  output reg         valid_out
);
  // zlib/Ethernet CRC32, reflected polynomial 0xEDB88320, {w} byte(s) per
  // cycle, bytes consumed LSB-first (little-endian packing on the wire).
  function [31:0] stepw;
    input [31:0] c;
    input [{dm}:0] d;
    integer i, k;
    reg [31:0] x;
    reg [7:0]  b;
    begin
      x = c;
      for (i = 0; i < {w}; i = i + 1) begin
        b = (d >> (8 * i));
        x = x ^ {{24'd0, b}};
        for (k = 0; k < 8; k = k + 1)
          x = (x >> 1) ^ (32'hEDB88320 & {{32{{x[0]}}}});
      end
      stepw = x;
    end
  endfunction

  reg [31:0] state;
  reg [31:0] nxt;
  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= 32'hFFFFFFFF;
      crc_out   <= 32'd0;
      valid_out <= 1'b0;
    end else if (clear) begin
      state     <= 32'hFFFFFFFF;
      valid_out <= 1'b0;
    end else if (valid_in) begin
      nxt = stepw(state, data);
      state     <= nxt;
      crc_out   <= {out};
      valid_out <= 1'b1;
    end else begin
      valid_out <= 1'b0;
    end
  end
endmodule
""".format(out=out_expr, w=w, dm=8 * w - 1)
