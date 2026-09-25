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


class RuleBasedAgent:
    def propose(self, spec, feedback_history):
        fixes = self.diagnose(spec, feedback_history)
        # Dispatch on the block's top module, not its name: specgen derives
        # names that carry the parameters (mac8_gpt2_124m, crc32_endpoint_16B)
        # and those change with the model and the link rate.
        if spec["top_module"] == "crc32":
            return self.render_crc(spec, fixes), sorted(fixes)
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
                elif m.get("got_acc") != m.get("expected_acc"):
                    fixes.add(FIX_WIDTH)
        return fixes

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
