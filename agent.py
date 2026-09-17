"""Design agent. Rule-based stand-in for an LLM: propose(spec, feedback_history)
returns Verilog source derived from parsed tool feedback, never from an
iteration counter. To upgrade, replace RuleBasedAgent with a class whose
propose() sends spec plus feedback to an LLM and returns its Verilog.

Two block types are handled, dispatched on spec name: the fabric endpoint
(crc32_endpoint) and the compute chiplet (any MAC spec, including the ones
specgen.py derives from the model, at whatever data and accumulator widths
the derivation chose). Each carries its own seeded first-cut bugs so the demo
shows the feedback loop converging on both."""

FIX_WIDTH = "widen_product_register"
FIX_CLEAR = "implement_sync_clear"
FIX_XOR = "apply_final_inversion"


class RuleBasedAgent:
    def propose(self, spec, feedback_history):
        fixes = self.diagnose(spec, feedback_history)
        if spec["name"] == "crc32_endpoint":
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
        return """module mac (
  input              clk,
  input              rst_n,
  input              clear,
  input      [{d}:0] a,
  input      [{d}:0] b,
  input              valid_in,
  output reg [{a}:0] acc,
  output reg         valid_out
);
  reg [{p}:0] prod;
  reg         vpipe;
  always @(posedge clk) begin
    if (!rst_n) begin
      prod      <= 0;
      vpipe     <= 1'b0;
      acc       <= 0;
      valid_out <= 1'b0;
    end else begin
      prod  <= a * b;
      vpipe <= valid_in;
{logic}
    end
  end
endmodule
""".format(d=dw - 1, a=aw - 1, p=pw - 1, logic=acc_logic)

    def render_crc(self, spec, fixes):
        # First cut forgets the standard final inversion (crc_out = state
        # instead of state ^ 0xFFFFFFFF), a classic CRC32 bring-up bug: every
        # frame checksum is wrong by the same transformation, which the
        # golden-vector testbench catches on the first frame.
        out_expr = "nxt ^ 32'hFFFFFFFF" if FIX_XOR in fixes else "nxt"
        return """module crc32 (
  input             clk,
  input             rst_n,
  input             clear,
  input      [31:0] data,
  input             valid_in,
  output reg [31:0] crc_out,
  output reg        valid_out
);
  // zlib/Ethernet CRC32, reflected polynomial, 32 bits per cycle.
  function [31:0] step32;
    input [31:0] c;
    input [31:0] d;
    integer k;
    reg [31:0] x;
    begin
      x = c ^ d;
      for (k = 0; k < 32; k = k + 1)
        x = (x >> 1) ^ (32'hEDB88320 & {{32{{x[0]}}}});
      step32 = x;
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
      nxt = step32(state, data);
      state     <= nxt;
      crc_out   <= {out};
      valid_out <= 1'b1;
    end else begin
      valid_out <= 1'b0;
    end
  end
endmodule
""".format(out=out_expr)
