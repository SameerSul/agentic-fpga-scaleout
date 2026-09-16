"""Design agent. Rule-based stand-in for an LLM: propose(spec, feedback_history)
returns Verilog source derived from parsed tool feedback, never from an
iteration counter. To upgrade, replace RuleBasedAgent with a class whose
propose() sends spec plus feedback to an LLM and returns its Verilog."""

FIX_WIDTH = "widen_product_register"
FIX_CLEAR = "implement_sync_clear"


class RuleBasedAgent:
    def propose(self, spec, feedback_history):
        fixes = self.diagnose(feedback_history)
        return self.render(spec, fixes), sorted(fixes)

    def diagnose(self, history):
        """Map parsed feedback to fix intents, mimicking how an LLM would read
        tool output. Clear-test mismatches implicate the clear path, other
        accumulator mismatches implicate datapath width."""
        fixes = set()
        for fb in history:
            if fb.get("stage") != "sim":
                continue
            for m in fb.get("mismatches", []):
                if "clear" in m.get("test", "").lower():
                    fixes.add(FIX_CLEAR)
                elif m.get("got_acc") != m.get("expected_acc"):
                    fixes.add(FIX_WIDTH)
        return fixes

    def render(self, spec, fixes):
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
