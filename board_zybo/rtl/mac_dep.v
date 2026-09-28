module mac (
  input                     clk,
  input                     rst_n,
  input                     clear,
  input      signed [15:0] a,
  input      signed [15:0] b,
  input                     valid_in,
  output reg signed [36:0] acc,
  output reg                valid_out
);
  reg signed [31:0] p_lo, p_hi;
  reg signed [31:0] prod;
  reg         vpipe, vpipe2;
  always @(posedge clk) begin
    if (!rst_n) begin
      p_lo      <= 0;
      p_hi      <= 0;
      prod      <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      acc       <= 0;
      valid_out <= 1'b0;
    end else begin
      p_lo   <= a * $signed({1'b0, b[7:0]});
      p_hi   <= a * $signed(b[15:8]);
      prod   <= p_lo + (p_hi <<< 8);
      vpipe  <= valid_in;
      vpipe2 <= vpipe;
      if (clear) begin
        acc <= 37'd0;
        valid_out <= 1'b0;
      end else begin
        if (vpipe2) acc <= acc + prod;
        valid_out <= vpipe2;
      end
    end
  end
endmodule
