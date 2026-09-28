module requant (
  input                     clk,
  input                     rst_n,
  input      signed [23:0] acc_in,
  input             [17:0] scale,
  input             [6:0] shift,
  input                     valid_in,
  output reg signed [7:0] q_out,
  output reg                sat,
  output reg                valid_out
);
  // scale is unsigned, so it is zero-extended before the signed multiply:
  // mixing a signed and an unsigned operand makes the whole expression
  // unsigned in Verilog and silently breaks every negative accumulator.
  // half depends only on shift, so it is built before the tree and never
  // sits on the rounding path.
  reg  signed [41:0] pp0_0;
  reg  signed [41:0] pp0_1;
  reg  signed [41:0] pp0_2;
  reg  signed [41:0] s1_0;
  reg  signed [41:0] s1_1;
  reg  signed [41:0] s2_0;
  reg  signed [41:0] summed;
  reg  signed [41:0] shifted;
  reg  signed [41:0] hf0, hf1, hf2;
  reg  [6:0] sh0, sh1, sh2, sh3;
  reg  v0, v1, v2, v3, v4;
  wire signed [41:0] half_w = (shift == 0)
        ? 42'sd0 : (42'sd1 <<< (shift - 1));
  always @(posedge clk) begin
    if (!rst_n) begin
      pp0_0      <= 0;
      pp0_1      <= 0;
      pp0_2      <= 0;
      s1_0       <= 0;
      s1_1       <= 0;
      s2_0       <= 0;
      summed     <= 0;
      shifted    <= 0;
      hf0        <= 0;
      hf1        <= 0;
      hf2        <= 0;
      sh0        <= 0;
      sh1        <= 0;
      sh2        <= 0;
      sh3        <= 0;
      v0         <= 1'b0;
      v1         <= 1'b0;
      v2         <= 1'b0;
      v3         <= 1'b0;
      v4         <= 1'b0;
      q_out      <= 0;
      sat        <= 1'b0;
      valid_out  <= 1'b0;
    end else begin
      pp0_0      <= $signed(acc_in[23:0]) * $signed({1'b0, scale[5:0]});
      pp0_1      <= ($signed(acc_in[23:0]) * $signed({1'b0, scale[11:6]})) <<< 6;
      pp0_2      <= ($signed(acc_in[23:0]) * $signed({1'b0, scale[17:12]})) <<< 12;
      hf0        <= half_w;
      sh0        <= shift;
      s1_0       <= pp0_0 + pp0_1;
      s1_1       <= pp0_2;
      hf1        <= hf0;
      sh1        <= sh0;
      s2_0       <= s1_0 + s1_1;
      hf2        <= hf1;
      sh2        <= sh1;
      summed     <= s2_0 + hf2;
      sh3        <= sh2;
      shifted    <= summed >>> sh3;
      v0         <= valid_in;
      v1         <= v0;
      v2         <= v1;
      v3         <= v2;
      v4         <= v3;
      if (shifted > 42'sd127) begin
        q_out <= 8'sd127;
        sat   <= 1'b1;
      end else if (shifted < -42'sd128) begin
        q_out <= -8'sd128;
        sat   <= 1'b1;
      end else begin
        q_out <= shifted[7:0];
        sat   <= 1'b0;
      end
      valid_out  <= v4;
    end
  end
endmodule
