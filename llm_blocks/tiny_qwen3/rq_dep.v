module requant(
  input clk,
  input rst_n,
  input signed [31:0] acc_in,
  input [17:0] scale,
  input [6:0] shift,
  input valid_in,
  output signed [15:0] q_out,
  output sat,
  output valid_out
);

  reg signed [37:0] pp0_s1, pp1_s1, pp2_s1;
  reg signed [35:0] pp3_s1;
  reg [6:0] shift_s1;
  reg valid_s1;

  reg signed [49:0] sum0_s2, sum1_s2;
  reg [6:0] shift_s2;
  reg valid_s2;

  reg signed [49:0] product_s3;
  reg [6:0] shift_s3;
  reg valid_s3;

  reg signed [49:0] shifted_s4;
  reg should_round_s4;
  reg valid_s4;

  reg signed [49:0] rounded_s5;
  reg valid_s5;

  reg signed [15:0] q_out_s6;
  reg sat_s6;
  reg valid_s6;

  reg signed [15:0] q_out_s7, q_out_s8, q_out_s9;
  reg sat_s7, sat_s8, sat_s9;
  reg valid_s7, valid_s8, valid_s9;

  assign q_out = q_out_s9;
  assign sat = sat_s9;
  assign valid_out = valid_s9;

  always @(posedge clk) begin
    if (!rst_n) begin
      pp0_s1 <= 38'h0;
      pp1_s1 <= 38'h0;
      pp2_s1 <= 38'h0;
      pp3_s1 <= 36'h0;
      shift_s1 <= 7'h0;
      valid_s1 <= 1'h0;

      sum0_s2 <= 50'h0;
      sum1_s2 <= 50'h0;
      shift_s2 <= 7'h0;
      valid_s2 <= 1'h0;

      product_s3 <= 50'h0;
      shift_s3 <= 7'h0;
      valid_s3 <= 1'h0;

      shifted_s4 <= 50'h0;
      should_round_s4 <= 1'h0;
      valid_s4 <= 1'h0;

      rounded_s5 <= 50'h0;
      valid_s5 <= 1'h0;

      q_out_s6 <= 16'h0;
      sat_s6 <= 1'h0;
      valid_s6 <= 1'h0;

      q_out_s7 <= 16'h0;
      sat_s7 <= 1'h0;
      valid_s7 <= 1'h0;

      q_out_s8 <= 16'h0;
      sat_s8 <= 1'h0;
      valid_s8 <= 1'h0;

      q_out_s9 <= 16'h0;
      sat_s9 <= 1'h0;
      valid_s9 <= 1'h0;
    end else begin
      pp0_s1 <= acc_in * $signed({1'b0, scale[4:0]});
      pp1_s1 <= acc_in * $signed({1'b0, scale[9:5]});
      pp2_s1 <= acc_in * $signed({1'b0, scale[14:10]});
      pp3_s1 <= acc_in * $signed({1'b0, scale[17:15]});
      shift_s1 <= shift;
      valid_s1 <= valid_in;

      sum0_s2 <= pp0_s1 + (pp1_s1 << 5);
      sum1_s2 <= pp2_s1 + (pp3_s1 << 5);
      shift_s2 <= shift_s1;
      valid_s2 <= valid_s1;

      product_s3 <= sum0_s2 + (sum1_s2 << 10);
      shift_s3 <= shift_s2;
      valid_s3 <= valid_s2;

      shifted_s4 <= product_s3 >>> shift_s3;
      should_round_s4 <= (shift_s3 > 0) && product_s3[shift_s3 - 1];
      valid_s4 <= valid_s3;

      rounded_s5 <= should_round_s4 ? (shifted_s4 + 50'h1) : shifted_s4;
      valid_s5 <= valid_s4;

      if (rounded_s5 > 32767) begin
        q_out_s6 <= 16'sh7FFF;
        sat_s6 <= 1'b1;
      end else if (rounded_s5 < -32768) begin
        q_out_s6 <= 16'sh8000;
        sat_s6 <= 1'b1;
      end else begin
        q_out_s6 <= rounded_s5[15:0];
        sat_s6 <= 1'b0;
      end
      valid_s6 <= valid_s5;

      q_out_s7 <= q_out_s6;
      sat_s7 <= sat_s6;
      valid_s7 <= valid_s6;

      q_out_s8 <= q_out_s7;
      sat_s8 <= sat_s7;
      valid_s8 <= valid_s7;

      q_out_s9 <= q_out_s8;
      sat_s9 <= sat_s8;
      valid_s9 <= valid_s8;
    end
  end

endmodule
