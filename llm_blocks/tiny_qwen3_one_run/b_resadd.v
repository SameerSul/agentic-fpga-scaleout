module resadd (
  input clk,
  input rst_n,
  input signed [15:0] a,
  input signed [15:0] b,
  input [17:0] scale_a,
  input [17:0] scale_b,
  input [6:0] shift,
  input valid_in,
  output reg signed [15:0] y,
  output reg valid_out
);

  reg signed [34:0] prod_a_s1;
  reg signed [34:0] prod_b_s1;
  reg [6:0] shift_s1;
  reg valid_s1;

  always @(posedge clk) begin
    if (!rst_n) begin
      prod_a_s1 <= 35'sd0;
      prod_b_s1 <= 35'sd0;
      shift_s1 <= 7'b0;
      valid_s1 <= 1'b0;
    end else begin
      prod_a_s1 <= a * $signed({1'b0, scale_a});
      prod_b_s1 <= b * $signed({1'b0, scale_b});
      shift_s1 <= shift;
      valid_s1 <= valid_in;
    end
  end

  wire signed [35:0] v;
  assign v = prod_a_s1 + prod_b_s1;

  reg signed [35:0] v_s2;
  reg [6:0] shift_s2;
  reg valid_s2;

  always @(posedge clk) begin
    if (!rst_n) begin
      v_s2 <= 36'sd0;
      shift_s2 <= 7'b0;
      valid_s2 <= 1'b0;
    end else begin
      v_s2 <= v;
      shift_s2 <= shift_s1;
      valid_s2 <= valid_s1;
    end
  end

  wire signed [35:0] round_term;
  assign round_term = (shift_s2 > 0) ? (36'sd1 << (shift_s2 - 1)) : 36'sd0;

  wire signed [35:0] v_rounded;
  assign v_rounded = v_s2 + round_term;

  reg signed [35:0] v_rounded_s3;
  reg [6:0] shift_s3;
  reg valid_s3;

  always @(posedge clk) begin
    if (!rst_n) begin
      v_rounded_s3 <= 36'sd0;
      shift_s3 <= 7'b0;
      valid_s3 <= 1'b0;
    end else begin
      v_rounded_s3 <= v_rounded;
      shift_s3 <= shift_s2;
      valid_s3 <= valid_s2;
    end
  end

  wire signed [35:0] r;
  assign r = v_rounded_s3 >>> shift_s3;

  reg signed [35:0] r_s4;
  reg valid_s4;

  always @(posedge clk) begin
    if (!rst_n) begin
      r_s4 <= 36'sd0;
      valid_s4 <= 1'b0;
    end else begin
      r_s4 <= r;
      valid_s4 <= valid_s3;
    end
  end

  wire signed [15:0] y_next;
  assign y_next = (r_s4 > 32767) ? 32767 :
                  (r_s4 < -32768) ? -32768 :
                  r_s4[15:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      y <= 16'sd0;
      valid_out <= 1'b0;
    end else begin
      y <= y_next;
      valid_out <= valid_s4;
    end
  end

endmodule
