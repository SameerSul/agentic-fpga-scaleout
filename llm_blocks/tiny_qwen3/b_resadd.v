module resadd (
  input wire clk,
  input wire rst_n,
  input wire signed [15:0] a,
  input wire signed [15:0] b,
  input wire [17:0] scale_a,
  input wire [17:0] scale_b,
  input wire [6:0] shift,
  input wire valid_in,
  output reg signed [15:0] y,
  output reg valid_out
);

  reg signed [15:0] a_p1, b_p1;
  reg [17:0] scale_a_p1, scale_b_p1;
  reg [6:0] shift_p1;
  reg valid_p1;

  always @(posedge clk) begin
    if (!rst_n) begin
      a_p1 <= 0;
      b_p1 <= 0;
      scale_a_p1 <= 0;
      scale_b_p1 <= 0;
      shift_p1 <= 0;
      valid_p1 <= 0;
    end else begin
      a_p1 <= a;
      b_p1 <= b;
      scale_a_p1 <= scale_a;
      scale_b_p1 <= scale_b;
      shift_p1 <= shift;
      valid_p1 <= valid_in;
    end
  end

  wire signed [18:0] scale_a_p1_ext = {{1{1'b0}}, scale_a_p1};
  wire signed [18:0] scale_b_p1_ext = {{1{1'b0}}, scale_b_p1};
  wire signed [35:0] prod_a_p1 = a_p1 * scale_a_p1_ext;
  wire signed [35:0] prod_b_p1 = b_p1 * scale_b_p1_ext;

  reg signed [35:0] prod_a_p2, prod_b_p2;
  reg [6:0] shift_p2;
  reg valid_p2;

  always @(posedge clk) begin
    if (!rst_n) begin
      prod_a_p2 <= 0;
      prod_b_p2 <= 0;
      shift_p2 <= 0;
      valid_p2 <= 0;
    end else begin
      prod_a_p2 <= prod_a_p1;
      prod_b_p2 <= prod_b_p1;
      shift_p2 <= shift_p1;
      valid_p2 <= valid_p1;
    end
  end

  wire signed [35:0] sum_p2 = prod_a_p2 + prod_b_p2;
  wire [35:0] round_bits_p2 = (shift_p2 > 0) ? (36'd1 << (shift_p2 - 1)) : 36'd0;
  wire signed [35:0] rounded_p2 = sum_p2 + round_bits_p2;

  reg signed [35:0] rounded_p3;
  reg [6:0] shift_p3;
  reg valid_p3;

  always @(posedge clk) begin
    if (!rst_n) begin
      rounded_p3 <= 0;
      shift_p3 <= 0;
      valid_p3 <= 0;
    end else begin
      rounded_p3 <= rounded_p2;
      shift_p3 <= shift_p2;
      valid_p3 <= valid_p2;
    end
  end

  wire signed [35:0] shifted_p3 = rounded_p3 >>> shift_p3;

  reg signed [35:0] shifted_p4;
  reg valid_p4;

  always @(posedge clk) begin
    if (!rst_n) begin
      shifted_p4 <= 0;
      valid_p4 <= 0;
    end else begin
      shifted_p4 <= shifted_p3;
      valid_p4 <= valid_p3;
    end
  end

  wire signed [15:0] saturated_p4 = (shifted_p4 > 32767) ? 16'h7FFF :
                                     (shifted_p4 < -32768) ? 16'h8000 :
                                     shifted_p4[15:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      y <= 0;
      valid_out <= 0;
    end else begin
      y <= saturated_p4;
      valid_out <= valid_p4;
    end
  end

endmodule
