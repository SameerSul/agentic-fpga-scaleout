module mac_s(
  input clk,
  input rst_n,
  input clear,
  input signed [15:0] a,
  input signed [15:0] b,
  input valid_in,
  output reg signed [36:0] acc,
  output reg valid_out
);

  wire signed [15:0] a_low = {{8{1'b0}}, a[7:0]};
  wire signed [15:0] a_high = {{8{a[15]}}, a[15:8]};
  
  wire signed [31:0] pp_low = a_low * b;
  wire signed [31:0] pp_high = a_high * b;
  
  reg signed [31:0] pp_low_r, pp_high_r;
  reg valid_s1;
  
  always @(posedge clk) begin
    if (!rst_n) begin
      pp_low_r <= 32'sb0;
      pp_high_r <= 32'sb0;
      valid_s1 <= 1'b0;
    end else begin
      pp_low_r <= pp_low;
      pp_high_r <= pp_high;
      valid_s1 <= valid_in;
    end
  end
  
  wire signed [31:0] product_comb = pp_low_r + (pp_high_r << 8);
  
  reg signed [31:0] product;
  reg valid_s2;
  
  always @(posedge clk) begin
    if (!rst_n) begin
      product <= 32'sb0;
      valid_s2 <= 1'b0;
    end else begin
      product <= product_comb;
      valid_s2 <= valid_s1;
    end
  end
  
  wire signed [36:0] acc_next = acc + {{5{product[31]}}, product};
  
  always @(posedge clk) begin
    if (!rst_n) begin
      acc <= 37'sb0;
      valid_out <= 1'b0;
    end else if (clear) begin
      acc <= 37'sb0;
      valid_out <= 1'b0;
    end else begin
      if (valid_s2) begin
        acc <= acc_next;
      end
      valid_out <= valid_s2;
    end
  end

endmodule
