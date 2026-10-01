module rsqrt(
  input clk,
  input rst_n,
  input [39:0] x,
  input valid_in,
  output reg [16:0] y,
  output reg [5:0] e,
  output reg valid_out
);

  function [5:0] find_highest_bit;
    input [39:0] val;
    integer i;
    begin
      find_highest_bit = 6'b0;
      for (i = 0; i < 40; i = i + 1) begin
        if (val[i]) begin
          find_highest_bit = i;
        end
      end
    end
  endfunction

  // Stage 1: Normalization
  wire [5:0] b = find_highest_bit(x);
  wire [5:0] e_s1 = b >> 1;
  wire [5:0] s = 38 - (e_s1 << 1);
  wire [39:0] xn_s1 = x << s;
  wire is_zero_s1 = (x == 40'b0);

  reg [5:0] e_s2;
  reg [39:0] xn_s2;
  reg valid_s2;
  reg is_zero_s2;

  always @(posedge clk) begin
    if (!rst_n) begin
      e_s2 <= 6'b0;
      xn_s2 <= 40'b0;
      valid_s2 <= 1'b0;
      is_zero_s2 <= 1'b0;
    end else begin
      e_s2 <= e_s1;
      xn_s2 <= xn_s1;
      valid_s2 <= valid_in;
      is_zero_s2 <= is_zero_s1;
    end
  end

  // Stage 2: ROM lookup
  wire [7:0] idx = xn_s2[39:32];
  wire [16:0] rom_val;

  rsqrt_rom rom_inst(
    .idx(idx),
    .val(rom_val)
  );

  reg [16:0] mantissa_s3;
  reg [5:0] e_s3;
  reg valid_s3;
  reg is_zero_s3;

  always @(posedge clk) begin
    if (!rst_n) begin
      mantissa_s3 <= 17'b0;
      e_s3 <= 6'b0;
      valid_s3 <= 1'b0;
      is_zero_s3 <= 1'b0;
    end else begin
      mantissa_s3 <= rom_val;
      e_s3 <= e_s2;
      valid_s3 <= valid_s2;
      is_zero_s3 <= is_zero_s2;
    end
  end

  // Stage 3: Output
  always @(posedge clk) begin
    if (!rst_n) begin
      y <= 17'b0;
      e <= 6'b0;
      valid_out <= 1'b0;
    end else begin
      valid_out <= valid_s3;
      if (is_zero_s3) begin
        y <= 17'h1FFFF;
        e <= 6'b0;
      end else begin
        y <= mantissa_s3;
        e <= e_s3;
      end
    end
  end

endmodule
