module rsqrt(
  input clk,
  input rst_n,
  input [39:0] x,
  input valid_in,
  output [16:0] y,
  output [5:0] e,
  output valid_out
);

  reg [5:0] e_s1;
  reg [39:0] xn_s1;
  reg valid_s1;
  reg is_zero_s1;

  reg [16:0] rom_val_s2;
  reg [5:0] e_s2;
  reg valid_s2;
  reg is_zero_s2;

  reg [16:0] y_s3;
  reg [5:0] e_s3;
  reg valid_s3;

  function [5:0] find_msb_pos;
    input [39:0] val;
    integer i;
    begin
      find_msb_pos = 6'b0;
      for (i = 0; i < 40; i = i + 1) begin
        if (val[i]) begin
          find_msb_pos = i;
        end
      end
    end
  endfunction

  wire [5:0] b;
  wire [5:0] e_calc;
  wire [5:0] s;
  wire [39:0] x_shifted;
  wire is_zero_input;

  assign is_zero_input = (x == 40'b0);
  assign b = find_msb_pos(x);
  assign e_calc = b >> 1;
  assign s = 6'd38 - (e_calc << 1);
  assign x_shifted = x << s;

  always @(posedge clk) begin
    if (~rst_n) begin
      e_s1 <= 6'b0;
      xn_s1 <= 40'b0;
      valid_s1 <= 1'b0;
      is_zero_s1 <= 1'b0;
    end else begin
      e_s1 <= e_calc;
      xn_s1 <= x_shifted;
      valid_s1 <= valid_in;
      is_zero_s1 <= is_zero_input;
    end
  end

  wire [7:0] idx_stage2;
  wire [16:0] rom_val;

  assign idx_stage2 = xn_s1[39:32];

  rsqrt_rom rsqrt_rom_inst(
    .idx(idx_stage2),
    .val(rom_val)
  );

  always @(posedge clk) begin
    if (~rst_n) begin
      rom_val_s2 <= 17'b0;
      e_s2 <= 6'b0;
      valid_s2 <= 1'b0;
      is_zero_s2 <= 1'b0;
    end else begin
      rom_val_s2 <= rom_val;
      e_s2 <= e_s1;
      valid_s2 <= valid_s1;
      is_zero_s2 <= is_zero_s1;
    end
  end

  wire [16:0] y_value;
  wire [5:0] e_value;

  assign y_value = is_zero_s2 ? 17'h1FFFF : rom_val_s2;
  assign e_value = is_zero_s2 ? 6'b0 : e_s2;

  always @(posedge clk) begin
    if (~rst_n) begin
      y_s3 <= 17'b0;
      e_s3 <= 6'b0;
      valid_s3 <= 1'b0;
    end else begin
      y_s3 <= y_value;
      e_s3 <= e_value;
      valid_s3 <= valid_s2;
    end
  end

  assign y = y_s3;
  assign e = e_s3;
  assign valid_out = valid_s3;

endmodule
