module rsqrt (
  input               clk,
  input               rst_n,
  input      [21:0] x,
  input               valid_in,
  output reg [16:0] y,
  output reg [4:0] e,
  output reg          valid_out
);
  // Normalise by an even number of bits so the halved exponent is an
  // integer, which makes the mantissa span [1,4) and the table index the
  // top bits of it directly. The consumer applies the shift, so the
  // mantissa stays full width.
  function [4:0] msb;
    input [21:0] v;
    integer i;
    begin
      msb = 0;
      for (i = 21; i >= 0; i = i - 1)
        if (v[i] && msb == 0) msb = i[4:0];
    end
  endfunction

  wire [4:0] e_w  = msb(x) >> 1;
  wire [4:0] s_w  = 20 - (e_w << 1);
  wire [21:0] xn_w = x << s_w;
  reg  [21:0] xn;
  wire [16:0] lut_out;
  rsqrt_rom rom (.idx(xn[21:14]), .val(lut_out));
  reg  [4:0] e1, e2;
  reg  [16:0] m;
  reg            vpipe, vpipe2;
  always @(posedge clk) begin
    if (!rst_n) begin
      xn        <= 0;
      e1        <= 0;
      e2        <= 0;
      m         <= 0;
      y         <= 0;
      e         <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      valid_out <= 1'b0;
    end else begin
      xn        <= xn_w;
      e1        <= e_w;
      vpipe     <= valid_in;
      m         <= lut_out;
      e2        <= e1;
      vpipe2    <= vpipe;
      y         <= m;
      e         <= e2;
      valid_out <= vpipe2;
    end
  end
endmodule
