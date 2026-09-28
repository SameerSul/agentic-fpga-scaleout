module recip (
  input               clk,
  input               rst_n,
  input      [19:0] x,
  input               valid_in,
  output reg [16:0] y,
  output reg [4:0] k,
  output reg          valid_out
);
  // Normalise x into [1,2) so a single table covers every input, then
  // hand back that table entry and the normalisation count. The consumer
  // folds the shift into the multiply it was going to do anyway, which
  // keeps the mantissa at full width.
  function [4:0] lzc;
    input [19:0] v;
    integer i;
    begin
      lzc = 20;
      for (i = 19; i >= 0; i = i - 1)
        if (v[i] && lzc == 20) lzc = 19 - i;
    end
  endfunction

  wire [4:0] k_w  = lzc(x);
  wire [19:0] xn_w = x << k_w;
  reg  [19:0] xn;
  wire [16:0] lut_out;
  recip_rom rom (.idx(xn[18:11]), .val(lut_out));
  reg  [4:0] k1, k2;
  reg  [16:0] m;
  reg            vpipe, vpipe2;
  always @(posedge clk) begin
    if (!rst_n) begin
      xn        <= 0;
      k1        <= 0;
      k2        <= 0;
      m         <= 0;
      y         <= 0;
      k         <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      valid_out <= 1'b0;
    end else begin
      xn        <= xn_w;
      k1        <= k_w;
      vpipe     <= valid_in;
      m         <= lut_out;
      k2        <= k1;
      vpipe2    <= vpipe;
      y         <= m;
      k         <= k2;
      valid_out <= vpipe2;
    end
  end
endmodule
