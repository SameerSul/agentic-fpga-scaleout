module expu (
  input                     clk,
  input                     rst_n,
  input      signed [12:0] x,
  input                     valid_in,
  output reg        [15:0] y,
  output reg                valid_out
);
  // exp(x) = 2**(x*log2(e)); the fraction of x*log2(e) indexes a table of
  // 2**f and its integer part is a right shift. x is non-positive, which
  // softmax guarantees by subtracting the row maximum, so the shift only
  // ever goes right.
  reg signed [30:0] t;
  reg        [15:0] m;
  reg        [4:0] sh;
  reg                  vpipe, vpipe2;
  wire [15:0] lut_out;
  exp_rom rom (.idx(t[7:0]), .val(lut_out));
  wire signed [30:0] prod = x * $signed({1'b0, 18'd94548});
  wire signed [30:0] tt   = prod >>> 16;
  wire signed [30:0] n    = t >>> 8;
  always @(posedge clk) begin
    if (!rst_n) begin
      t         <= 0;
      m         <= 0;
      sh        <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      y         <= 0;
      valid_out <= 1'b0;
    end else begin
      t         <= tt;
      vpipe     <= valid_in;
      m         <= lut_out;
      // Clamping the shift is what drives an underflowing input to zero:
      // the table entry is 16 bits, so a shift of 16 empties it. A
      // separate flush flag was redundant.
      sh        <= (-n) > 15 ? 16 : (-n);
      vpipe2    <= vpipe;
      y         <= m >> sh;
      valid_out <= vpipe2;
    end
  end
endmodule
