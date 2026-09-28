module rope (
  input                    clk,
  input                    rst_n,
  input      signed [7:0] x1,
  input      signed [7:0] x2,
  input      [2:0] idx,
  input      [4:0] pos,
  input                    valid_in,
  output reg signed [7:0] y1,
  output reg signed [7:0] y2,
  output reg               valid_out
);
  // Stage 0: the angle, in turns, modulo one turn.
  wire [16:0] f;
  rope_freq fq (.idx(idx), .val(f));
  wire [21:0] turn = pos * f;
  reg [18:0] ph0;
  reg signed [7:0] a0, b0;
  reg v0;

  // Stage 1: the nearest table entry, and a quarter turn on for cosine.
  wire [19:0] rph = {1'b0, ph0} + 20'd64;
  wire [11:0] ts = rph[18:7];
  wire [11:0] tc = ts + 12'd1024;
  wire [15:0] s_raw, c_raw;
  rope_sin rs (.idx(ts), .val(s_raw));
  rope_sin rc (.idx(tc), .val(c_raw));
  reg signed [15:0] s1, c1;
  reg signed [7:0] a1, b1;
  reg v1;

  // Stage 2: the four products.
  reg signed [23:0] p_xc1, p_ys, p_yc2, p_xs;
  reg v2;

  // Stage 3: sum, round half up, shift, saturate.
  wire signed [24:0] sum1 = p_xc1 - p_ys + 25'sd8192;
  wire signed [24:0] sum2 = p_yc2 + p_xs + 25'sd8192;
  wire signed [24:0] r1 = sum1 >>> 14;
  wire signed [24:0] r2 = sum2 >>> 14;

  always @(posedge clk) begin
    if (!rst_n) begin
      ph0 <= 0; a0 <= 0; b0 <= 0; v0 <= 1'b0;
      s1 <= 0; c1 <= 0; a1 <= 0; b1 <= 0; v1 <= 1'b0;
      p_xc1 <= 0; p_ys <= 0; p_yc2 <= 0; p_xs <= 0; v2 <= 1'b0;
      y1 <= 0; y2 <= 0; valid_out <= 1'b0;
    end else begin
      ph0 <= turn[18:0]; a0 <= x1; b0 <= x2; v0 <= valid_in;
      s1 <= s_raw; c1 <= c_raw; a1 <= a0; b1 <= b0; v1 <= v0;
      p_xc1 <= a1 * c1; p_ys <= b1 * s1; p_yc2 <= b1 * c1; p_xs <= a1 * s1;
      v2 <= v1;
      y1 <= (r1 > 127) ? 8'sd127 : (r1 < -128) ? -8'sd128 : r1[7:0];
      y2 <= (r2 > 127) ? 8'sd127 : (r2 < -128) ? -8'sd128 : r2[7:0];
      valid_out <= v2;
    end
  end
endmodule
