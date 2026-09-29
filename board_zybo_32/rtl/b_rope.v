module rope (
  input                    clk,
  input                    rst_n,
  input      signed [15:0] x1,
  input      signed [15:0] x2,
  input      [4:0] idx,
  input      [7:0] pos,
  input                    valid_in,
  output reg signed [15:0] y1,
  output reg signed [15:0] y2,
  output reg               valid_out
);
  // Stage 0: the angle, in turns, modulo one turn.
  wire [19:0] f;
  rope_freq fq (.idx(idx), .val(f));
  wire [27:0] turn = pos * f;
  reg [21:0] ph0;
  reg signed [15:0] a0, b0;
  reg v0;

  // Stage 1: the nearest table entry, and a quarter turn on for cosine.
  wire [22:0] rph = {1'b0, ph0} + 23'd512;
  wire [11:0] ts = rph[21:10];
  wire [11:0] tc = ts + 12'd1024;
  wire [23:0] s_raw, c_raw;
  rope_sin rs (.idx(ts), .val(s_raw));
  rope_sin rc (.idx(tc), .val(c_raw));
  reg signed [23:0] s1, c1;
  reg signed [15:0] a1, b1;
  reg v1;

  // Stage 2: the four products.
  reg signed [39:0] p_xc1, p_ys, p_yc2, p_xs;
  reg v2;
  reg signed [39:0] q_xc1h, q_xc1l, q_ysh, q_ysl, q_yc2h, q_yc2l,
                     q_xsh, q_xsl;
  reg v2a;
  wire signed [11:0] c_hi = c1[23:12], s_hi = s1[23:12];
  wire signed [12:0] c_lo = {1'b0, c1[11:0]}, s_lo = {1'b0, s1[11:0]};

  // Stage 3: sum, round half up, shift, saturate.
  wire signed [40:0] sum1 = p_xc1 - p_ys + 41'sd2097152;
  wire signed [40:0] sum2 = p_yc2 + p_xs + 41'sd2097152;
  wire signed [40:0] r1 = sum1 >>> 22;
  wire signed [40:0] r2 = sum2 >>> 22;

  always @(posedge clk) begin
    if (!rst_n) begin
      ph0 <= 0; a0 <= 0; b0 <= 0; v0 <= 1'b0;
      s1 <= 0; c1 <= 0; a1 <= 0; b1 <= 0; v1 <= 1'b0;
      p_xc1 <= 0; p_ys <= 0; p_yc2 <= 0; p_xs <= 0; v2 <= 1'b0;
      q_xc1h <= 0; q_xc1l <= 0; q_ysh <= 0; q_ysl <= 0; q_yc2h <= 0;
      q_yc2l <= 0; q_xsh <= 0; q_xsl <= 0; v2a <= 1'b0;
      y1 <= 0; y2 <= 0; valid_out <= 1'b0;
    end else begin
      ph0 <= turn[21:0]; a0 <= x1; b0 <= x2; v0 <= valid_in;
      s1 <= s_raw; c1 <= c_raw; a1 <= a0; b1 <= b0; v1 <= v0;
      q_xc1h <= a1 * c_hi; q_xc1l <= a1 * c_lo;
      q_ysh <= b1 * s_hi; q_ysl <= b1 * s_lo;
      q_yc2h <= b1 * c_hi; q_yc2l <= b1 * c_lo;
      q_xsh <= a1 * s_hi; q_xsl <= a1 * s_lo;
      v2a <= v1;
      p_xc1 <= (q_xc1h <<< 12) + q_xc1l; p_ys <= (q_ysh <<< 12) + q_ysl;
      p_yc2 <= (q_yc2h <<< 12) + q_yc2l; p_xs <= (q_xsh <<< 12) + q_xsl;
      v2 <= v2a;
      y1 <= (r1 > 32767) ? 16'sd32767 : (r1 < -32768) ? -16'sd32768 : r1[15:0];
      y2 <= (r2 > 32767) ? 16'sd32767 : (r2 < -32768) ? -16'sd32768 : r2[15:0];
      valid_out <= v2;
    end
  end
endmodule
