module rope (
  input clk,
  input rst_n,
  input signed [15:0] x1,
  input signed [15:0] x2,
  input [3:0] idx,
  input [7:0] pos,
  input valid_in,
  output signed [15:0] y1,
  output signed [15:0] y2,
  output valid_out
);

  wire [19:0] freq;
  rope_freq freq_lut (.idx(idx), .val(freq));

  wire [27:0] phase_prod;
  assign phase_prod = pos * freq;

  wire [21:0] phase;
  assign phase = phase_prod[21:0];

  wire [22:0] phase_plus_512;
  assign phase_plus_512 = {1'b0, phase} + 23'd512;

  wire [11:0] t;
  assign t = phase_plus_512[21:10];

  wire [11:0] t_cos;
  assign t_cos = (t + 12'd1024) & 12'hfff;

  wire signed [23:0] sin_t_comb, cos_t_comb;
  rope_sin sin_lut0 (.idx(t),     .val(sin_t_comb));
  rope_sin cos_lut  (.idx(t_cos), .val(cos_t_comb));

  // Stage 1: register inputs and sin/cos values
  reg signed [15:0] x1_p1, x2_p1;
  reg signed [23:0] sin_t_p1, cos_t_p1;
  reg valid_p1;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1_p1    <= 16'b0;
      x2_p1    <= 16'b0;
      sin_t_p1 <= 24'b0;
      cos_t_p1 <= 24'b0;
      valid_p1 <= 1'b0;
    end else begin
      x1_p1    <= x1;
      x2_p1    <= x2;
      sin_t_p1 <= sin_t_comb;
      cos_t_p1 <= cos_t_comb;
      valid_p1 <= valid_in;
    end
  end

  // Stage 1b: split 16x24 multiply into two 16x12 partial products
  // cos/sin split: upper 12 bits (signed), lower 12 bits (unsigned extended to 13)
  wire signed [11:0] cos_hi = cos_t_p1[23:12];
  wire signed [11:0] sin_hi = sin_t_p1[23:12];
  wire [11:0] cos_lo = cos_t_p1[11:0];
  wire [11:0] sin_lo = sin_t_p1[11:0];

  // 16-bit signed x 12-bit signed -> 28-bit signed
  wire signed [27:0] x1c_hi_w = x1_p1 * $signed(cos_hi);
  wire signed [27:0] x2s_hi_w = x2_p1 * $signed(sin_hi);
  wire signed [27:0] x2c_hi_w = x2_p1 * $signed(cos_hi);
  wire signed [27:0] x1s_hi_w = x1_p1 * $signed(sin_hi);

  // 16-bit signed x 13-bit signed-positive -> 29-bit signed
  wire signed [28:0] x1c_lo_w = x1_p1 * $signed({1'b0, cos_lo});
  wire signed [28:0] x2s_lo_w = x2_p1 * $signed({1'b0, sin_lo});
  wire signed [28:0] x2c_lo_w = x2_p1 * $signed({1'b0, cos_lo});
  wire signed [28:0] x1s_lo_w = x1_p1 * $signed({1'b0, sin_lo});

  reg signed [27:0] x1c_hi_p1b, x2s_hi_p1b, x2c_hi_p1b, x1s_hi_p1b;
  reg signed [28:0] x1c_lo_p1b, x2s_lo_p1b, x2c_lo_p1b, x1s_lo_p1b;
  reg valid_p1b;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1c_hi_p1b <= 28'b0;
      x2s_hi_p1b <= 28'b0;
      x2c_hi_p1b <= 28'b0;
      x1s_hi_p1b <= 28'b0;
      x1c_lo_p1b <= 29'b0;
      x2s_lo_p1b <= 29'b0;
      x2c_lo_p1b <= 29'b0;
      x1s_lo_p1b <= 29'b0;
      valid_p1b  <= 1'b0;
    end else begin
      x1c_hi_p1b <= x1c_hi_w;
      x2s_hi_p1b <= x2s_hi_w;
      x2c_hi_p1b <= x2c_hi_w;
      x1s_hi_p1b <= x1s_hi_w;
      x1c_lo_p1b <= x1c_lo_w;
      x2s_lo_p1b <= x2s_lo_w;
      x2c_lo_p1b <= x2c_lo_w;
      x1s_lo_p1b <= x1s_lo_w;
      valid_p1b  <= valid_p1;
    end
  end

  // Stage 2: reconstruct full 40-bit products and register
  // full = hi * 2^12 + lo  (sign-extend both to 40 bits)
  wire signed [39:0] x1_cos_w = $signed({x1c_hi_p1b, 12'b0}) + {{11{x1c_lo_p1b[28]}}, x1c_lo_p1b};
  wire signed [39:0] x2_sin_w = $signed({x2s_hi_p1b, 12'b0}) + {{11{x2s_lo_p1b[28]}}, x2s_lo_p1b};
  wire signed [39:0] x2_cos_w = $signed({x2c_hi_p1b, 12'b0}) + {{11{x2c_lo_p1b[28]}}, x2c_lo_p1b};
  wire signed [39:0] x1_sin_w = $signed({x1s_hi_p1b, 12'b0}) + {{11{x1s_lo_p1b[28]}}, x1s_lo_p1b};

  reg signed [39:0] x1_cos_p2, x2_sin_p2, x2_cos_p2, x1_sin_p2;
  reg valid_p2;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1_cos_p2 <= 40'b0;
      x2_sin_p2 <= 40'b0;
      x2_cos_p2 <= 40'b0;
      x1_sin_p2 <= 40'b0;
      valid_p2  <= 1'b0;
    end else begin
      x1_cos_p2 <= x1_cos_w;
      x2_sin_p2 <= x2_sin_w;
      x2_cos_p2 <= x2_cos_w;
      x1_sin_p2 <= x1_sin_w;
      valid_p2  <= valid_p1b;
    end
  end

  wire signed [40:0] y1_sum = x1_cos_p2 - x2_sin_p2 + 41'sd2097152;
  wire signed [40:0] y2_sum = x2_cos_p2 + x1_sin_p2 + 41'sd2097152;

  wire signed [18:0] y1_shifted = y1_sum[40:22];
  wire signed [18:0] y2_shifted = y2_sum[40:22];

  wire signed [15:0] y1_sat = (y1_shifted > 19'sd32767)  ? 16'sh7fff :
                              (y1_shifted < -19'sd32768) ? 16'sh8000 :
                              y1_shifted[15:0];

  wire signed [15:0] y2_sat = (y2_shifted > 19'sd32767)  ? 16'sh7fff :
                              (y2_shifted < -19'sd32768) ? 16'sh8000 :
                              y2_shifted[15:0];

  reg signed [15:0] y1_reg, y2_reg;
  reg valid_reg;

  always @(posedge clk) begin
    if (!rst_n) begin
      y1_reg    <= 16'b0;
      y2_reg    <= 16'b0;
      valid_reg <= 1'b0;
    end else begin
      y1_reg    <= y1_sat;
      y2_reg    <= y2_sat;
      valid_reg <= valid_p2;
    end
  end

  assign y1 = y1_reg;
  assign y2 = y2_reg;
  assign valid_out = valid_reg;

endmodule
