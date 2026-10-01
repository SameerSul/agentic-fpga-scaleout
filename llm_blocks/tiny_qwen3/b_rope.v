module rope(
  input clk,
  input rst_n,
  input signed [15:0] x1,
  input signed [15:0] x2,
  input [3:0] idx,
  input [7:0] pos,
  input valid_in,
  output reg signed [15:0] y1,
  output reg signed [15:0] y2,
  output reg valid_out
);

  wire [19:0] freq_val;
  rope_freq freq_inst(.idx(idx), .val(freq_val));

  wire [27:0] phase_full = {20'b0, pos} * {8'b0, freq_val};
  wire [21:0] phase      = phase_full[21:0];
  wire [22:0] phase_r    = {1'b0, phase} + 23'd512;
  wire [11:0] t_s        = phase_r[21:10];
  wire [11:0] t_c        = t_s + 12'd1024;

  // P1: register table indices and data
  reg signed [15:0] x1_p1, x2_p1;
  reg [11:0] ts_p1, tc_p1;
  reg v_p1;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1_p1<=0; x2_p1<=0; ts_p1<=0; tc_p1<=0; v_p1<=0;
    end else begin
      x1_p1<=x1; x2_p1<=x2; ts_p1<=t_s; tc_p1<=t_c; v_p1<=valid_in;
    end
  end

  wire [23:0] sin_lut, cos_lut;
  rope_sin sin_inst(.idx(ts_p1), .val(sin_lut));
  rope_sin cos_inst(.idx(tc_p1), .val(cos_lut));

  // P2: register x, and split 24-bit coefs into signed hi (12->13b) and unsigned lo (12b)
  reg signed [15:0] x1_p2, x2_p2;
  reg signed [12:0] sh_p2, ch_p2;
  reg [11:0]        sl_p2, cl_p2;
  reg v_p2;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1_p2<=0; x2_p2<=0;
      sh_p2<=0; ch_p2<=0; sl_p2<=0; cl_p2<=0;
      v_p2<=0;
    end else begin
      x1_p2<=x1_p1; x2_p2<=x2_p1;
      sh_p2<=$signed(sin_lut[23:12]); ch_p2<=$signed(cos_lut[23:12]);
      sl_p2<=sin_lut[11:0];           cl_p2<=cos_lut[11:0];
      v_p2<=v_p1;
    end
  end

  // 8 partial multiplies: 16s x 13s = 29s, each fits in one DSP
  wire signed [28:0] m_x1ch = x1_p2 * ch_p2;
  wire signed [28:0] m_x2sh = x2_p2 * sh_p2;
  wire signed [28:0] m_x2ch = x2_p2 * ch_p2;
  wire signed [28:0] m_x1sh = x1_p2 * sh_p2;
  wire signed [28:0] m_x1cl = x1_p2 * $signed({1'b0, cl_p2});
  wire signed [28:0] m_x2sl = x2_p2 * $signed({1'b0, sl_p2});
  wire signed [28:0] m_x2cl = x2_p2 * $signed({1'b0, cl_p2});
  wire signed [28:0] m_x1sl = x1_p2 * $signed({1'b0, sl_p2});

  // P3: register partial products
  reg signed [28:0] x1ch_p3, x2sh_p3, x2ch_p3, x1sh_p3;
  reg signed [28:0] x1cl_p3, x2sl_p3, x2cl_p3, x1sl_p3;
  reg v_p3;

  always @(posedge clk) begin
    if (!rst_n) begin
      x1ch_p3<=0; x2sh_p3<=0; x2ch_p3<=0; x1sh_p3<=0;
      x1cl_p3<=0; x2sl_p3<=0; x2cl_p3<=0; x1sl_p3<=0;
      v_p3<=0;
    end else begin
      x1ch_p3<=m_x1ch; x2sh_p3<=m_x2sh; x2ch_p3<=m_x2ch; x1sh_p3<=m_x1sh;
      x1cl_p3<=m_x1cl; x2sl_p3<=m_x2sl; x2cl_p3<=m_x2cl; x1sl_p3<=m_x1sl;
      v_p3<=v_p2;
    end
  end

  // Combine: x*coef = (x*coef_hi)<<12 + x*coef_lo, full 22 frac bits
  wire signed [40:0] x1c = ({{12{x1ch_p3[28]}}, x1ch_p3} << 12) + {{12{x1cl_p3[28]}}, x1cl_p3};
  wire signed [40:0] x2s = ({{12{x2sh_p3[28]}}, x2sh_p3} << 12) + {{12{x2sl_p3[28]}}, x2sl_p3};
  wire signed [40:0] x2c = ({{12{x2ch_p3[28]}}, x2ch_p3} << 12) + {{12{x2cl_p3[28]}}, x2cl_p3};
  wire signed [40:0] x1s = ({{12{x1sh_p3[28]}}, x1sh_p3} << 12) + {{12{x1sl_p3[28]}}, x1sl_p3};

  // Sum with round-half-up (+2^21) then arithmetic shift right 22
  wire signed [41:0] y1_raw = x1c - x2s + 42'sd2097152;
  wire signed [41:0] y2_raw = x2c + x1s + 42'sd2097152;

  wire signed [19:0] y1_sh = y1_raw[41:22];
  wire signed [19:0] y2_sh = y2_raw[41:22];

  // P4: saturate to 16 bits and output
  always @(posedge clk) begin
    if (!rst_n) begin
      y1<=0; y2<=0; valid_out<=0;
    end else begin
      valid_out <= v_p3;
      if (y1_sh[19:15] == {5{y1_sh[15]}})
        y1 <= y1_sh[15:0];
      else
        y1 <= y1_sh[19] ? 16'sh8000 : 16'sh7FFF;
      if (y2_sh[19:15] == {5{y2_sh[15]}})
        y2 <= y2_sh[15:0];
      else
        y2 <= y2_sh[19] ? 16'sh8000 : 16'sh7FFF;
    end
  end

endmodule
