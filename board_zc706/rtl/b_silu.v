module silu (
  input                    clk,
  input                    rst_n,
  input      signed [12:0] x,
  input                    valid_in,
  output reg signed [12:0] y,
  output reg               valid_out
);
  // Stage 0: |x|, clamped so that -|x| fits the exponential's input.
  reg signed [12:0] na;
  reg v0, neg0;
  reg signed [12:0] x0;
  wire [12:0] ax = x[12] ? -x : x;
  wire [15:0] e_y;
  wire e_v;
  expu ex (.clk(clk), .rst_n(rst_n), .x(na), .valid_in(v0), .y(e_y),
           .valid_out(e_v));

  // x and its sign travel alongside the exponential.
  reg signed [12:0] xe [0:2];
  reg nege [0:2];

  // 1 + e into the reciprocal; e, x and the sign travel alongside it.
  reg [23:0] dd;
  reg vd;
  reg [15:0] e_r [0:2];
  reg signed [12:0] x_r [0:2];
  reg neg_r [0:2];
  reg [15:0] ed;
  reg signed [12:0] xd;
  reg negd;
  wire [16:0] r_m;
  wire [4:0] r_k;
  wire r_v;
  recip rc (.clk(clk), .rst_n(rst_n), .x(dd), .valid_in(vd), .y(r_m),
            .k(r_k), .valid_out(r_v));

  // The sigmoid, then the product.
  reg [33:0] prod;
  reg [4:0] k1;
  reg v1;
  reg signed [12:0] x1;
  reg [15:0] sig;
  reg v2;
  reg signed [12:0] x2;
  wire [33:0] sh = prod >> (25 - k1);
  wire signed [29:0] xy = x2 * $signed({1'b0, sig});
  integer n;

  always @(posedge clk) begin
    if (!rst_n) begin
      na <= 0; v0 <= 1'b0; neg0 <= 1'b0; x0 <= 0; dd <= 0; vd <= 1'b0;
      ed <= 0; xd <= 0; negd <= 1'b0; prod <= 0; k1 <= 0; v1 <= 1'b0;
      x1 <= 0; sig <= 0; v2 <= 1'b0; x2 <= 0; y <= 0; valid_out <= 1'b0;
      for (n = 0; n <= 2; n = n + 1) begin xe[n] <= 0; nege[n] <= 1'b0; end
      for (n = 0; n <= 2; n = n + 1) begin
        e_r[n] <= 0; x_r[n] <= 0; neg_r[n] <= 1'b0;
      end
    end else begin
      v0 <= valid_in; x0 <= x; neg0 <= x[12];
      na <= (x == {1'b1, 12'd0}) ? -13'sd4095 : -$signed(ax);

      xe[0] <= x0; nege[0] <= neg0;
      for (n = 1; n <= 2; n = n + 1) begin
        xe[n] <= xe[n-1]; nege[n] <= nege[n-1];
      end

      vd <= e_v;
      dd <= 24'd32768 + e_y;
      ed <= e_y; xd <= xe[2]; negd <= nege[2];
      e_r[0] <= ed; x_r[0] <= xd; neg_r[0] <= negd;
      for (n = 1; n <= 2; n = n + 1) begin
        e_r[n] <= e_r[n-1]; x_r[n] <= x_r[n-1]; neg_r[n] <= neg_r[n-1];
      end

      v1 <= r_v; k1 <= r_k; x1 <= x_r[2];
      prod <= (neg_r[2] ? {1'b0, e_r[2]} : 17'd32768) * r_m;

      v2 <= v1; x2 <= x1;
      sig <= (sh > 34'd32768) ? 16'd32768 : sh[15:0];

      valid_out <= v2;
      y <= xy >>> 15;
    end
  end
endmodule
