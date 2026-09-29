module rmsnorm_hd (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [43:0] eps,
  input      [17:0] scale_o,
  input      [6:0] shift_o,
  output reg [6:0] x_addr,
  input      signed [15:0] x_data,
  output reg [6:0] g_addr,
  input      signed [15:0] g_data,
  output reg               o_valid,
  output reg [6:0] o_index,
  output reg signed [15:0] o_data,
  output reg               busy
);
  reg [43:0] ssq;
  reg [16:0] rs_m;
  reg [5:0] rs_e;
  reg [1:0] st;
  localparam S_IDLE = 2'd0, S_P1 = 2'd1, S_RS = 2'd2, S_P2 = 2'd3;

  // Issue counter shared by both passes. Every read is two edges from
  // issue to data: one for the address register, one for the memory.
  reg issuing;
  reg [6:0] ii;
  reg v0, v1, v2, l0, l1, l2;
  reg [6:0] i0, i1, i2, i3, i4, i5;
  reg v3, v4, u5;

  // Pass 1: squares into the sum.
  reg [31:0] sqr;
  // Pass 2: x*g, then times the mantissa as four partial products, then
  // the shift. Both factors are split in half: the whole product is
  // 2*dw+ow bits, and at 16-bit operands one multiply missed timing by
  // 1.54 ns and a split of the mantissa alone still by 0.40 ns.
  reg signed [31:0] pr;
  reg signed [49:0] p00, p01, p10, p11, s0, s1, pm;

  reg rs_vin, rs_go;
  wire [16:0] rs_y;
  wire [5:0] rs_eo;
  wire rs_vout;
  rsqrt rs (.clk(clk), .rst_n(rst_n), .x(ssq), .valid_in(rs_vin),
            .y(rs_y), .e(rs_eo), .valid_out(rs_vout));

  reg  rq_vin;
  reg  signed [35:0] rq_acc;
  reg  [6:0] rq_idx;
  wire signed [15:0] rq_q;
  wire rq_sat, rq_vout;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc),
              .scale(scale_o), .shift(shift_o), .valid_in(rq_vin),
              .q_out(rq_q), .sat(rq_sat), .valid_out(rq_vout));
  reg [6:0] idx_pipe [0:10];
  reg [7:0] ocnt;
  wire signed [49:0] shv = pm >>> (rs_e + 6'd0);
  integer n;

  always @(posedge clk) begin
    if (!rst_n) begin
      ssq <= 0; rs_m <= 0; rs_e <= 0; st <= S_IDLE; busy <= 1'b0;
      issuing <= 1'b0; ii <= 0; x_addr <= 0; g_addr <= 0;
      v0 <= 1'b0; v1 <= 1'b0; v2 <= 1'b0; l0 <= 1'b0; l1 <= 1'b0;
      l2 <= 1'b0; i0 <= 0; i1 <= 0; i2 <= 0; i3 <= 0; i4 <= 0;
      i5 <= 0; v3 <= 1'b0; v4 <= 1'b0; u5 <= 1'b0; sqr <= 0; pr <= 0;
      p00 <= 0; p01 <= 0; p10 <= 0; p11 <= 0; s0 <= 0; s1 <= 0;
      pm <= 0; rs_vin <= 1'b0; rs_go <= 1'b0;
      rq_vin <= 1'b0; rq_acc <= 0; rq_idx <= 0; ocnt <= 0;
      o_valid <= 1'b0; o_index <= 0; o_data <= 0;
      for (n = 0; n <= 10; n = n + 1) idx_pipe[n] <= 0;
    end else begin
      rs_vin  <= 1'b0;
      rq_vin  <= 1'b0;
      o_valid <= 1'b0;
      idx_pipe[0] <= rq_idx;
      for (n = 1; n <= 10; n = n + 1) idx_pipe[n] <= idx_pipe[n-1];

      v0 <= 1'b0;
      if (issuing) begin
        x_addr <= ii; g_addr <= ii;
        v0 <= 1'b1; l0 <= (ii == 127); i0 <= ii;
        if (ii == 127) begin issuing <= 1'b0; ii <= 0; end
        else ii <= ii + 1;
      end
      v1 <= v0; l1 <= l0; i1 <= i0;
      v2 <= v1; l2 <= l1; i2 <= i1;
      if (v1) begin
        sqr <= x_data * x_data;
        pr  <= x_data * g_data;
      end
      v3 <= v2; i3 <= i2;
      if (v2) begin
        p00 <= $signed({1'b0, pr[15:0]}) * $signed({1'b0, rs_m[8:0]});
        p01 <= $signed({1'b0, pr[15:0]}) * $signed({1'b0, rs_m[16:9]});
        p10 <= $signed(pr[31:16]) * $signed({1'b0, rs_m[8:0]});
        p11 <= $signed(pr[31:16]) * $signed({1'b0, rs_m[16:9]});
      end
      v4 <= v3; i4 <= i3;
      if (v3) begin
        s0 <= p00 + (p01 <<< 9);
        s1 <= p10 + (p11 <<< 9);
      end
      u5 <= (st == S_P2) && v4; i5 <= i4;
      if (v4) pm <= s0 + (s1 <<< 16);

      case (st)
        S_IDLE: if (start) begin
          st <= S_P1; busy <= 1'b1; ssq <= eps; issuing <= 1'b1;
          ii <= 0; ocnt <= 0;
        end
        S_P1: if (v2) begin
          ssq <= ssq + sqr;
          if (l2) begin st <= S_RS; rs_go <= 1'b1; end
        end
        S_RS: begin
          if (rs_go) begin rs_vin <= 1'b1; rs_go <= 1'b0; end
          if (rs_vout) begin
            rs_m <= rs_y; rs_e <= rs_eo; st <= S_P2; issuing <= 1'b1;
            ii <= 0;
          end
        end
        S_P2: begin
          if (u5) begin
            rq_acc <= shv[35:0]; rq_idx <= i5; rq_vin <= 1'b1;
          end
          if (rq_vout) begin
            o_valid <= 1'b1; o_index <= idx_pipe[10]; o_data <= rq_q;
            ocnt <= ocnt + 1;
          end
          if (ocnt == 128) begin st <= S_IDLE; busy <= 1'b0; end
        end
      endcase
    end
  end
endmodule
