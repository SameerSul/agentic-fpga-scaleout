module attnn (
  input                    clk,
  input                    rst_n,
  input                    load_valid,
  input      signed [15:0] load_data,
  input                    start,
  input      [8:0] n,
  input      [4:0]  shift_s,
  input      [17:0] scale_o,
  input      [6:0] shift_o,
  output     [9:0] k_addr,
  input      [255:0] k_data,
  output     [9:0] v_addr,
  input      [255:0] v_data,
  output reg               o_valid,
  output reg [5:0] o_index,
  output reg signed [15:0] o_data,
  output reg               busy
);
  reg signed [15:0] qbuf [0:63];
  reg signed [20:0] sbuf [0:255];
  reg        [15:0] pbuf [0:255];
  reg [6:0] lptr;
  reg [8:0] n_r;
  reg [1:0] st;
  localparam S_IDLE = 2'd0, S_SCORE = 2'd1, S_SOFT = 2'd2, S_WSUM = 2'd3;

  // ---- scores: lane l is position g0 + l
  reg issuing;
  reg [6:0] r;
  reg [8:0] g0;
  reg [9:0] kbase;
  assign k_addr = kbase + r;
  reg signed [15:0] a_data;
  always @(posedge clk) a_data <= qbuf[r[5:0]];
  reg v1;
  reg [4:0] lastp;
  reg [8:0] gp [0:4];
  reg mclr;
  wire signed [37:0] acc [0:15];
  genvar l;
  generate
    for (l = 0; l < 16; l = l + 1) begin : slane
      mac_s mc (.clk(clk), .rst_n(rst_n), .clear(mclr), .a(a_data),
              .b(k_data[16*l +: 16]), .valid_in(v1), .acc(acc[l]), .valid_out());
    end
  endgenerate
  reg signed [37:0] shadow [0:15];
  reg [8:0] cap0;
  reg [4:0] dj, dn;
  reg pend;
  wire [8:0] next0 = g0 + 16;

  // Score quantizer, as the one-lane head's: round, shift, clamp.
  wire signed [42:0] rnd_s = (shift_s == 5'd0) ? 43'sd0
                                : (43'sd1 <<< (shift_s - 5'd1));
  reg  signed [42:0] rnd_r;
  reg  sv1;
  reg  signed [42:0] st1;
  reg  [7:0] si1;
  wire signed [42:0] shv = st1 >>> shift_s;
  reg  [8:0] scnt;

  // ---- the softmax, one lane
  reg  sm_start, sm_ran;
  wire [7:0] sm_saddr, sm_wi;
  reg  signed [20:0] sm_sdata;
  wire sm_wv, sm_busy;
  wire [15:0] sm_wd;
  always @(posedge clk) sm_sdata <= sbuf[sm_saddr];
  softmax sm (.clk(clk), .rst_n(rst_n), .start(sm_start), .n(n_r),
              .s_addr(sm_saddr), .s_data(sm_sdata), .w_valid(sm_wv),
              .w_index(sm_wi), .w_data(sm_wd), .busy(sm_busy));
  reg  [8:0] pcnt;

  // ---- weighted sum: lane l is dimension dg * 16 + l
  reg wiss;
  reg [8:0] jj;
  reg [1:0] dg;
  reg [9:0] vaddr;
  assign v_addr = vaddr;
  reg [15:0] p_r;
  always @(posedge clk) p_r <= pbuf[jj[7:0]];
  reg u1, f1, uA, fA;
  reg [3:0] lastw;
  reg [1:0] dgp [0:3];
  reg signed [32:0] prod [0:15];
  reg signed [31:0] accw [0:15];
  reg signed [31:0] shadow2 [0:15];
  reg [1:0] capd;
  reg [4:0] dj2, dn2;
  reg pend2;
  wire signed [31:0] x = shadow2[dj2[3:0]];

  reg  rq_vin;
  reg  signed [36:0] rq_acc;
  reg  [5:0] rq_idx;
  wire signed [15:0] rq_q;
  wire rq_sat, rq_vout;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc),
              .scale(scale_o), .shift(shift_o), .valid_in(rq_vin),
              .q_out(rq_q), .sat(rq_sat), .valid_out(rq_vout));
  reg [5:0] idx_pipe [0:12];
  reg [6:0] ocnt;
  integer k;

  always @(posedge clk) begin
    if (!rst_n) begin
      lptr <= 0; n_r <= 0; st <= S_IDLE; busy <= 1'b0; issuing <= 1'b0;
      r <= 0; g0 <= 0; kbase <= 0; v1 <= 1'b0; lastp <= 0; mclr <= 1'b0;
      cap0 <= 0; dj <= 0; dn <= 0; pend <= 1'b0; rnd_r <= 0; sv1 <= 1'b0;
      st1 <= 0; si1 <= 0; scnt <= 0; sm_start <= 1'b0; sm_ran <= 1'b0;
      pcnt <= 0; wiss <= 1'b0; jj <= 0; dg <= 0; vaddr <= 0; u1 <= 1'b0;
      f1 <= 1'b0; uA <= 1'b0; fA <= 1'b0; lastw <= 0; capd <= 0; dj2 <= 0;
      dn2 <= 0; pend2 <= 1'b0; rq_vin <= 1'b0; rq_acc <= 0; rq_idx <= 0;
      ocnt <= 0; o_valid <= 1'b0; o_index <= 0; o_data <= 0;
      for (k = 0; k <= 4; k = k + 1) gp[k] <= 0;
      for (k = 0; k < 4; k = k + 1) dgp[k] <= 0;
      for (k = 0; k <= 12; k = k + 1) idx_pipe[k] <= 0;
    end else begin
      mclr <= 1'b0; sm_start <= 1'b0; rq_vin <= 1'b0; o_valid <= 1'b0;
      sv1 <= 1'b0;
      if (load_valid) begin
        qbuf[lptr[5:0]] <= load_data; lptr <= lptr + 1;
      end

      // Scores.
      v1 <= issuing;
      lastp <= {lastp[3:0], issuing && r == 63};
      gp[0] <= g0;
      for (k = 1; k <= 4; k = k + 1) gp[k] <= gp[k-1];
      if (issuing) begin
        if (r == 63) begin issuing <= 1'b0; r <= 0; end
        else r <= r + 1;
      end
      if (lastp[4] || pend) begin
        if (dj == dn) begin
          pend <= 1'b0;
          for (k = 0; k < 16; k = k + 1) shadow[k] <= acc[k];
          cap0 <= gp[4]; dj <= 0;
          dn <= (n_r - gp[4] < 16) ? n_r - gp[4] : 16;
          mclr <= 1'b1;
          if (next0 < n_r) begin
            g0 <= next0; kbase <= kbase + 64; issuing <= 1'b1;
          end
        end else pend <= 1'b1;
      end
      if (dj != dn) begin
        st1 <= {shadow[dj[3:0]][37], shadow[dj[3:0]], 4'd0} + rnd_r; si1 <= cap0 + dj; sv1 <= 1'b1; dj <= dj + 1;
      end
      if (sv1) begin
        if (shv > 43'sd1048575) sbuf[si1] <= 21'sd1048575;
        else if (shv < -43'sd1048576) sbuf[si1] <= -21'sd1048576;
        else sbuf[si1] <= shv[20:0];
        scnt <= scnt + 1;
      end

      // Weights.
      if (sm_wv) begin pbuf[sm_wi] <= sm_wd; pcnt <= pcnt + 1; end

      // Weighted sum.
      u1 <= wiss; f1 <= wiss && jj == 0;
      uA <= u1; fA <= f1;
      lastw <= {lastw[2:0], wiss && jj == n_r - 1};
      dgp[0] <= dg;
      for (k = 1; k < 4; k = k + 1) dgp[k] <= dgp[k-1];
      for (k = 0; k < 16; k = k + 1) begin
        if (u1) prod[k] <= $signed({1'b0, p_r}) * $signed(v_data[k*16 +: 16]);
        if (uA) accw[k] <= fA ? prod[k] : accw[k] + prod[k];
      end
      if (wiss) begin
        if (jj == n_r - 1) wiss <= 1'b0;
        else begin jj <= jj + 1; vaddr <= vaddr + 4; end
      end
      if (lastw[3] || pend2) begin
        if (dj2 == dn2) begin
          pend2 <= 1'b0;
          for (k = 0; k < 16; k = k + 1) shadow2[k] <= accw[k];
          capd <= dgp[3]; dj2 <= 0; dn2 <= 16;
          if (dg + 1 < 4) begin
            dg <= dg + 1; jj <= 0; vaddr <= dg + 1; wiss <= 1'b1;
          end
        end else pend2 <= 1'b1;
      end
      if (dj2 != dn2) begin
        rq_acc <= {{5{x[31]}}, x}; rq_idx <= capd * 16 + dj2; rq_vin <= 1'b1;
        dj2 <= dj2 + 1;
      end
      idx_pipe[0] <= rq_idx;
      for (k = 1; k <= 12; k = k + 1) idx_pipe[k] <= idx_pipe[k-1];
      if (rq_vout) begin
        o_valid <= 1'b1; o_index <= idx_pipe[12]; o_data <= rq_q;
        ocnt <= ocnt + 1;
      end

      case (st)
        S_IDLE: if (start) begin
          busy <= 1'b1; n_r <= n; st <= S_SCORE; rnd_r <= rnd_s;
          issuing <= 1'b1; r <= 0; g0 <= 0; kbase <= 0; mclr <= 1'b1;
          dj <= 0; dn <= 0; pend <= 1'b0; scnt <= 0; pcnt <= 0; ocnt <= 0;
        end
        S_SCORE: if (scnt == n_r && !sv1 && dj == dn) begin
          sm_start <= 1'b1; sm_ran <= 1'b0; st <= S_SOFT;
        end
        S_SOFT: begin
          if (sm_busy) sm_ran <= 1'b1;
          if (sm_ran && !sm_busy && pcnt == n_r) begin
            st <= S_WSUM; wiss <= 1'b1; jj <= 0; dg <= 0; vaddr <= 0;
            dj2 <= 0; dn2 <= 0; pend2 <= 1'b0;
          end
        end
        S_WSUM: if (ocnt == 64) begin
          st <= S_IDLE; busy <= 1'b0; lptr <= 0;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
