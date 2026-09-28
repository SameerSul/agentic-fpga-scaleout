module attn (
  input                    clk,
  input                    rst_n,
  input                    load_valid,
  input      signed [7:0] load_data,
  input                    start,
  input      [5:0] n,
  input      [4:0]  shift_s,
  input      [17:0] scale_o,
  input      [6:0] shift_o,
  output     [8:0] k_addr,
  input      signed [7:0] k_data,
  output reg [8:0] v_addr,
  input      signed [7:0] v_data,
  output reg               o_valid,
  output reg [3:0] o_index,
  output reg signed [7:0] o_data,
  output reg               busy
);
  reg signed [7:0] qbuf [0:15];
  reg signed [20:0] sbuf [0:31];
  reg        [15:0] pbuf [0:31];
  reg [3:0] lptr;
  reg [5:0] n_r;
  reg [2:0] st;
  localparam S_IDLE = 3'd0, S_SCORE = 3'd1, S_SOFT = 3'd2, S_OUT = 3'd3;

  // Scores: matvec walks the key cache as a matrix of n columns of
  // head_dim, so its weight address is the key address.
  reg  mv_start;
  wire [6:0] mv_a_addr;
  wire [13:0] mv_w_addr;
  wire mv_valid, mv_clear, mv_colv, mv_busy;
  wire [6:0] mv_coli;
  assign k_addr = mv_w_addr[8:0];
  reg signed [7:0] a_data;
  always @(posedge clk) a_data <= qbuf[mv_a_addr[3:0]];
  matvec mv (.clk(clk), .rst_n(rst_n), .start(mv_start),
             .depth(7'd16), .cols({{1{1'b0}}, n_r}),
             .a_addr(mv_a_addr), .w_addr(mv_w_addr),
             .mac_valid(mv_valid), .mac_clear(mv_clear),
             .col_valid(mv_colv), .col_index(mv_coli), .busy(mv_busy));
  wire signed [23:0] acc;
  wire mac_vout;
  mac mc (.clk(clk), .rst_n(rst_n), .clear(mv_clear), .a(a_data),
          .b(k_data), .valid_in(mv_valid), .acc(acc),
          .valid_out(mac_vout));

  // Score quantizer: round, shift, clamp. Two stages, so the rounding
  // add and the variable shift are not in the same cycle.
  wire signed [28:0] rnd_s = (shift_s == 5'd0) ? 29'sd0
                                : (29'sd1 <<< (shift_s - 5'd1));
  // shift_s is held for the whole run, so its rounding constant is
  // registered once at start. Built from the port every cycle, the
  // shifter sat in front of the rounding add and missed timing by 0.64 ns
  // at 16-bit operands, where the add is 47 bits.
  reg  signed [28:0] rnd_r;
  reg  sv1;
  reg  signed [28:0] st1;
  reg  [4:0] si1;
  wire signed [28:0] shv = st1 >>> shift_s;
  reg  [5:0] scnt;

  // Weights: the softmax block reads the score buffer.
  reg  sm_start;
  wire [4:0] sm_saddr, sm_wi;
  reg  signed [20:0] sm_sdata;
  wire sm_wv, sm_busy;
  wire [15:0] sm_wd;
  always @(posedge clk) sm_sdata <= sbuf[sm_saddr];
  softmax sm (.clk(clk), .rst_n(rst_n), .start(sm_start), .n(n_r),
              .s_addr(sm_saddr), .s_data(sm_sdata), .w_valid(sm_wv),
              .w_index(sm_wi), .w_data(sm_wd), .busy(sm_busy));
  reg  [5:0] pcnt;

  // Output: for each d, the weighted sum over positions j.
  reg issuing;
  reg [3:0] dd;
  reg [5:0] jj;
  reg [8:0] vbase;
  reg iv0, iv1, iv2, f0, f1, f2, l0, l1, l2;
  reg [3:0] d0, d1, d2;
  reg [4:0] pidx;
  reg [15:0] p_r;
  reg signed [24:0] prod2;
  // The weighted sum is sized from its own bound, the weights summing to
  // about 1.0, and widened to the requantizer's input only at the end.
  reg signed [23:0] accb;
  wire signed [23:0] sum3 = accb + prod2;

  reg  rq_vin;
  reg  signed [23:0] rq_acc;
  reg  [3:0] rq_idx;
  wire signed [7:0] rq_q;
  wire rq_sat, rq_vout;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc),
              .scale(scale_o), .shift(shift_o), .valid_in(rq_vin),
              .q_out(rq_q), .sat(rq_sat), .valid_out(rq_vout));
  // The index travels with its data through the requantizer's depth.
  reg [3:0] idx_pipe [0:5];
  reg [4:0] ocnt;
  integer k;

  always @(posedge clk) begin
    if (!rst_n) begin
      lptr <= 0; n_r <= 0; st <= S_IDLE; busy <= 1'b0;
      mv_start <= 1'b0; sm_start <= 1'b0; sv1 <= 1'b0; st1 <= 0;
      si1 <= 0; scnt <= 0; pcnt <= 0; issuing <= 1'b0; dd <= 0;
      rnd_r <= 0;
      jj <= 0; vbase <= 0; v_addr <= 0; pidx <= 0; p_r <= 0;
      iv0 <= 1'b0; iv1 <= 1'b0; iv2 <= 1'b0; f0 <= 1'b0; f1 <= 1'b0;
      f2 <= 1'b0; l0 <= 1'b0; l1 <= 1'b0; l2 <= 1'b0; d0 <= 0;
      d1 <= 0; d2 <= 0; prod2 <= 0; accb <= 0; rq_vin <= 1'b0;
      rq_acc <= 0; rq_idx <= 0; ocnt <= 0; o_valid <= 1'b0;
      o_index <= 0; o_data <= 0;
      for (k = 0; k <= 5; k = k + 1) idx_pipe[k] <= 0;
    end else begin
      mv_start <= 1'b0;
      sm_start <= 1'b0;
      rq_vin   <= 1'b0;
      o_valid  <= 1'b0;
      idx_pipe[0] <= rq_idx;
      for (k = 1; k <= 5; k = k + 1)
        idx_pipe[k] <= idx_pipe[k-1];

      if (load_valid && !busy) begin
        qbuf[lptr] <= load_data;
        lptr <= lptr + 1;
      end

      // Score quantizer.
      sv1 <= mv_colv;
      if (mv_colv) begin
        st1 <= {acc[23], acc, 4'd0} + rnd_r;
        si1 <= mv_coli[4:0];
      end
      if (sv1) begin
        if (shv > 29'sd1048575)
          sbuf[si1] <= 21'sd1048575;
        else if (shv < -29'sd1048576)
          sbuf[si1] <= -21'sd1048576;
        else
          sbuf[si1] <= shv[20:0];
        scnt <= scnt + 1;
      end

      if (sm_wv) begin
        pbuf[sm_wi] <= sm_wd;
        pcnt <= pcnt + 1;
      end

      // Weighted sum, four stages: issue the address, register the
      // weight while the value cache reads, form the product, add.
      iv0 <= 1'b0;
      if (issuing) begin
        v_addr <= vbase + dd;
        pidx <= jj[4:0];
        iv0 <= 1'b1; f0 <= (jj == 0); l0 <= (jj == n_r - 1); d0 <= dd;
        if (jj == n_r - 1) begin
          jj <= 0; vbase <= 0;
          if (dd == 15) issuing <= 1'b0;
          else dd <= dd + 1;
        end else begin
          jj <= jj + 1; vbase <= vbase + 9'd16;
        end
      end
      p_r <= pbuf[pidx];
      iv1 <= iv0; f1 <= f0; l1 <= l0; d1 <= d0;
      iv2 <= iv1; f2 <= f1; l2 <= l1; d2 <= d1;
      if (iv1) prod2 <= $signed({1'b0, p_r}) * v_data;
      if (iv2) begin
        accb <= l2 ? 24'sd0 : sum3;
        if (l2) begin
          rq_acc <= sum3;
          rq_idx <= d2; rq_vin <= 1'b1;
        end
      end

      if (rq_vout) begin
        o_valid <= 1'b1;
        o_index <= idx_pipe[5];
        o_data  <= rq_q;
        ocnt <= ocnt + 1;
      end

      case (st)
        S_IDLE: if (start) begin
          st <= S_SCORE; busy <= 1'b1; n_r <= n; scnt <= 0; pcnt <= 0;
          rnd_r <= rnd_s;
          ocnt <= 0; mv_start <= 1'b1;
        end
        S_SCORE: if (scnt == n_r) begin
          st <= S_SOFT; sm_start <= 1'b1;
        end
        S_SOFT: if (pcnt == n_r) begin
          st <= S_OUT; issuing <= 1'b1; dd <= 0; jj <= 0; vbase <= 0;
          accb <= 0;
        end
        S_OUT: if (ocnt == 16) begin
          st <= S_IDLE; busy <= 1'b0; lptr <= 0;
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
