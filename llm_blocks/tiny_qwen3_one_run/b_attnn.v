module attnn(
  input clk,
  input rst_n,
  input load_valid,
  input signed [15:0] load_data,
  input start,
  input [8:0] n,
  input [4:0] shift_s,
  input [17:0] scale_o,
  input [6:0] shift_o,
  output [8:0] k_addr,
  input [255:0] k_data,
  output reg [8:0] v_addr,
  input [255:0] v_data,
  output o_valid,
  output [4:0] o_index,
  output signed [15:0] o_data,
  output busy
);

  reg signed [15:0] q [0:31];
  reg [4:0] q_wr_idx;
  reg signed [20:0] sbuf [0:255];
  reg [15:0] pbuf [0:255];

  reg busy_r;
  assign busy = busy_r;

  localparam S_IDLE  = 3'd0,
             S_SCORE = 3'd1,
             S_SMAX  = 3'd2,
             S_WSUM  = 3'd3,
             S_FEED  = 3'd4,
             S_DRAIN = 3'd5;
  reg [2:0] state;

  reg [8:0] n_r;
  reg [4:0] shift_s_r;
  reg [17:0] scale_o_r;
  reg [6:0] shift_o_r;

  reg mv_start;
  wire [8:0] mv_a_addr;
  wire [17:0] mv_w_addr;
  wire mv_mac_valid;
  wire mv_mac_clear;
  wire mv_col_valid;
  wire [8:0] mv_col_index;
  wire mv_busy;
  reg [8:0] cols_r;

  matvec u_mv (
    .clk(clk), .rst_n(rst_n),
    .start(mv_start),
    .depth(9'd32),
    .cols(cols_r),
    .a_addr(mv_a_addr),
    .w_addr(mv_w_addr),
    .mac_valid(mv_mac_valid),
    .mac_clear(mv_mac_clear),
    .col_valid(mv_col_valid),
    .col_index(mv_col_index),
    .busy(mv_busy)
  );

  reg signed [15:0] q_a_reg;
  always @(posedge clk) q_a_reg <= q[mv_a_addr[4:0]];

  assign k_addr = mv_w_addr[8:0];

  wire signed [36:0] s_acc [0:15];
  wire s_valid_out [0:15];
  genvar gl;
  generate
    for (gl=0; gl<16; gl=gl+1) begin : score_lanes
      mac_s u_mac_s (
        .clk(clk), .rst_n(rst_n),
        .clear(mv_mac_clear),
        .a(q_a_reg),
        .b(k_data[16*gl+15:16*gl]),
        .valid_in(mv_mac_valid),
        .acc(s_acc[gl]),
        .valid_out(s_valid_out[gl])
      );
    end
  endgenerate

  function signed [40:0] calc_pre;
    input signed [36:0] t;
    input [4:0] sh;
    reg signed [40:0] u, bias;
    begin
      u = {t, 4'b0000};
      if (sh != 5'd0)
        bias = (41'sd1 <<< (sh - 5'd1));
      else
        bias = 41'sd0;
      calc_pre = u + bias;
    end
  endfunction

  function signed [20:0] clamp_shift;
    input signed [40:0] pre;
    input [4:0] sh;
    reg signed [40:0] s;
    begin
      if (sh != 5'd0) s = pre >>> sh;
      else s = pre;
      if (s > 41'sd1048575)       clamp_shift = 21'sd1048575;
      else if (s < -41'sd1048576) clamp_shift = -21'sd1048576;
      else                        clamp_shift = s[20:0];
    end
  endfunction

  reg signed [40:0] pre_pipe [0:15];
  reg [4:0] sh_pipe;
  reg [7:0] jbase_pipe;
  reg val_pipe;
  reg [15:0] lane_val_pipe;

  reg sm_start;
  wire [7:0] sm_s_addr;
  reg signed [20:0] sm_s_data_reg;
  wire sm_w_valid;
  wire [7:0] sm_w_index;
  wire [15:0] sm_w_data;
  wire sm_busy;
  always @(posedge clk) sm_s_data_reg <= sbuf[sm_s_addr];

  softmax u_sm (
    .clk(clk), .rst_n(rst_n),
    .start(sm_start),
    .n(n_r),
    .s_addr(sm_s_addr),
    .s_data(sm_s_data_reg),
    .w_valid(sm_w_valid),
    .w_index(sm_w_index),
    .w_data(sm_w_data),
    .busy(sm_busy)
  );

  reg [8:0] sm_remaining;
  reg [8:0] score_cols_rem;
  reg score_done_pending;

  reg dg;
  reg [8:0] j_issue;
  reg vA_valid, vB_valid;
  reg [15:0] w_a, w_b;
  reg signed [31:0] acc [0:15];

  reg [4:0] feed_idx;
  reg req_valid_in;
  reg signed [31:0] req_acc_in;
  wire signed [15:0] req_q_out;
  wire req_sat;
  wire req_valid_out;

  requant u_rq (
    .clk(clk), .rst_n(rst_n),
    .acc_in(req_acc_in),
    .scale(scale_o_r),
    .shift(shift_o_r),
    .valid_in(req_valid_in),
    .q_out(req_q_out),
    .sat(req_sat),
    .valid_out(req_valid_out)
  );

  reg [5:0] out_count;

  assign o_valid = req_valid_out;
  assign o_index = out_count[4:0];
  assign o_data  = req_q_out;

  integer l;
  integer j_tmp;
  always @(posedge clk) begin
    if (!rst_n) begin
      state <= S_IDLE;
      busy_r <= 1'b0;
      mv_start <= 1'b0;
      sm_start <= 1'b0;
      req_valid_in <= 1'b0;
      dg <= 1'b0;
      j_issue <= 9'd0;
      vA_valid <= 1'b0;
      vB_valid <= 1'b0;
      feed_idx <= 5'd0;
      out_count <= 6'd0;
      v_addr <= 9'd0;
      cols_r <= 9'd0;
      score_cols_rem <= 9'd0;
      score_done_pending <= 1'b0;
      sm_remaining <= 9'd0;
      n_r <= 9'd0;
      shift_s_r <= 5'd0;
      scale_o_r <= 18'd0;
      shift_o_r <= 7'd0;
      w_a <= 16'd0;
      w_b <= 16'd0;
      q_wr_idx <= 5'd0;
      req_acc_in <= 32'd0;
      val_pipe <= 1'b0;
      sh_pipe <= 5'd0;
      jbase_pipe <= 8'd0;
      lane_val_pipe <= 16'd0;
      for (l=0; l<16; l=l+1) acc[l] <= 32'd0;
    end else begin
      mv_start <= 1'b0;
      sm_start <= 1'b0;
      req_valid_in <= 1'b0;
      val_pipe <= 1'b0;

      if (load_valid) begin
        q[q_wr_idx] <= load_data;
        q_wr_idx <= q_wr_idx + 5'd1;
      end

      if (val_pipe) begin
        for (l=0; l<16; l=l+1) begin
          if (lane_val_pipe[l])
            sbuf[jbase_pipe + l[7:0]] <= clamp_shift(pre_pipe[l], sh_pipe);
        end
      end

      case (state)
        S_IDLE: begin
          if (start) begin
            busy_r <= 1'b1;
            n_r <= n;
            shift_s_r <= shift_s;
            scale_o_r <= scale_o;
            shift_o_r <= shift_o;
            cols_r <= (n + 9'd15) >> 4;
            score_cols_rem <= (n + 9'd15) >> 4;
            score_done_pending <= 1'b0;
            mv_start <= 1'b1;
            state <= S_SCORE;
          end
        end

        S_SCORE: begin
          if (mv_col_valid) begin
            for (l=0; l<16; l=l+1) begin
              pre_pipe[l] <= calc_pre(s_acc[l], shift_s_r);
              j_tmp = mv_col_index * 16 + l;
              lane_val_pipe[l] <= (j_tmp < n_r);
            end
            sh_pipe <= shift_s_r;
            jbase_pipe <= {mv_col_index[3:0], 4'b0000};
            val_pipe <= 1'b1;
            score_cols_rem <= score_cols_rem - 9'd1;
            if (score_cols_rem == 9'd1)
              score_done_pending <= 1'b1;
          end
          if (score_done_pending && !val_pipe) begin
            state <= S_SMAX;
            sm_start <= 1'b1;
            sm_remaining <= n_r;
            score_done_pending <= 1'b0;
          end
        end

        S_SMAX: begin
          if (sm_w_valid) begin
            pbuf[sm_w_index] <= sm_w_data;
            sm_remaining <= sm_remaining - 9'd1;
            if (sm_remaining == 9'd1) begin
              state <= S_WSUM;
              dg <= 1'b0;
              j_issue <= 9'd0;
              vA_valid <= 1'b0;
              vB_valid <= 1'b0;
              for (l=0; l<16; l=l+1) acc[l] <= 32'd0;
            end
          end
        end

        S_WSUM: begin
          if (j_issue < n_r) begin
            v_addr <= {j_issue[7:0], dg};
            w_a <= pbuf[j_issue[7:0]];
            vA_valid <= 1'b1;
            j_issue <= j_issue + 9'd1;
          end else begin
            vA_valid <= 1'b0;
          end
          w_b <= w_a;
          vB_valid <= vA_valid;
          if (vB_valid) begin
            for (l=0; l<16; l=l+1) begin
              acc[l] <= acc[l] +
                        $signed({1'b0, w_b}) *
                        $signed(v_data[16*l +: 16]);
            end
          end
          if (j_issue >= n_r && !vA_valid && !vB_valid) begin
            state <= S_FEED;
            feed_idx <= 5'd0;
          end
        end

        S_FEED: begin
          if (feed_idx < 5'd16) begin
            req_valid_in <= 1'b1;
            req_acc_in <= acc[feed_idx[3:0]];
            feed_idx <= feed_idx + 5'd1;
          end else begin
            if (dg == 1'b0) begin
              dg <= 1'b1;
              j_issue <= 9'd0;
              vA_valid <= 1'b0;
              vB_valid <= 1'b0;
              for (l=0; l<16; l=l+1) acc[l] <= 32'd0;
              state <= S_WSUM;
            end else begin
              state <= S_DRAIN;
            end
          end
        end

        S_DRAIN: begin
        end

        default: state <= S_IDLE;
      endcase

      if (req_valid_out) begin
        out_count <= out_count + 6'd1;
        if (out_count == 6'd31) begin
          busy_r <= 1'b0;
          state <= S_IDLE;
          out_count <= 6'd0;
          q_wr_idx <= 5'd0;
          dg <= 1'b0;
        end
      end
    end
  end

endmodule
