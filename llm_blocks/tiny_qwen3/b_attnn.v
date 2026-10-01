module attnn (
  input                     clk,
  input                     rst_n,
  input                     load_valid,
  input  signed [15:0]      load_data,
  input                     start,
  input         [8:0]       n,
  input         [4:0]       shift_s,
  input         [17:0]      scale_o,
  input         [6:0]       shift_o,
  output        [8:0]       k_addr,
  input         [255:0]     k_data,
  output reg    [8:0]       v_addr,
  input         [255:0]     v_data,
  output reg                o_valid,
  output reg    [4:0]       o_index,
  output reg signed [15:0]  o_data,
  output reg                busy
);

  reg signed [15:0] qbuf [0:31];
  reg [4:0] q_wr;

  reg signed [20:0] sbuf [0:255];
  reg [15:0]        pbuf [0:255];

  localparam S_IDLE=3'd0, S_SCORE=3'd1, S_SOFTMAX=3'd2,
             S_WSUM=3'd3, S_EMIT=3'd4;
  reg [2:0] state;
  reg       dg;

  reg        mv_start;
  reg  [8:0] mv_cols;
  wire [8:0] mv_a_addr;
  wire [17:0] mv_w_addr;
  wire       mv_mac_valid, mv_mac_clear, mv_col_valid, mv_busy;
  wire [8:0] mv_col_index;
  reg        mv_seen;

  matvec u_mv (
    .clk(clk), .rst_n(rst_n), .start(mv_start),
    .depth(9'd32), .cols(mv_cols),
    .a_addr(mv_a_addr), .w_addr(mv_w_addr),
    .mac_valid(mv_mac_valid), .mac_clear(mv_mac_clear),
    .col_valid(mv_col_valid), .col_index(mv_col_index),
    .busy(mv_busy)
  );

  assign k_addr = mv_w_addr[8:0];

  reg signed [15:0] q_val_r;
  always @(posedge clk) q_val_r <= qbuf[mv_a_addr[4:0]];

  wire signed [36:0] lane_acc [0:15];
  wire               lane_vout [0:15];

  genvar gi;
  generate
    for (gi=0; gi<16; gi=gi+1) begin : sclane
      wire signed [15:0] k_elem = k_data[16*gi +: 16];
      mac_s u_ms (
        .clk(clk), .rst_n(rst_n), .clear(mv_mac_clear),
        .a(q_val_r), .b(k_elem),
        .valid_in(mv_mac_valid),
        .acc(lane_acc[gi]), .valid_out(lane_vout[gi])
      );
    end
  endgenerate

  function signed [20:0] clamp_s;
    input signed [40:0] s;
    begin
      if (s > 41'sd1048575)
        clamp_s = 21'sd1048575;
      else if (s < -41'sd1048576)
        clamp_s = -21'sd1048576;
      else
        clamp_s = s[20:0];
    end
  endfunction

  reg        sm_start;
  wire [7:0] sm_s_addr;
  reg signed [20:0] sm_s_data;
  wire       sm_w_valid, sm_busy;
  wire [7:0] sm_w_index;
  wire [15:0] sm_w_data;
  reg        sm_seen;

  softmax u_sm (
    .clk(clk), .rst_n(rst_n), .start(sm_start),
    .n(n),
    .s_addr(sm_s_addr), .s_data(sm_s_data),
    .w_valid(sm_w_valid), .w_index(sm_w_index), .w_data(sm_w_data),
    .busy(sm_busy)
  );

  always @(posedge clk) sm_s_data <= sbuf[sm_s_addr];

  reg [8:0]  j_cnt;
  reg        wsum_active;
  reg [8:0]  j_s1;
  reg        valid_s1;
  reg [8:0]  j_s2;
  reg        valid_s2;
  reg [15:0] w_s2;

  reg signed [31:0] out_acc [0:15];

  reg        rq_valid_in;
  reg signed [31:0] rq_acc_in;
  wire signed [15:0] rq_q_out;
  wire       rq_sat, rq_valid_out;

  requant u_rq (
    .clk(clk), .rst_n(rst_n),
    .acc_in(rq_acc_in), .scale(scale_o), .shift(shift_o),
    .valid_in(rq_valid_in),
    .q_out(rq_q_out), .sat(rq_sat), .valid_out(rq_valid_out)
  );

  reg [4:0] emit_in_idx;
  reg [4:0] emit_in_base;
  reg       emit_feeding;
  reg [4:0] emit_out_cnt;

  // Score quantize pipeline (4 stages)
  reg        col_v1, col_v2, col_v3, col_v4;
  reg [3:0]  col_g1, col_g2, col_g3, col_g4;
  reg [8:0]  n_s1,  n_s2,  n_s3,  n_s4;
  reg [4:0]  sh_s1, sh_s2;
  reg signed [36:0] lacc1 [0:15];
  reg signed [40:0] lacc2 [0:15];
  reg signed [40:0] lacc3 [0:15];

  integer i;
  reg signed [40:0] bias_tmp;

  always @(posedge clk) begin
    if (!rst_n) begin
      q_wr <= 5'd0;
      busy <= 1'b0;
      state <= S_IDLE;
      mv_start <= 1'b0;
      sm_start <= 1'b0;
      mv_cols <= 9'd0;
      mv_seen <= 1'b0;
      sm_seen <= 1'b0;
      o_valid <= 1'b0;
      o_index <= 5'd0;
      o_data  <= 16'sd0;
      v_addr  <= 9'd0;
      dg <= 1'b0;
      j_cnt <= 9'd0;
      wsum_active <= 1'b0;
      j_s1 <= 9'd0; valid_s1 <= 1'b0;
      j_s2 <= 9'd0; valid_s2 <= 1'b0; w_s2 <= 16'd0;
      rq_valid_in <= 1'b0;
      rq_acc_in <= 32'sd0;
      emit_in_idx <= 5'd0;
      emit_in_base <= 5'd0;
      emit_feeding <= 1'b0;
      emit_out_cnt <= 5'd0;
      col_v1 <= 1'b0; col_v2 <= 1'b0; col_v3 <= 1'b0; col_v4 <= 1'b0;
      col_g1 <= 4'd0; col_g2 <= 4'd0; col_g3 <= 4'd0; col_g4 <= 4'd0;
      n_s1 <= 9'd0; n_s2 <= 9'd0; n_s3 <= 9'd0; n_s4 <= 9'd0;
      sh_s1 <= 5'd0; sh_s2 <= 5'd0;
      for (i=0; i<16; i=i+1) out_acc[i] <= 32'sd0;
      for (i=0; i<16; i=i+1) lacc1[i] <= 37'sd0;
      for (i=0; i<16; i=i+1) lacc2[i] <= 41'sd0;
      for (i=0; i<16; i=i+1) lacc3[i] <= 41'sd0;
    end else begin
      mv_start    <= 1'b0;
      sm_start    <= 1'b0;
      rq_valid_in <= 1'b0;
      o_valid     <= 1'b0;

      if (load_valid && !busy && state == S_IDLE) begin
        qbuf[q_wr] <= load_data;
        q_wr <= q_wr + 5'd1;
      end

      if (sm_w_valid) begin
        pbuf[sm_w_index] <= sm_w_data;
      end

      // Score pipeline stage 1
      col_v1 <= mv_col_valid;
      col_g1 <= mv_col_index[3:0];
      n_s1   <= n;
      sh_s1  <= shift_s;
      for (i=0; i<16; i=i+1) lacc1[i] <= lane_acc[i];

      // Stage 2: shift left by 4, add bias
      col_v2 <= col_v1;
      col_g2 <= col_g1;
      n_s2   <= n_s1;
      sh_s2  <= sh_s1;
      if (sh_s1 == 5'd0)
        bias_tmp = 41'sd0;
      else
        bias_tmp = 41'sd1 <<< (sh_s1 - 5'd1);
      for (i=0; i<16; i=i+1) lacc2[i] <= {lacc1[i], 4'b0} + bias_tmp;

      // Stage 3: arithmetic right shift
      col_v3 <= col_v2;
      col_g3 <= col_g2;
      n_s3   <= n_s2;
      for (i=0; i<16; i=i+1) lacc3[i] <= lacc2[i] >>> sh_s2;

      // Stage 4: clamp and write sbuf
      col_v4 <= col_v3;
      col_g4 <= col_g3;
      n_s4   <= n_s3;
      if (col_v3) begin
        for (i=0; i<16; i=i+1) begin
          if ({col_g3, 4'b0} + i < {7'b0, n_s3}) begin
            sbuf[{col_g3, 4'b0} + i[7:0]] <= clamp_s(lacc3[i]);
          end
        end
      end

      // Weighted sum pipeline
      j_s2  <= j_s1;
      valid_s2 <= valid_s1;
      w_s2  <= pbuf[j_s1];

      if (valid_s2) begin
        for (i=0; i<16; i=i+1) begin
          out_acc[i] <= out_acc[i] +
                        $signed({1'b0, w_s2}) * $signed(v_data[16*i +: 16]);
        end
      end

      if (rq_valid_out) begin
        o_valid <= 1'b1;
        o_index <= emit_in_base + emit_out_cnt;
        o_data  <= rq_q_out;
        emit_out_cnt <= emit_out_cnt + 5'd1;
      end

      case (state)
        S_IDLE: begin
          valid_s1 <= 1'b0;
          if (start) begin
            busy <= 1'b1;
            mv_cols <= (n + 9'd15) >> 4;
            mv_start <= 1'b1;
            mv_seen  <= 1'b0;
            state <= S_SCORE;
          end
        end

        S_SCORE: begin
          if (mv_busy) mv_seen <= 1'b1;
          if (mv_seen && !mv_busy && !mv_col_valid &&
              !col_v1 && !col_v2 && !col_v3 && !col_v4) begin
            sm_start <= 1'b1;
            sm_seen  <= 1'b0;
            state <= S_SOFTMAX;
          end
        end

        S_SOFTMAX: begin
          if (sm_busy) sm_seen <= 1'b1;
          if (sm_seen && !sm_busy) begin
            dg <= 1'b0;
            j_cnt <= 9'd0;
            wsum_active <= 1'b1;
            v_addr <= 9'd0;
            for (i=0; i<16; i=i+1) out_acc[i] <= 32'sd0;
            state <= S_WSUM;
          end
        end

        S_WSUM: begin
          if (wsum_active) begin
            v_addr   <= (j_cnt << 1) | {8'b0, dg};
            j_s1     <= j_cnt;
            valid_s1 <= 1'b1;
            if (j_cnt == n - 9'd1) begin
              wsum_active <= 1'b0;
            end else begin
              j_cnt <= j_cnt + 9'd1;
            end
          end else begin
            valid_s1 <= 1'b0;
            if (!valid_s1 && !valid_s2) begin
              emit_feeding <= 1'b1;
              emit_in_idx  <= 5'd0;
              emit_in_base <= dg ? 5'd16 : 5'd0;
              emit_out_cnt <= 5'd0;
              state <= S_EMIT;
            end
          end
        end

        S_EMIT: begin
          if (emit_feeding) begin
            rq_valid_in <= 1'b1;
            rq_acc_in   <= out_acc[emit_in_idx];
            if (emit_in_idx == 5'd15) begin
              emit_feeding <= 1'b0;
            end
            emit_in_idx <= emit_in_idx + 5'd1;
          end
          if (!emit_feeding && (emit_out_cnt == 5'd16 ||
                                (emit_out_cnt == 5'd15 && rq_valid_out))) begin
            if (dg == 1'b0) begin
              dg <= 1'b1;
              j_cnt <= 9'd0;
              wsum_active <= 1'b1;
              v_addr <= 9'd1;
              for (i=0; i<16; i=i+1) out_acc[i] <= 32'sd0;
              state <= S_WSUM;
            end else begin
              busy <= 1'b0;
              q_wr <= 5'd0;
              state <= S_IDLE;
            end
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
