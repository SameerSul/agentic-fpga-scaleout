module softmax(
  input             clk,
  input             rst_n,
  input             start,
  input      [8:0]  n,
  output reg [7:0]  s_addr,
  input  signed [20:0] s_data,
  output reg        w_valid,
  output reg [7:0]  w_index,
  output reg [15:0] w_data,
  output reg        busy
);

  reg               expu_vin;
  reg  signed [12:0] expu_x;
  wire              expu_vout;
  wire [15:0]       expu_y;
  expu expu_inst(.clk(clk),.rst_n(rst_n),.x(expu_x),.valid_in(expu_vin),.y(expu_y),.valid_out(expu_vout));

  reg               recip_vin;
  reg  [23:0]       recip_x;
  wire              recip_vout;
  wire [16:0]       recip_y;
  wire [4:0]        recip_k;
  recip recip_inst(.clk(clk),.rst_n(rst_n),.x(recip_x),.valid_in(recip_vin),.y(recip_y),.k(recip_k),.valid_out(recip_vout));

  reg [15:0] exp_buf [0:255];

  localparam IDLE    = 4'd0;
  localparam P1      = 4'd1;
  localparam P2      = 4'd2;
  localparam P2_WAIT = 4'd3;
  localparam P3_R    = 4'd4;
  localparam P3_W    = 4'd5;
  localparam P3_E    = 4'd6;
  localparam DRAIN   = 4'd7;

  reg [3:0]  state;
  reg [8:0]  n_reg;
  reg [9:0]  pc;
  reg [8:0]  exp_cnt;
  reg [8:0]  emit_cnt;
  reg signed [20:0] mx;
  reg [23:0] S;
  reg [16:0] mant;
  reg [4:0]  shft;

  reg [7:0]  expu_in_idx;
  reg [7:0]  eip0, eip1, eip2;

  wire signed [21:0] diff22;
  wire signed [12:0] d_i;
  assign diff22 = {s_data[20], s_data} - {mx[20], mx};
  assign d_i = (diff22 < -22'sd4096) ? -13'sd4096 : diff22[12:0];

  reg [15:0] p0_e;
  reg [5:0]  p0_rsh;
  reg [7:0]  p0_idx;
  reg        p0_vld;

  reg [47:0] s1_full;
  reg [5:0]  s1_rsh;
  reg [7:0]  s1_idx;
  reg        s1_vld;
  reg [47:0] s2_res;
  reg [7:0]  s2_idx;
  reg        s2_vld;

  wire [32:0] em = p0_e * mant;
  wire [47:0] em_full = {em, 15'd0};

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= IDLE;
      busy      <= 0;
      s_addr    <= 0;
      expu_vin  <= 0; expu_x <= 0;
      recip_vin <= 0; recip_x <= 0;
      pc        <= 0; exp_cnt <= 0; emit_cnt <= 0;
      mx        <= 0; S <= 0;
      mant      <= 0; shft <= 0; n_reg <= 0;
      p0_vld    <= 0; p0_e <= 0; p0_rsh <= 0; p0_idx <= 0;
      s1_vld    <= 0; s1_full <= 0; s1_rsh <= 0; s1_idx <= 0;
      s2_vld    <= 0; s2_res <= 0; s2_idx <= 0;
      w_valid   <= 0; w_index <= 0; w_data <= 0;
      expu_in_idx <= 0; eip0 <= 0; eip1 <= 0; eip2 <= 0;
    end else begin
      expu_vin  <= 0;
      recip_vin <= 0;
      p0_vld    <= 0;

      eip0 <= expu_in_idx;
      eip1 <= eip0;
      eip2 <= eip1;

      if (expu_vout) begin
        exp_buf[eip2] <= expu_y;
        S <= S + {8'd0, expu_y};
        exp_cnt <= exp_cnt + 1;
      end

      s1_full <= em_full;
      s1_rsh  <= p0_rsh;
      s1_idx  <= p0_idx;
      s1_vld  <= p0_vld;

      s2_res <= s1_full >> s1_rsh;
      s2_idx <= s1_idx;
      s2_vld <= s1_vld;

      w_valid <= s2_vld;
      w_index <= s2_idx;
      if (s2_vld)
        w_data <= (s2_res >= 48'd32768) ? 16'd32768 : s2_res[15:0];

      case (state)
        IDLE: begin
          if (start) begin
            busy    <= 1;
            n_reg   <= n;
            s_addr  <= 8'd0;
            mx      <= 21'sh100000;
            S       <= 24'd0;
            exp_cnt <= 9'd0;
            pc      <= 10'd0;
            state   <= P1;
          end
        end

        P1: begin
          if (pc < {1'b0, n_reg}) s_addr <= pc[7:0];
          if (pc >= 10'd2) begin
            if ($signed(s_data) > $signed(mx)) mx <= s_data;
          end
          if (pc == {1'b0, n_reg} + 10'd1) begin
            pc     <= 10'd0;
            s_addr <= 8'd0;
            state  <= P2;
          end else begin
            pc <= pc + 10'd1;
          end
        end

        P2: begin
          if (pc < {1'b0, n_reg}) s_addr <= pc[7:0];
          if (pc >= 10'd2) begin
            expu_x      <= d_i;
            expu_vin    <= 1'b1;
            expu_in_idx <= pc[7:0] - 8'd2;
          end
          if (pc == {1'b0, n_reg} + 10'd1) begin
            pc    <= 10'd0;
            state <= P2_WAIT;
          end else begin
            pc <= pc + 10'd1;
          end
        end

        P2_WAIT: begin
          if (exp_cnt >= n_reg) state <= P3_R;
        end

        P3_R: begin
          recip_x   <= S;
          recip_vin <= 1'b1;
          state     <= P3_W;
        end

        P3_W: begin
          if (recip_vout) begin
            mant     <= recip_y;
            shft     <= recip_k;
            emit_cnt <= 9'd0;
            state    <= P3_E;
          end
        end

        P3_E: begin
          if (emit_cnt < n_reg) begin
            p0_e     <= exp_buf[emit_cnt[7:0]];
            p0_rsh   <= 6'd40 - {1'b0, shft};
            p0_idx   <= emit_cnt[7:0];
            p0_vld   <= 1'b1;
            emit_cnt <= emit_cnt + 9'd1;
          end else begin
            state <= DRAIN;
          end
        end

        DRAIN: begin
          if (!p0_vld && !s1_vld && !s2_vld && !w_valid) begin
            state <= IDLE;
            busy  <= 1'b0;
          end
        end

        default: state <= IDLE;
      endcase
    end
  end
endmodule
