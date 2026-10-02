module softmax(
  input clk,
  input rst_n,
  input start,
  input [8:0] n,
  output [7:0] s_addr,
  input signed [20:0] s_data,
  output reg w_valid,
  output reg [7:0] w_index,
  output reg [15:0] w_data,
  output reg busy
);

  localparam S_IDLE=3'd0, S_P1=3'd1, S_P2=3'd2, S_P2_DRAIN=3'd3,
             S_RECIP=3'd4, S_RECIP_WAIT=3'd5, S_P3=3'd6;

  reg [2:0] state;
  reg [8:0] n_r;
  reg [8:0] addr_cnt;
  reg signed [20:0] mx;
  reg [23:0] sum_s;
  reg [8:0] e_cnt;
  reg [8:0] p3_cnt;
  reg [16:0] m_r;
  reg [4:0]  k_r;
  reg [15:0] ebuf [0:255];

  assign s_addr = addr_cnt[7:0];

  wire signed [21:0] diff = {s_data[20], s_data} - {mx[20], mx};
  wire signed [12:0] d_clamp = (diff < -22'sd4096) ? -13'sd4096 :
                               (diff > 22'sd0)    ?  13'sd0    :
                               diff[12:0];

  reg               expu_vin;
  reg signed [12:0] expu_x;
  wire [15:0]       expu_y;
  wire              expu_vout;

  expu u_expu(.clk(clk), .rst_n(rst_n), .x(expu_x), .valid_in(expu_vin),
              .y(expu_y), .valid_out(expu_vout));

  reg          recip_vin;
  reg  [23:0]  recip_x;
  wire [16:0]  recip_m;
  wire [4:0]   recip_k;
  wire         recip_vout;

  recip u_recip(.clk(clk), .rst_n(rst_n), .x(recip_x), .valid_in(recip_vin),
                .y(recip_m), .k(recip_k), .valid_out(recip_vout));

  // Stage 0: ebuf read result registered
  reg          s0_valid;
  reg [7:0]    s0_idx;
  reg [15:0]   s0_e;
  reg [5:0]    s0_shamt;

  // Stage A: multiply
  reg          sA_valid;
  reg [7:0]    sA_idx;
  reg [47:0]   sA_prod;
  reg [5:0]    sA_shamt;

  wire [30:0] e_sh_a = {s0_e, 15'b0};

  // Stage B: shift and clamp
  wire [47:0] shifted = sA_prod >> sA_shamt;
  wire [15:0] w_calc  = (|shifted[47:15]) ? 16'h8000 : shifted[15:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      busy      <= 1'b0;
      w_valid   <= 1'b0;
      w_index   <= 8'd0;
      w_data    <= 16'd0;
      expu_vin  <= 1'b0;
      expu_x    <= 13'sd0;
      recip_vin <= 1'b0;
      recip_x   <= 24'd0;
      addr_cnt  <= 9'd0;
      e_cnt     <= 9'd0;
      p3_cnt    <= 9'd0;
      sum_s     <= 24'd0;
      mx        <= 21'sh100000;
      n_r       <= 9'd0;
      m_r       <= 17'd0;
      k_r       <= 5'd0;
      s0_valid  <= 1'b0;
      s0_idx    <= 8'd0;
      s0_e      <= 16'd0;
      s0_shamt  <= 6'd0;
      sA_valid  <= 1'b0;
      sA_idx    <= 8'd0;
      sA_prod   <= 48'd0;
      sA_shamt  <= 6'd0;
    end else begin
      w_valid   <= 1'b0;
      expu_vin  <= 1'b0;
      recip_vin <= 1'b0;
      s0_valid  <= 1'b0;
      sA_valid  <= 1'b0;

      if (expu_vout) begin
        ebuf[e_cnt[7:0]] <= expu_y;
        sum_s <= sum_s + {8'b0, expu_y};
        e_cnt <= e_cnt + 9'd1;
      end

      if (recip_vout) begin
        m_r <= recip_m;
        k_r <= recip_k;
      end

      // Advance pipeline stages always
      if (s0_valid) begin
        sA_valid <= 1'b1;
        sA_idx   <= s0_idx;
        sA_prod  <= e_sh_a * m_r;
        sA_shamt <= s0_shamt;
      end
      if (sA_valid) begin
        w_valid <= 1'b1;
        w_index <= sA_idx;
        w_data  <= w_calc;
      end

      case (state)
        S_IDLE: begin
          if (start) begin
            busy     <= 1'b1;
            n_r      <= n;
            addr_cnt <= 9'd0;
            mx       <= 21'sh100000;
            sum_s    <= 24'd0;
            e_cnt    <= 9'd0;
            p3_cnt   <= 9'd0;
            state    <= S_P1;
          end
        end

        S_P1: begin
          addr_cnt <= addr_cnt + 9'd1;
          if (addr_cnt >= 9'd1 && addr_cnt <= n_r) begin
            if ($signed(s_data) > $signed(mx)) mx <= s_data;
            if (addr_cnt == n_r) begin
              addr_cnt <= 9'd0;
              state    <= S_P2;
            end
          end
        end

        S_P2: begin
          addr_cnt <= addr_cnt + 9'd1;
          if (addr_cnt >= 9'd1 && addr_cnt <= n_r) begin
            expu_vin <= 1'b1;
            expu_x   <= d_clamp;
            if (addr_cnt == n_r) begin
              state <= S_P2_DRAIN;
            end
          end
        end

        S_P2_DRAIN: begin
          if ((e_cnt + {8'b0, expu_vout}) == n_r) begin
            state <= S_RECIP;
          end
        end

        S_RECIP: begin
          recip_vin <= 1'b1;
          recip_x   <= sum_s;
          state     <= S_RECIP_WAIT;
        end

        S_RECIP_WAIT: begin
          if (recip_vout) begin
            p3_cnt <= 9'd0;
            state  <= S_P3;
          end
        end

        S_P3: begin
          if (p3_cnt < n_r) begin
            s0_valid <= 1'b1;
            s0_idx   <= p3_cnt[7:0];
            s0_e     <= ebuf[p3_cnt[7:0]];
            s0_shamt <= 6'd40 - {1'b0, k_r};
            p3_cnt   <= p3_cnt + 9'd1;
          end
          if (p3_cnt >= n_r && !s0_valid && !sA_valid) begin
            busy  <= 1'b0;
            state <= S_IDLE;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
