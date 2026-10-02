module rmsnorm_hd (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [39:0]  eps,
    input  wire [17:0]  scale_o,
    input  wire  [6:0]  shift_o,
    output reg   [4:0]  x_addr,
    input  wire signed [15:0] x_data,
    output reg   [4:0]  g_addr,
    input  wire signed [15:0] g_data,
    output reg          o_valid,
    output reg   [4:0]  o_index,
    output reg  signed [15:0] o_data,
    output reg          busy
);

    reg [39:0] ssq;
    reg [16:0] rs_m;
    reg  [5:0] rs_e;

    wire [16:0] rsqrt_y;
    wire  [5:0] rsqrt_e_out;
    wire        rsqrt_vout;
    reg         rsqrt_vin;

    rsqrt rsqrt_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .x        (ssq),
        .valid_in (rsqrt_vin),
        .y        (rsqrt_y),
        .e        (rsqrt_e_out),
        .valid_out(rsqrt_vout)
    );

    wire signed [15:0] requant_qout;
    wire               requant_sat;
    wire               requant_vout;
    reg                requant_vin;
    reg  signed [31:0] requant_acc;

    requant requant_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .acc_in   (requant_acc),
        .scale    (scale_o),
        .shift    (shift_o),
        .valid_in (requant_vin),
        .q_out    (requant_qout),
        .sat      (requant_sat),
        .valid_out(requant_vout)
    );

    localparam S_IDLE  = 3'd0;
    localparam S_P1    = 3'd1;
    localparam S_RSQRT = 3'd2;
    localparam S_P2    = 3'd3;
    localparam S_DRAIN = 3'd4;

    reg [2:0] state;
    reg [5:0] cnt;
    reg [5:0] out_cnt;

    reg [31:0] x_sq_reg;
    reg        x_sq_valid;

    reg signed [31:0] xg_reg;
    reg               m0_valid;
    reg  [4:0]        m0_idx;

    reg signed [31:0] xg_reg_d;
    reg        [8:0]  rs_m_lo_d;
    reg        [7:0]  rs_m_hi_d;
    reg               m0b_valid;
    reg  [4:0]        m0b_idx;

    // New stage: sub-products
    reg signed [41:0] pp_lo_a;
    reg signed [41:0] pp_lo_b;
    reg signed [41:0] pp_hi_a;
    reg signed [41:0] pp_hi_b;
    reg               msa_valid;
    reg  [4:0]        msa_idx;

    reg signed [41:0] pp_lo;
    reg signed [40:0] pp_hi;
    reg               msb_valid;
    reg  [4:0]        msb_idx;

    reg signed [49:0] xgm_reg;
    reg               m1_valid;
    reg  [4:0]        m1_idx;

    reg [4:0] idx_pipe [0:9];

    wire [31:0] x_sq_comb;
    assign x_sq_comb = $signed(x_data) * $signed(x_data);

    wire signed [31:0] xg_comb;
    assign xg_comb = $signed(x_data) * $signed(g_data);

    wire signed [41:0] pp_lo_a_comb;
    wire signed [41:0] pp_lo_b_comb;
    wire signed [41:0] pp_hi_a_comb;
    wire signed [41:0] pp_hi_b_comb;
    assign pp_lo_a_comb = $signed(xg_reg_d) * $signed({1'b0, rs_m_lo_d[4:0]});
    assign pp_lo_b_comb = $signed(xg_reg_d) * $signed({1'b0, rs_m_lo_d[8:5]});
    assign pp_hi_a_comb = $signed(xg_reg_d) * $signed({1'b0, rs_m_hi_d[4:0]});
    assign pp_hi_b_comb = $signed(xg_reg_d) * $signed({1'b0, rs_m_hi_d[7:5]});

    wire signed [41:0] pp_lo_comb;
    wire signed [40:0] pp_hi_comb;
    assign pp_lo_comb = pp_lo_a + (pp_lo_b <<< 5);
    assign pp_hi_comb = pp_hi_a[40:0] + (pp_hi_b[40:0] <<< 5);

    wire signed [49:0] xgm_comb;
    assign xgm_comb = $signed(pp_lo) + ($signed(pp_hi) <<< 9);

    wire        [6:0]  shift_amt;
    wire signed [31:0] t_comb;
    assign shift_amt = {1'b0, rs_e} + 7'd2;
    assign t_comb    = $signed(xgm_reg) >>> shift_amt;

    integer ii;

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            cnt         <= 6'd0;
            out_cnt     <= 6'd0;
            busy        <= 1'b0;
            o_valid     <= 1'b0;
            o_index     <= 5'd0;
            o_data      <= 16'd0;
            x_addr      <= 5'd0;
            g_addr      <= 5'd0;
            ssq         <= 40'd0;
            rs_m        <= 17'd0;
            rs_e        <= 6'd0;
            rsqrt_vin   <= 1'b0;
            requant_vin <= 1'b0;
            requant_acc <= 32'd0;
            xg_reg      <= 32'd0;
            m0_valid    <= 1'b0;
            m0_idx      <= 5'd0;
            xg_reg_d    <= 32'd0;
            rs_m_lo_d   <= 9'd0;
            rs_m_hi_d   <= 8'd0;
            m0b_valid   <= 1'b0;
            m0b_idx     <= 5'd0;
            pp_lo_a     <= 42'd0;
            pp_lo_b     <= 42'd0;
            pp_hi_a     <= 42'd0;
            pp_hi_b     <= 42'd0;
            msa_valid   <= 1'b0;
            msa_idx     <= 5'd0;
            pp_lo       <= 42'd0;
            pp_hi       <= 41'd0;
            msb_valid   <= 1'b0;
            msb_idx     <= 5'd0;
            xgm_reg     <= 50'd0;
            m1_valid    <= 1'b0;
            m1_idx      <= 5'd0;
            x_sq_reg    <= 32'd0;
            x_sq_valid  <= 1'b0;
            for (ii = 0; ii <= 9; ii = ii + 1)
                idx_pipe[ii] <= 5'd0;
        end else begin
            rsqrt_vin   <= 1'b0;
            requant_vin <= 1'b0;
            o_valid     <= 1'b0;
            m0_valid    <= 1'b0;
            m0b_valid   <= 1'b0;
            msa_valid   <= 1'b0;
            msb_valid   <= 1'b0;
            m1_valid    <= 1'b0;
            x_sq_valid  <= 1'b0;

            for (ii = 1; ii <= 9; ii = ii + 1)
                idx_pipe[ii] <= idx_pipe[ii-1];

            if (m0_valid) begin
                xg_reg_d  <= xg_reg;
                rs_m_lo_d <= rs_m[8:0];
                rs_m_hi_d <= rs_m[16:9];
                m0b_valid <= 1'b1;
                m0b_idx   <= m0_idx;
            end

            if (m0b_valid) begin
                pp_lo_a   <= pp_lo_a_comb;
                pp_lo_b   <= pp_lo_b_comb;
                pp_hi_a   <= pp_hi_a_comb;
                pp_hi_b   <= pp_hi_b_comb;
                msa_valid <= 1'b1;
                msa_idx   <= m0b_idx;
            end

            if (msa_valid) begin
                pp_lo     <= pp_lo_comb;
                pp_hi     <= pp_hi_comb;
                msb_valid <= 1'b1;
                msb_idx   <= msa_idx;
            end

            if (msb_valid) begin
                xgm_reg  <= xgm_comb;
                m1_valid <= 1'b1;
                m1_idx   <= msb_idx;
            end

            if (m1_valid) begin
                requant_acc <= t_comb;
                requant_vin <= 1'b1;
                idx_pipe[0] <= m1_idx;
            end

            if (x_sq_valid) begin
                ssq <= ssq + {{8{1'b0}}, x_sq_reg};
            end

            if (requant_vout) begin
                o_valid <= 1'b1;
                o_index <= idx_pipe[9];
                o_data  <= requant_qout;
                out_cnt <= out_cnt + 6'd1;
                if (out_cnt == 6'd31) begin
                    busy    <= 1'b0;
                    state   <= S_IDLE;
                    out_cnt <= 6'd0;
                end
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy   <= 1'b1;
                        ssq    <= eps;
                        x_addr <= 5'd0;
                        g_addr <= 5'd0;
                        cnt    <= 6'd0;
                        state  <= S_P1;
                    end
                end

                S_P1: begin
                    cnt <= cnt + 6'd1;
                    if (cnt <= 6'd30)
                        x_addr <= cnt[4:0] + 5'd1;
                    if (cnt >= 6'd1 && cnt <= 6'd32) begin
                        x_sq_reg   <= x_sq_comb;
                        x_sq_valid <= 1'b1;
                    end
                    if (cnt == 6'd33) begin
                        state <= S_RSQRT;
                        cnt   <= 6'd0;
                    end
                end

                S_RSQRT: begin
                    cnt <= cnt + 6'd1;
                    if (cnt == 6'd0)
                        rsqrt_vin <= 1'b1;
                    if (rsqrt_vout) begin
                        rs_m   <= rsqrt_y;
                        rs_e   <= rsqrt_e_out;
                        x_addr <= 5'd0;
                        g_addr <= 5'd0;
                        cnt    <= 6'd0;
                        state  <= S_P2;
                    end
                end

                S_P2: begin
                    cnt <= cnt + 6'd1;
                    if (cnt <= 6'd30) begin
                        x_addr <= cnt[4:0] + 5'd1;
                        g_addr <= cnt[4:0] + 5'd1;
                    end
                    if (cnt >= 6'd1 && cnt <= 6'd32) begin
                        xg_reg   <= xg_comb;
                        m0_valid <= 1'b1;
                        m0_idx   <= cnt[4:0] - 5'd1;
                    end
                    if (cnt == 6'd33) begin
                        state <= S_DRAIN;
                        cnt   <= 6'd0;
                    end
                end

                S_DRAIN: begin
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
