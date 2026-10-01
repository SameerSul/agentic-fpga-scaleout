module rmsnorm_hd(
    input clk,
    input rst_n,
    input start,
    input [39:0] eps,
    input [17:0] scale_o,
    input [6:0] shift_o,
    output [4:0] x_addr,
    input signed [15:0] x_data,
    output [4:0] g_addr,
    input signed [15:0] g_data,
    output o_valid,
    output [4:0] o_index,
    output signed [15:0] o_data,
    output busy
);

    localparam IDLE=2'd0, PASS1=2'd1, WAIT_RSQRT=2'd2, PASS2=2'd3;

    reg [1:0] state;
    reg [5:0] cnt;
    reg [39:0] ssq;
    reg [16:0] rs_m;
    reg [5:0]  rs_e;
    reg        busy_reg;
    reg [5:0]  done_cnt;

    reg [4:0] x_addr_r, g_addr_r;

    reg signed [31:0] x_sq_reg;

    reg signed [31:0] xg_reg;
    reg               xg_valid;
    reg [4:0]         xg_idx;

    reg signed [48:0] xgm_reg;
    reg               xgm_valid;
    reg [4:0]         xgm_idx;

    reg signed [31:0] t_val_reg;
    reg               t_valid_reg;
    reg [4:0]         t_idx_reg;

    reg [4:0] rq_idx [0:8];

    reg rsqrt_vin;

    wire signed [48:0] xgm_full;
    wire [6:0] shamt;
    wire signed [48:0] t_shifted;

    wire [16:0] rsqrt_y;
    wire [5:0]  rsqrt_e_out;
    wire        rsqrt_valid_out;
    wire signed [15:0] requant_q_out;
    wire               requant_sat;
    wire               requant_valid_out;

    assign xgm_full  = $signed({{17{xg_reg[31]}}, xg_reg}) * $signed({1'b0, rs_m});
    assign shamt     = {1'b0, rs_e} + 7'd2;
    assign t_shifted = xgm_reg >>> shamt;

    assign x_addr  = x_addr_r;
    assign g_addr  = g_addr_r;
    assign busy    = busy_reg;
    assign o_valid = requant_valid_out;
    assign o_data  = requant_q_out;
    assign o_index = rq_idx[8];

    rsqrt rsqrt_inst (
        .clk(clk), .rst_n(rst_n), .x(ssq), .valid_in(rsqrt_vin),
        .y(rsqrt_y), .e(rsqrt_e_out), .valid_out(rsqrt_valid_out)
    );

    requant requant_inst (
        .clk(clk), .rst_n(rst_n), .acc_in(t_val_reg),
        .scale(scale_o), .shift(shift_o), .valid_in(t_valid_reg),
        .q_out(requant_q_out), .sat(requant_sat), .valid_out(requant_valid_out)
    );

    integer kk;

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= IDLE;
            busy_reg    <= 1'b0;
            cnt         <= 6'd0;
            ssq         <= 40'd0;
            rs_m        <= 17'd0;
            rs_e        <= 6'd0;
            x_addr_r    <= 5'd0;
            g_addr_r    <= 5'd0;
            done_cnt    <= 6'd0;
            x_sq_reg    <= 32'd0;
            xg_reg      <= 32'd0;
            xg_valid    <= 1'b0;
            xg_idx      <= 5'd0;
            xgm_reg     <= 49'd0;
            xgm_valid   <= 1'b0;
            xgm_idx     <= 5'd0;
            t_val_reg   <= 32'd0;
            t_valid_reg <= 1'b0;
            t_idx_reg   <= 5'd0;
            rsqrt_vin   <= 1'b0;
            for (kk=0; kk<9; kk=kk+1) rq_idx[kk] <= 5'd0;
        end else begin
            rsqrt_vin <= 1'b0;

            x_sq_reg  <= x_data * x_data;
            xg_reg    <= x_data * g_data;
            xg_valid  <= 1'b0;

            xgm_reg   <= xgm_full;
            xgm_valid <= xg_valid;
            xgm_idx   <= xg_idx;

            t_val_reg   <= t_shifted[31:0];
            t_valid_reg <= xgm_valid;
            t_idx_reg   <= xgm_idx;

            rq_idx[0] <= t_idx_reg;
            for (kk=1; kk<9; kk=kk+1) rq_idx[kk] <= rq_idx[kk-1];

            case (state)
                IDLE: begin
                    if (start) begin
                        busy_reg <= 1'b1;
                        ssq      <= eps;
                        x_addr_r <= 5'd0;
                        g_addr_r <= 5'd0;
                        cnt      <= 6'd0;
                        state    <= PASS1;
                    end
                end

                PASS1: begin
                    if (cnt < 6'd31) begin
                        x_addr_r <= x_addr_r + 5'd1;
                        g_addr_r <= g_addr_r + 5'd1;
                    end
                    if (cnt >= 6'd2 && cnt <= 6'd33) begin
                        ssq <= ssq + {{8{x_sq_reg[31]}}, x_sq_reg};
                    end
                    cnt <= cnt + 6'd1;
                    if (cnt == 6'd34) begin
                        state     <= WAIT_RSQRT;
                        rsqrt_vin <= 1'b1;
                        cnt       <= 6'd0;
                    end
                end

                WAIT_RSQRT: begin
                    if (rsqrt_valid_out) begin
                        rs_m     <= rsqrt_y;
                        rs_e     <= rsqrt_e_out;
                        x_addr_r <= 5'd0;
                        g_addr_r <= 5'd0;
                        cnt      <= 6'd0;
                        done_cnt <= 6'd0;
                        state    <= PASS2;
                    end
                end

                PASS2: begin
                    if (cnt < 6'd31) begin
                        x_addr_r <= x_addr_r + 5'd1;
                        g_addr_r <= g_addr_r + 5'd1;
                    end
                    if (cnt >= 6'd1 && cnt <= 6'd32) begin
                        xg_valid <= 1'b1;
                        xg_idx   <= cnt[4:0] - 5'd1;
                    end
                    if (cnt < 6'd63) cnt <= cnt + 6'd1;

                    if (requant_valid_out) begin
                        done_cnt <= done_cnt + 6'd1;
                        if (done_cnt == 6'd31) begin
                            state    <= IDLE;
                            busy_reg <= 1'b0;
                        end
                    end
                end
            endcase
        end
    end

endmodule
