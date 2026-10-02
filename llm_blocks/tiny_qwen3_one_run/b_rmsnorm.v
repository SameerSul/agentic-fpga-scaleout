module rmsnorm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [39:0] eps,
    input  wire [17:0] scale_o,
    input  wire [ 6:0] shift_o,
    output reg  [ 5:0] x_addr,
    input  wire [15:0] x_data,
    output reg  [ 5:0] g_addr,
    input  wire [15:0] g_data,
    output reg         o_valid,
    output reg  [ 5:0] o_index,
    output reg  [15:0] o_data,
    output reg         busy
);

    reg [39:0] ssq;
    reg [16:0] rs_m;
    reg [ 5:0] rs_e;

    reg [17:0] saved_scale;
    reg [ 6:0] saved_shift;

    localparam ST_IDLE  = 3'd0;
    localparam ST_P1    = 3'd1;
    localparam ST_RSQRT = 3'd2;
    localparam ST_P2    = 3'd3;

    reg [2:0] state;

    reg [6:0] issue_ctr;
    reg [6:0] accum_ctr;
    reg [2:0] p1_pipe_v;
    reg [5:0] p1_pipe_idx [0:2];

    reg signed [31:0] sq_reg;

    reg         rsqrt_vin;
    wire [16:0] rsqrt_y;
    wire [ 5:0] rsqrt_e_out;
    wire        rsqrt_vout;

    rsqrt rsqrt_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .x        (ssq),
        .valid_in (rsqrt_vin),
        .y        (rsqrt_y),
        .e        (rsqrt_e_out),
        .valid_out(rsqrt_vout)
    );

    reg [16:0] p2_m;
    reg [ 5:0] p2_e;
    reg [ 6:0] p2_issue_ctr;
    reg        p2_issuing;

    reg        m1_v;  reg [5:0] m1_idx;
    reg        m2_v;  reg [5:0] m2_idx;

    reg        xg_v;  reg [5:0] xg_idx;
    reg signed [31:0] xg_prod;

    reg        xgm_v; reg [5:0] xgm_idx;
    reg signed [48:0] xgm_full;

    reg        t_v;   reg [5:0] t_idx;
    reg signed [31:0] t_val;

    reg        rq_vin;
    reg signed [31:0] rq_acc_in;
    wire       rq_vout;
    wire signed [15:0] rq_qout;
    wire       rq_sat;

    reg [5:0] rq_idx_pipe [0:9];
    integer   ii;

    requant requant_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .acc_in   (rq_acc_in),
        .scale    (saved_scale),
        .shift    (saved_shift),
        .valid_in (rq_vin),
        .q_out    (rq_qout),
        .sat      (rq_sat),
        .valid_out(rq_vout)
    );

    wire signed [15:0] x_data_s = $signed(x_data);
    wire signed [15:0] g_data_s = $signed(g_data);

    wire signed [31:0] x_sq_s = x_data_s * x_data_s;

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            busy         <= 1'b0;
            o_valid      <= 1'b0;
            o_index      <= 6'd0;
            o_data       <= 16'd0;
            ssq          <= 40'd0;
            rs_m         <= 17'd0;
            rs_e         <= 6'd0;
            x_addr       <= 6'd0;
            g_addr       <= 6'd0;
            issue_ctr    <= 7'd0;
            accum_ctr    <= 7'd0;
            p1_pipe_v    <= 3'b000;
            p1_pipe_idx[0] <= 6'd0;
            p1_pipe_idx[1] <= 6'd0;
            p1_pipe_idx[2] <= 6'd0;
            sq_reg       <= 32'd0;
            rsqrt_vin    <= 1'b0;
            saved_scale  <= 18'd0;
            saved_shift  <= 7'd0;
            p2_issue_ctr <= 7'd0;
            p2_issuing   <= 1'b0;
            p2_m         <= 17'd0;
            p2_e         <= 6'd0;
            m1_v <= 1'b0; m1_idx <= 6'd0;
            m2_v <= 1'b0; m2_idx <= 6'd0;
            xg_v <= 1'b0; xg_idx <= 6'd0; xg_prod <= 32'd0;
            xgm_v <= 1'b0; xgm_idx <= 6'd0; xgm_full <= 49'd0;
            t_v <= 1'b0; t_idx <= 6'd0; t_val <= 32'd0;
            rq_vin <= 1'b0; rq_acc_in <= 32'd0;
            for (ii = 0; ii < 10; ii = ii + 1)
                rq_idx_pipe[ii] <= 6'd0;
        end else begin
            rsqrt_vin <= 1'b0;
            o_valid   <= 1'b0;
            rq_vin    <= 1'b0;

            for (ii = 9; ii > 0; ii = ii - 1)
                rq_idx_pipe[ii] <= rq_idx_pipe[ii-1];

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        busy        <= 1'b1;
                        saved_scale <= scale_o;
                        saved_shift <= shift_o;
                        ssq         <= eps;
                        x_addr      <= 6'd0;
                        issue_ctr   <= 7'd1;
                        accum_ctr   <= 7'd0;
                        p1_pipe_v      <= 3'b001;
                        p1_pipe_idx[0] <= 6'd0;
                        p1_pipe_idx[1] <= 6'd0;
                        p1_pipe_idx[2] <= 6'd0;
                        state       <= ST_P1;
                    end
                end

                ST_P1: begin
                    p1_pipe_v[2]   <= p1_pipe_v[1];
                    p1_pipe_idx[2] <= p1_pipe_idx[1];
                    p1_pipe_v[1]   <= p1_pipe_v[0];
                    p1_pipe_idx[1] <= p1_pipe_idx[0];
                    p1_pipe_v[0]   <= 1'b0;

                    if (issue_ctr <= 7'd63) begin
                        x_addr         <= issue_ctr[5:0];
                        p1_pipe_v[0]   <= 1'b1;
                        p1_pipe_idx[0] <= issue_ctr[5:0];
                        issue_ctr      <= issue_ctr + 7'd1;
                    end

                    if (p1_pipe_v[1]) begin
                        sq_reg <= x_sq_s;
                    end

                    if (p1_pipe_v[2]) begin
                        ssq       <= ssq + {{8{sq_reg[31]}}, sq_reg};
                        accum_ctr <= accum_ctr + 7'd1;
                        if (accum_ctr == 7'd63) begin
                            state <= ST_RSQRT;
                        end
                    end
                end

                ST_RSQRT: begin
                    rsqrt_vin <= 1'b1;
                    if (rsqrt_vout) begin
                        rs_m <= rsqrt_y;
                        rs_e <= rsqrt_e_out;
                        p2_m <= rsqrt_y;
                        p2_e <= rsqrt_e_out;
                        x_addr       <= 6'd0;
                        g_addr       <= 6'd0;
                        p2_issue_ctr <= 7'd1;
                        p2_issuing   <= 1'b1;
                        m1_v         <= 1'b1;
                        m1_idx       <= 6'd0;
                        state        <= ST_P2;
                    end
                end

                ST_P2: begin
                    if (p2_issuing) begin
                        if (p2_issue_ctr <= 7'd63) begin
                            x_addr       <= p2_issue_ctr[5:0];
                            g_addr       <= p2_issue_ctr[5:0];
                            m1_v         <= 1'b1;
                            m1_idx       <= p2_issue_ctr[5:0];
                            p2_issue_ctr <= p2_issue_ctr + 7'd1;
                        end else begin
                            m1_v       <= 1'b0;
                            p2_issuing <= 1'b0;
                        end
                    end else begin
                        m1_v <= 1'b0;
                    end

                    m2_v   <= m1_v;
                    m2_idx <= m1_idx;

                    xg_v   <= m2_v;
                    xg_idx <= m2_idx;
                    if (m2_v) begin
                        xg_prod <= x_data_s * g_data_s;
                    end

                    xgm_v   <= xg_v;
                    xgm_idx <= xg_idx;
                    if (xg_v) begin
                        xgm_full <= $signed({{17{xg_prod[31]}}, xg_prod}) * $signed({1'b0, p2_m});
                    end

                    t_v   <= xgm_v;
                    t_idx <= xgm_idx;
                    if (xgm_v) begin
                        t_val <= $signed(xgm_full) >>> (p2_e + 6'd2);
                    end

                    rq_vin <= t_v;
                    if (t_v) begin
                        rq_acc_in      <= t_val;
                        rq_idx_pipe[0] <= t_idx;
                    end

                    if (rq_vout) begin
                        o_valid <= 1'b1;
                        o_index <= rq_idx_pipe[9];
                        o_data  <= rq_qout;
                        if (rq_idx_pipe[9] == 6'd63) begin
                            busy  <= 1'b0;
                            state <= ST_IDLE;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
