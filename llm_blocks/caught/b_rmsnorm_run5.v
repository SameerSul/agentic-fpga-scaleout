module rmsnorm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [39:0] eps,
    input  wire [17:0] scale_o,
    input  wire  [6:0] shift_o,
    output reg   [5:0] x_addr,
    input  wire signed [15:0] x_data,
    output reg   [5:0] g_addr,
    input  wire signed [15:0] g_data,
    output reg         o_valid,
    output reg   [5:0] o_index,
    output reg  signed [15:0] o_data,
    output reg         busy
);

    localparam S_IDLE  = 2'd0;
    localparam S_PASS1 = 2'd1;
    localparam S_RSQRT = 2'd2;
    localparam S_PASS2 = 2'd3;

    reg [1:0] state;

    reg [39:0] ssq;

    // Pass 1 counters
    reg [6:0] p1_issue;  // next address to issue
    reg [6:0] p1_accum;  // counts data arrivals

    // Registered square for timing pipeline break
    reg [31:0] x_sq_r;
    reg        x_sq_v;   // valid flag one cycle after x_data arrives

    // rsqrt outputs (registered)
    reg [16:0] rs_m;
    reg  [5:0] rs_e;

    wire [16:0] rsqrt_y;
    wire  [5:0] rsqrt_e;
    wire        rsqrt_valid_out;
    reg         rsqrt_valid_in;

    // Pass 2 pipeline
    reg  [6:0] p2_issue;

    // Stage valid/index shift registers
    reg        p2_s1_v;
    reg  [5:0] p2_s1_idx;
    reg        p2_s2_v;
    reg  [5:0] p2_s2_idx;
    reg        p2_s3_v;
    reg  [5:0] p2_s3_idx;
    reg        p2_s4_v;
    reg  [5:0] p2_s4_idx;
    reg        p2_s5_v;
    reg  [5:0] p2_s5_idx;

    // Pass 2 data pipeline
    reg signed [31:0] p2_xg;
    reg signed [48:0] p2_xgm;
    reg signed [31:0] p2_t;

    // Shift amount for >>> in pass 2
    reg [6:0] shift_amt;

    // requant
    wire signed [15:0] req_q_out;
    wire               req_sat;
    wire               req_valid_out;
    reg                req_valid_in;
    reg signed [31:0]  req_acc_in;

    // Index pipeline through requant (9 stages)
    reg [5:0] req_idx_pipe [0:8];
    integer k;

    reg [17:0] scale_o_r;
    reg  [6:0] shift_o_r;

    // Signed extended activation and gain
    wire signed [31:0] x_ext;
    wire signed [31:0] g_ext;
    assign x_ext = {{16{x_data[15]}}, x_data};
    assign g_ext = {{16{g_data[15]}}, g_data};

    // Combinational square (registered on next cycle for timing break)
    wire [31:0] x_sq_comb;
    assign x_sq_comb = x_ext * x_ext;

    rsqrt rsqrt_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .x        (ssq),
        .valid_in (rsqrt_valid_in),
        .y        (rsqrt_y),
        .e        (rsqrt_e),
        .valid_out(rsqrt_valid_out)
    );

    requant requant_inst (
        .clk      (clk),
        .rst_n    (rst_n),
        .acc_in   (req_acc_in),
        .scale    (scale_o_r),
        .shift    (shift_o_r),
        .valid_in (req_valid_in),
        .q_out    (req_q_out),
        .sat      (req_sat),
        .valid_out(req_valid_out)
    );

    // Index pipeline for requant
    always @(posedge clk) begin
        if (!rst_n) begin
            for (k = 0; k <= 8; k = k + 1)
                req_idx_pipe[k] <= 6'd0;
        end else begin
            req_idx_pipe[0] <= p2_s5_idx;
            for (k = 1; k <= 8; k = k + 1)
                req_idx_pipe[k] <= req_idx_pipe[k-1];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            o_valid        <= 1'b0;
            o_index        <= 6'd0;
            o_data         <= 16'd0;
            ssq            <= 40'd0;
            x_addr         <= 6'd0;
            g_addr         <= 6'd0;
            p1_issue       <= 7'd0;
            p1_accum       <= 7'd0;
            x_sq_r         <= 32'd0;
            x_sq_v         <= 1'b0;
            rsqrt_valid_in <= 1'b0;
            rs_m           <= 17'd0;
            rs_e           <= 6'd0;
            p2_issue       <= 7'd0;
            p2_s1_v        <= 1'b0;
            p2_s1_idx      <= 6'd0;
            p2_s2_v        <= 1'b0;
            p2_s2_idx      <= 6'd0;
            p2_s3_v        <= 1'b0;
            p2_s3_idx      <= 6'd0;
            p2_xg          <= 32'd0;
            p2_s4_v        <= 1'b0;
            p2_s4_idx      <= 6'd0;
            p2_xgm         <= 49'd0;
            p2_s5_v        <= 1'b0;
            p2_s5_idx      <= 6'd0;
            p2_t           <= 32'd0;
            shift_amt      <= 7'd0;
            req_valid_in   <= 1'b0;
            req_acc_in     <= 32'd0;
            scale_o_r      <= 18'd0;
            shift_o_r      <= 7'd0;
        end else begin
            // Defaults
            o_valid        <= 1'b0;
            rsqrt_valid_in <= 1'b0;
            req_valid_in   <= 1'b0;
            x_sq_v         <= 1'b0;

            // Pass 2 pipeline: shift valid/index
            p2_s2_v   <= p2_s1_v;
            p2_s2_idx <= p2_s1_idx;
            p2_s3_v   <= p2_s2_v;
            p2_s3_idx <= p2_s2_idx;
            p2_s4_v   <= p2_s3_v;
            p2_s4_idx <= p2_s3_idx;
            p2_s5_v   <= p2_s4_v;
            p2_s5_idx <= p2_s4_idx;

            // Pass 2 data pipeline
            p2_xg  <= x_ext * g_ext;
            p2_xgm <= $signed(p2_xg) * $signed({1'b0, rs_m});
            p2_t   <= $signed(p2_xgm) >>> shift_amt;

            // Default: s1 not valid
            p2_s1_v <= 1'b0;

            // requant feed from s5
            req_valid_in <= p2_s5_v;
            if (p2_s5_v) begin
                req_acc_in <= p2_t;
            end

            // Capture requant output
            if (req_valid_out) begin
                o_valid <= 1'b1;
                o_index <= req_idx_pipe[8];
                o_data  <= req_q_out;
            end

            // Pipeline stage 1: register x_sq from x_data
            // This fires whenever we're in pass 1 and data is valid (p1_accum in 1..64)
            // We gate with x_sq_v on next cycle to accumulate
            if (state == S_PASS1 && p1_accum >= 7'd1 && p1_accum <= 7'd64) begin
                x_sq_r <= x_sq_comb;
                x_sq_v <= 1'b1;
            end

            // Accumulate registered square (one cycle later)
            if (x_sq_v) begin
                ssq <= ssq + {{8{1'b0}}, x_sq_r};
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy       <= 1'b1;
                        ssq        <= eps;
                        scale_o_r  <= scale_o;
                        shift_o_r  <= shift_o;
                        x_addr     <= 6'd0;
                        g_addr     <= 6'd0;
                        p1_issue   <= 7'd1;
                        p1_accum   <= 7'd0;
                        state      <= S_PASS1;
                    end
                end

                S_PASS1: begin
                    // Issue next address if more remain
                    if (p1_issue < 7'd64) begin
                        x_addr   <= p1_issue[5:0];
                        g_addr   <= p1_issue[5:0];
                        p1_issue <= p1_issue + 7'd1;
                    end

                    p1_accum <= p1_accum + 7'd1;

                    // When p1_accum reaches 65, all 64 squares have been registered
                    // x_sq_v will go high for p1_accum==65 cycle (last square accumulation)
                    // After p1_accum==64 we stop issuing; at p1_accum==65 last x_sq_r accumulates
                    // We need to send rsqrt_valid_in after the last accumulation is done
                    // That happens when x_sq_v goes high for the 64th time, i.e., p1_accum==65
                    if (p1_accum == 7'd65) begin
                        // ssq will be updated this cycle with last x_sq_r
                        // We must fire rsqrt after ssq settles, so fire next cycle
                        // Use a flag: transition to S_RSQRT, fire rsqrt_valid_in there
                        state <= S_RSQRT;
                        // ssq will be correct after this clock edge (x_sq_v fires accumulation)
                        // rsqrt needs ssq stable, so fire valid_in next cycle
                        rsqrt_valid_in <= 1'b1;
                    end
                end

                S_RSQRT: begin
                    if (rsqrt_valid_out) begin
                        rs_m      <= rsqrt_y;
                        rs_e      <= rsqrt_e;
                        shift_amt <= {1'b0, rsqrt_e} + 7'd2;
                        x_addr    <= 6'd0;
                        g_addr    <= 6'd0;
                        p2_issue  <= 7'd1;
                        state     <= S_PASS2;
                    end
                end

                S_PASS2: begin
                    if (p2_issue < 7'd64) begin
                        x_addr   <= p2_issue[5:0];
                        g_addr   <= p2_issue[5:0];
                        p2_issue <= p2_issue + 7'd1;
                    end else begin
                        p2_issue <= p2_issue + 7'd1;
                    end

                    if (p2_issue >= 7'd1 && p2_issue <= 7'd64) begin
                        p2_s1_v   <= 1'b1;
                        p2_s1_idx <= p2_issue[5:0] - 6'd1;
                    end else begin
                        p2_s1_v <= 1'b0;
                    end

                    if (req_valid_out && req_idx_pipe[8] == 6'd63) begin
                        busy  <= 1'b0;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
