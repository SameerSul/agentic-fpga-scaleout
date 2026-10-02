module requant (
    input clk,
    input rst_n,
    input signed [31:0] acc_in,
    input [17:0] scale,
    input [6:0] shift,
    input valid_in,
    output signed [15:0] q_out,
    output sat,
    output valid_out
);

    wire signed [49:0] pp0 = acc_in * $signed({1'b0, scale[4:0]});
    wire signed [49:0] pp1 = acc_in * $signed({1'b0, scale[9:5]});
    wire signed [49:0] pp2 = acc_in * $signed({1'b0, scale[14:10]});
    wire signed [49:0] pp3 = acc_in * $signed({1'b0, scale[17:15]});

    wire signed [49:0] rounding_term = (shift > 0) ? (50'b1 << (shift - 1)) : 50'b0;

    // Stage 1
    reg signed [49:0] pp0_r, pp1_r, pp2_r, pp3_r;
    reg signed [49:0] rounding_term_r;
    reg [6:0] shift_r;
    reg valid_r;

    always @(posedge clk) begin
        if (~rst_n) begin
            pp0_r <= 50'b0; pp1_r <= 50'b0; pp2_r <= 50'b0; pp3_r <= 50'b0;
            rounding_term_r <= 50'b0; shift_r <= 7'b0; valid_r <= 1'b0;
        end else begin
            pp0_r <= pp0; pp1_r <= pp1; pp2_r <= pp2; pp3_r <= pp3;
            rounding_term_r <= rounding_term; shift_r <= shift; valid_r <= valid_in;
        end
    end

    wire signed [49:0] term01 = (pp0_r) + (pp1_r << 5);
    wire signed [49:0] term23 = (pp2_r << 10) + (pp3_r << 15);

    // Stage 2
    reg signed [49:0] term01_r, term23_r;
    reg signed [49:0] rounding_term_r2;
    reg [6:0] shift_r2;
    reg valid_r2;

    always @(posedge clk) begin
        if (~rst_n) begin
            term01_r <= 50'b0; term23_r <= 50'b0;
            rounding_term_r2 <= 50'b0; shift_r2 <= 7'b0; valid_r2 <= 1'b0;
        end else begin
            term01_r <= term01; term23_r <= term23;
            rounding_term_r2 <= rounding_term_r; shift_r2 <= shift_r; valid_r2 <= valid_r;
        end
    end

    wire signed [49:0] p_comb = term01_r + term23_r;

    // Stage 3: register full product and rounding term
    reg signed [49:0] p_r;
    reg signed [49:0] rounding_term_r3;
    reg [6:0] shift_r3;
    reg valid_r3;

    always @(posedge clk) begin
        if (~rst_n) begin
            p_r <= 50'b0; rounding_term_r3 <= 50'b0; shift_r3 <= 7'b0; valid_r3 <= 1'b0;
        end else begin
            p_r <= p_comb; rounding_term_r3 <= rounding_term_r2; shift_r3 <= shift_r2; valid_r3 <= valid_r2;
        end
    end

    // Lower 25 bits add (with carry out)
    wire [25:0] p_lo_sum = {1'b0, p_r[24:0]} + {1'b0, rounding_term_r3[24:0]};

    // Stage 4: register lower half result, upper half inputs
    reg [25:0] p_lo_r4;
    reg [24:0] p_hi_r4;
    reg [24:0] rounding_term_hi_r4;
    reg [6:0] shift_r4;
    reg valid_r4;

    always @(posedge clk) begin
        if (~rst_n) begin
            p_lo_r4 <= 26'b0; p_hi_r4 <= 25'b0; rounding_term_hi_r4 <= 25'b0;
            shift_r4 <= 7'b0; valid_r4 <= 1'b0;
        end else begin
            p_lo_r4 <= p_lo_sum;
            p_hi_r4 <= p_r[49:25];
            rounding_term_hi_r4 <= rounding_term_r3[49:25];
            shift_r4 <= shift_r3; valid_r4 <= valid_r3;
        end
    end

    // Upper 25 bits add with carry from lower half
    wire [25:0] p_hi_sum = {1'b0, p_hi_r4} + {1'b0, rounding_term_hi_r4} + {25'b0, p_lo_r4[25]};
    wire signed [49:0] p_rounded = {p_hi_sum[24:0], p_lo_r4[24:0]};

    // Stage 5: register rounded product
    reg signed [49:0] p_rounded_r;
    reg [6:0] shift_r5;
    reg valid_r5;

    always @(posedge clk) begin
        if (~rst_n) begin
            p_rounded_r <= 50'b0; shift_r5 <= 7'b0; valid_r5 <= 1'b0;
        end else begin
            p_rounded_r <= p_rounded; shift_r5 <= shift_r4; valid_r5 <= valid_r4;
        end
    end

    wire signed [49:0] shifted = (shift_r5 == 0) ? p_rounded_r : (p_rounded_r >>> shift_r5);

    // Stage 6
    reg signed [49:0] shifted_r;
    reg valid_r6;

    always @(posedge clk) begin
        if (~rst_n) begin
            shifted_r <= 50'b0; valid_r6 <= 1'b0;
        end else begin
            shifted_r <= shifted; valid_r6 <= valid_r5;
        end
    end

    wire is_sat = (shifted_r > 32767) || (shifted_r < -32768);
    wire signed [15:0] q_saturated = (shifted_r > 32767) ? 16'sh7FFF :
                                     (shifted_r < -32768) ? 16'sh8000 :
                                     shifted_r[15:0];

    // Stage 7
    reg signed [15:0] q_out_r;
    reg sat_r;
    reg valid_r7;

    always @(posedge clk) begin
        if (~rst_n) begin
            q_out_r <= 16'b0; sat_r <= 1'b0; valid_r7 <= 1'b0;
        end else begin
            q_out_r <= q_saturated; sat_r <= is_sat; valid_r7 <= valid_r6;
        end
    end

    // Stage 8
    reg signed [15:0] q_out_r8;
    reg sat_r8;
    reg valid_r8;

    always @(posedge clk) begin
        if (~rst_n) begin
            q_out_r8 <= 16'b0; sat_r8 <= 1'b0; valid_r8 <= 1'b0;
        end else begin
            q_out_r8 <= q_out_r; sat_r8 <= sat_r; valid_r8 <= valid_r7;
        end
    end

    // Stage 9
    reg signed [15:0] q_out_r9;
    reg sat_r9;
    reg valid_r9;

    always @(posedge clk) begin
        if (~rst_n) begin
            q_out_r9 <= 16'b0; sat_r9 <= 1'b0; valid_r9 <= 1'b0;
        end else begin
            q_out_r9 <= q_out_r8; sat_r9 <= sat_r8; valid_r9 <= valid_r8;
        end
    end

    assign q_out = q_out_r9;
    assign sat = sat_r9;
    assign valid_out = valid_r9;

endmodule
