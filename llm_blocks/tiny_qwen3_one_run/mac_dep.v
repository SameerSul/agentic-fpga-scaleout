module mac (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire signed [15:0] a,
    input  wire signed [15:0] b,
    input  wire        valid_in,
    output reg  signed [31:0] acc,
    output reg         valid_out
);

    // Stage 1: register inputs and compute partial products
    // Split a into unsigned low byte and signed high byte
    // a = a_hi_s * 256 + a_lo_u
    // product = a_hi_s*b*256 + a_lo_u*b
    reg signed [23:0] pp_hi_s1;  // signed(a[15:8]) * signed(b), 8s x 16s = 24s
    reg        [23:0] pp_lo_s1;  // unsigned(a[7:0]) * signed(b), stored as 24 bits
    reg valid_s1;

    wire signed [7:0]  a_hi = a[15:8];
    wire        [7:0]  a_lo = a[7:0];
    wire signed [15:0] b_s  = b;

    always @(posedge clk) begin
        if (!rst_n) begin
            pp_hi_s1 <= 24'sd0;
            pp_lo_s1 <= 24'd0;
            valid_s1 <= 1'b0;
        end else begin
            pp_hi_s1 <= a_hi * b_s;
            pp_lo_s1 <= $signed({1'b0, a_lo}) * b_s;
            valid_s1 <= valid_in;
        end
    end

    // Stage 2: combine partial products into full 32-bit product
    reg signed [31:0] product_s2;
    reg valid_s2;

    always @(posedge clk) begin
        if (!rst_n) begin
            product_s2 <= 32'sd0;
            valid_s2   <= 1'b0;
        end else begin
            product_s2 <= ($signed(pp_hi_s1) <<< 8) + $signed(pp_lo_s1);
            valid_s2   <= valid_s1;
        end
    end

    // Stage 3: accumulate
    always @(posedge clk) begin
        if (!rst_n) begin
            acc       <= 32'sd0;
            valid_out <= 1'b0;
        end else if (clear) begin
            acc       <= 32'sd0;
            valid_out <= 1'b0;
        end else begin
            if (valid_s2)
                acc <= acc + product_s2;
            valid_out <= valid_s2;
        end
    end

endmodule
