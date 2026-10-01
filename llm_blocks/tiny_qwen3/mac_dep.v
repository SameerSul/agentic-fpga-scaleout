module mac(
    input clk,
    input rst_n,
    input clear,
    input signed [15:0] a,
    input signed [15:0] b,
    input valid_in,
    output reg signed [31:0] acc,
    output reg valid_out
);

    // Stage 1: Compute partial products
    wire signed [23:0] pp0 = b * $signed({1'b0, a[7:0]});
    wire signed [23:0] pp1 = b * $signed(a[15:8]);
    
    reg signed [23:0] pp0_r;
    reg signed [23:0] pp1_r;
    reg valid_r1;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            pp0_r <= 24'b0;
            pp1_r <= 24'b0;
            valid_r1 <= 1'b0;
        end else begin
            pp0_r <= pp0;
            pp1_r <= pp1;
            valid_r1 <= valid_in;
        end
    end
    
    // Stage 2: Combine partial products
    wire signed [31:0] product = {{8{pp0_r[23]}}, pp0_r} + {pp1_r, 8'b0};
    reg signed [31:0] product_r;
    reg valid_r2;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            product_r <= 32'b0;
            valid_r2 <= 1'b0;
        end else begin
            product_r <= product;
            valid_r2 <= valid_r1;
        end
    end
    
    // Stage 3: Accumulate
    always @(posedge clk) begin
        if (!rst_n) begin
            acc <= 32'b0;
            valid_out <= 1'b0;
        end else if (clear) begin
            acc <= 32'b0;
            valid_out <= 1'b0;
        end else if (valid_r2) begin
            acc <= acc + product_r;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
