module mac_s(
    input clk,
    input rst_n,
    input clear,
    input signed [15:0] a,
    input signed [15:0] b,
    input valid_in,
    output reg signed [36:0] acc,
    output reg valid_out
);

wire signed [7:0]  a_hi = a[15:8];
wire signed [8:0]  a_lo = {1'b0, a[7:0]};

wire signed [23:0] p_hi = a_hi * b;
wire signed [24:0] p_lo = a_lo * b;

reg signed [23:0] p_hi_r;
reg signed [24:0] p_lo_r;
reg valid_r1;

always @(posedge clk) begin
    if (!rst_n) begin
        p_hi_r   <= 24'b0;
        p_lo_r   <= 25'b0;
        valid_r1 <= 1'b0;
    end else begin
        p_hi_r   <= p_hi;
        p_lo_r   <= p_lo;
        valid_r1 <= valid_in;
    end
end

wire signed [32:0] p_hi_ext = {{9{p_hi_r[23]}}, p_hi_r};
wire signed [32:0] p_lo_ext = {{8{p_lo_r[24]}}, p_lo_r};
wire signed [32:0] product_full = (p_hi_ext <<< 8) + p_lo_ext;

reg signed [31:0] product_r;
reg valid_r2;

always @(posedge clk) begin
    if (!rst_n) begin
        product_r <= 32'b0;
        valid_r2  <= 1'b0;
    end else begin
        product_r <= product_full[31:0];
        valid_r2  <= valid_r1;
    end
end

wire signed [36:0] product_ext = {{5{product_r[31]}}, product_r};

always @(posedge clk) begin
    if (!rst_n) begin
        acc       <= 37'b0;
        valid_out <= 1'b0;
    end else if (clear) begin
        acc       <= 37'b0;
        valid_out <= 1'b0;
    end else begin
        valid_out <= valid_r2;
        if (valid_r2)
            acc <= acc + product_ext;
    end
end

endmodule
