module silu(
    input clk,
    input rst_n,
    input signed [12:0] x,
    input valid_in,
    output signed [12:0] y,
    output valid_out
);

    wire signed [12:0] x_abs;
    wire signed [12:0] x_clamped;
    wire signed [12:0] neg_x;
    
    assign x_abs = (x < 0) ? -x : x;
    assign x_clamped = (x_abs > 13'd4095) ? 13'd4095 : x_abs;
    assign neg_x = -x_clamped;
    
    wire [15:0] exp_out;
    wire exp_valid;
    
    expu expu_inst(
        .clk(clk),
        .rst_n(rst_n),
        .x(neg_x),
        .valid_in(valid_in),
        .y(exp_out),
        .valid_out(exp_valid)
    );
    
    wire [23:0] d;
    assign d = {8'b0, 16'h8000} + {8'b0, exp_out};
    
    wire [16:0] recip_m;
    wire [4:0] recip_k;
    wire recip_valid;
    
    recip recip_inst(
        .clk(clk),
        .rst_n(rst_n),
        .x(d),
        .valid_in(exp_valid),
        .y(recip_m),
        .k(recip_k),
        .valid_out(recip_valid)
    );
    
    reg signed [12:0] x_p0, x_p1, x_p2, x_p3, x_p4, x_p5, x_p6, x_p7;
    reg [15:0] e_p0, e_p1, e_p2;
    reg [47:0] product_p0;
    reg [4:0] k_p0;
    reg [15:0] sig_p0;
    reg valid_p0, valid_p1;
    
    wire [15:0] num;
    assign num = (x_p5 >= 0) ? 16'h8000 : e_p2;
    
    wire [30:0] num_shifted;
    assign num_shifted = num << 15;
    
    wire [47:0] product;
    assign product = num_shifted * recip_m;
    
    wire [5:0] shift_amount;
    assign shift_amount = 6'd40 - {1'b0, k_p0};
    
    wire [47:0] product_shifted;
    assign product_shifted = product_p0 >> shift_amount;
    
    wire [15:0] sig;
    assign sig = (product_shifted > 16'h8000) ? 16'h8000 : product_shifted[15:0];
    
    always @(posedge clk) begin
        if (!rst_n) begin
            x_p0 <= 0; x_p1 <= 0; x_p2 <= 0; x_p3 <= 0; x_p4 <= 0; x_p5 <= 0; x_p6 <= 0; x_p7 <= 0;
            e_p0 <= 0; e_p1 <= 0; e_p2 <= 0;
            product_p0 <= 0;
            k_p0 <= 0;
            sig_p0 <= 0;
            valid_p0 <= 0;
            valid_p1 <= 0;
        end else begin
            x_p0 <= x;
            x_p1 <= x_p0;
            x_p2 <= x_p1;
            x_p3 <= x_p2;
            x_p4 <= x_p3;
            x_p5 <= x_p4;
            x_p6 <= x_p5;
            x_p7 <= x_p6;
            
            e_p0 <= exp_out;
            e_p1 <= e_p0;
            e_p2 <= e_p1;
            
            product_p0 <= product;
            k_p0 <= recip_k;
            sig_p0 <= sig;
            valid_p0 <= recip_valid;
            valid_p1 <= valid_p0;
        end
    end
    
    wire signed [28:0] product_xy;
    assign product_xy = {{16{x_p7[12]}}, x_p7} * sig_p0;
    
    wire signed [12:0] y_result;
    assign y_result = product_xy >>> 15;
    
    assign y = y_result;
    assign valid_out = valid_p1;

endmodule
