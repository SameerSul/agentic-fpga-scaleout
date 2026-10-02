module expu(
    input clk,
    input rst_n,
    input signed [12:0] x,
    input valid_in,
    output reg [15:0] y,
    output reg valid_out
);

    // Stage 1: Multiply by log2(e) and arithmetic shift
    wire signed [17:0] log2e = 18'sd94548;
    wire signed [30:0] prod = x * log2e;
    wire signed [13:0] t_comb = prod >>> 16;
    
    reg signed [13:0] t_s1;
    reg valid_s1;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            t_s1 <= 0;
            valid_s1 <= 0;
        end else begin
            t_s1 <= t_comb;
            valid_s1 <= valid_in;
        end
    end
    
    // Stage 2: Split integer and fractional parts, look up ROM
    wire signed [13:0] n_comb = t_s1 >>> 8;
    wire [7:0] f_comb = t_s1 - (n_comb << 8);
    wire [4:0] sh_comb = -n_comb;
    
    wire [15:0] rom_val_comb;
    exp_rom rom_inst(
        .idx(f_comb),
        .val(rom_val_comb)
    );
    
    reg [15:0] rom_val_s2;
    reg [4:0] sh_s2;
    reg valid_s2;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            rom_val_s2 <= 0;
            sh_s2 <= 0;
            valid_s2 <= 0;
        end else begin
            rom_val_s2 <= rom_val_comb;
            sh_s2 <= sh_comb;
            valid_s2 <= valid_s1;
        end
    end
    
    // Stage 3: Right shift the result
    always @(posedge clk) begin
        if (!rst_n) begin
            y <= 0;
            valid_out <= 0;
        end else begin
            if (sh_s2 > 15) begin
                y <= 0;
            end else begin
                y <= rom_val_s2 >> sh_s2;
            end
            valid_out <= valid_s2;
        end
    end

endmodule
