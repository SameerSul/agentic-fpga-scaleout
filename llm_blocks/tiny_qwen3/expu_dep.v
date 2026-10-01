module expu(
  input clk,
  input rst_n,
  input signed [12:0] x,
  input valid_in,
  output reg [15:0] y,
  output reg valid_out
);

  wire signed [29:0] product = x * 18'sd94548;
  wire signed [13:0] t = product >>> 16;

  reg signed [13:0] t_r1;
  reg valid_r1;

  always @(posedge clk) begin
    if (!rst_n) begin
      t_r1 <= 14'sb0;
      valid_r1 <= 1'b0;
    end else begin
      t_r1 <= t;
      valid_r1 <= valid_in;
    end
  end

  wire signed [5:0] n = t_r1 >>> 8;
  wire [7:0] f = t_r1[7:0];

  wire [15:0] rom_val;
  exp_rom rom_inst(
    .idx(f),
    .val(rom_val)
  );

  reg [15:0] rom_val_r2;
  reg [5:0] sh_r2;
  reg valid_r2;

  always @(posedge clk) begin
    if (!rst_n) begin
      rom_val_r2 <= 16'b0;
      sh_r2 <= 6'b0;
      valid_r2 <= 1'b0;
    end else begin
      rom_val_r2 <= rom_val;
      sh_r2 <= -n;
      valid_r2 <= valid_r1;
    end
  end

  wire [15:0] shifted = (sh_r2 > 6'd15) ? 16'b0 : (rom_val_r2 >> sh_r2);

  always @(posedge clk) begin
    if (!rst_n) begin
      y <= 16'b0;
      valid_out <= 1'b0;
    end else begin
      y <= shifted;
      valid_out <= valid_r2;
    end
  end

endmodule
