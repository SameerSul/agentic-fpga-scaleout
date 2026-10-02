module recip(
  input wire clk,
  input wire rst_n,
  input wire [23:0] x,
  input wire valid_in,
  output wire [16:0] y,
  output wire [4:0] k,
  output wire valid_out
);

  wire [4:0] k_s1;
  wire [23:0] xn_s1;

  assign k_s1 = (x[23] ? 5'd0 :
                 x[22] ? 5'd1 :
                 x[21] ? 5'd2 :
                 x[20] ? 5'd3 :
                 x[19] ? 5'd4 :
                 x[18] ? 5'd5 :
                 x[17] ? 5'd6 :
                 x[16] ? 5'd7 :
                 x[15] ? 5'd8 :
                 x[14] ? 5'd9 :
                 x[13] ? 5'd10 :
                 x[12] ? 5'd11 :
                 x[11] ? 5'd12 :
                 x[10] ? 5'd13 :
                 x[9] ? 5'd14 :
                 x[8] ? 5'd15 :
                 x[7] ? 5'd16 :
                 x[6] ? 5'd17 :
                 x[5] ? 5'd18 :
                 x[4] ? 5'd19 :
                 x[3] ? 5'd20 :
                 x[2] ? 5'd21 :
                 x[1] ? 5'd22 :
                 x[0] ? 5'd23 : 5'd24);

  assign xn_s1 = x << k_s1;

  reg [4:0] k_s2;
  reg [23:0] xn_s2;
  reg valid_s2;

  always @(posedge clk) begin
    if (!rst_n) begin
      k_s2 <= 5'd0;
      xn_s2 <= 24'd0;
      valid_s2 <= 1'b0;
    end else begin
      k_s2 <= k_s1;
      xn_s2 <= xn_s1;
      valid_s2 <= valid_in;
    end
  end

  wire [7:0] idx;
  wire [16:0] rom_out;

  assign idx = xn_s2[22:15];

  recip_rom rom_inst(
    .idx(idx),
    .val(rom_out)
  );

  reg [16:0] y_s3;
  reg [4:0] k_s3;
  reg valid_s3;

  always @(posedge clk) begin
    if (!rst_n) begin
      y_s3 <= 17'd0;
      k_s3 <= 5'd0;
      valid_s3 <= 1'b0;
    end else begin
      if (xn_s2 == 24'd0) begin
        y_s3 <= {17{1'b1}};
        k_s3 <= 5'd0;
      end else begin
        y_s3 <= rom_out;
        k_s3 <= k_s2;
      end
      valid_s3 <= valid_s2;
    end
  end

  reg [16:0] y_s4;
  reg [4:0] k_s4;
  reg valid_s4;

  always @(posedge clk) begin
    if (!rst_n) begin
      y_s4 <= 17'd0;
      k_s4 <= 5'd0;
      valid_s4 <= 1'b0;
    end else begin
      y_s4 <= y_s3;
      k_s4 <= k_s3;
      valid_s4 <= valid_s3;
    end
  end

  assign y = y_s4;
  assign k = k_s4;
  assign valid_out = valid_s4;

endmodule
