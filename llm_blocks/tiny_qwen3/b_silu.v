module silu(
  input clk,
  input rst_n,
  input signed [12:0] x,
  input valid_in,
  output signed [12:0] y,
  output valid_out
);

  wire [12:0] x_abs = x[12] ? (-x) : x;
  wire [12:0] a = (x_abs > 13'd4095) ? 13'd4095 : x_abs;
  wire [12:0] neg_a = -a;

  wire [15:0] expu_y;
  wire expu_valid;
  expu expu_inst (
    .clk(clk),
    .rst_n(rst_n),
    .x(neg_a),
    .valid_in(valid_in),
    .y(expu_y),
    .valid_out(expu_valid)
  );

  wire [23:0] d = {8'b0, expu_y} + 24'd32768;

  wire [16:0] recip_y;
  wire [4:0] recip_k;
  wire recip_valid;
  recip recip_inst (
    .clk(clk),
    .rst_n(rst_n),
    .x(d),
    .valid_in(expu_valid),
    .y(recip_y),
    .k(recip_k),
    .valid_out(recip_valid)
  );

  reg signed [12:0] x_d1, x_d2, x_d3, x_d4, x_d5, x_d6, x_d7, x_d8, x_d9;
  reg [15:0] e_d1, e_d2, e_d3, e_d4, e_d5, e_d6;
  
  always @(posedge clk) begin
    if (!rst_n) begin
      x_d1 <= 0; x_d2 <= 0; x_d3 <= 0; x_d4 <= 0; x_d5 <= 0;
      x_d6 <= 0; x_d7 <= 0; x_d8 <= 0; x_d9 <= 0;
      e_d1 <= 0; e_d2 <= 0; e_d3 <= 0; e_d4 <= 0; e_d5 <= 0; e_d6 <= 0;
    end else begin
      x_d1 <= x;     x_d2 <= x_d1;   x_d3 <= x_d2;   x_d4 <= x_d3;   x_d5 <= x_d4;
      x_d6 <= x_d5;  x_d7 <= x_d6;   x_d8 <= x_d7;   x_d9 <= x_d8;
      e_d1 <= expu_y; e_d2 <= e_d1;  e_d3 <= e_d2;   e_d4 <= e_d3;   e_d5 <= e_d4;  e_d6 <= e_d5;
    end
  end

  wire [15:0] num = (x_d6 >= 0) ? 16'd32768 : e_d3;
  
  reg [31:0] num_shifted_d6;
  reg [16:0] m_d6;
  reg [4:0] k_d6;
  always @(posedge clk) begin
    if (!rst_n) begin
      num_shifted_d6 <= 0;
      m_d6 <= 0;
      k_d6 <= 0;
    end else begin
      num_shifted_d6 <= num << 15;
      m_d6 <= recip_y;
      k_d6 <= recip_k;
    end
  end

  wire [48:0] mul_comb = num_shifted_d6 * m_d6;
  reg [48:0] mul_d7;
  reg [4:0] k_d7;
  always @(posedge clk) begin
    if (!rst_n) begin
      mul_d7 <= 0;
      k_d7 <= 0;
    end else begin
      mul_d7 <= mul_comb;
      k_d7 <= k_d6;
    end
  end

  wire [48:0] sig_unsat = mul_d7 >> (40 - k_d7);
  wire [15:0] sig = (sig_unsat > 49'd32768) ? 16'd32768 : sig_unsat[15:0];
  reg [15:0] sig_d8;
  always @(posedge clk) begin
    if (!rst_n) begin
      sig_d8 <= 0;
    end else begin
      sig_d8 <= sig;
    end
  end

  wire signed [28:0] prod = x_d9 * $signed({1'b0, sig_d8});
  wire signed [12:0] y_result = prod >>> 15;

  reg valid_d1, valid_d2, valid_d3, valid_d4, valid_d5, valid_d6, valid_d7, valid_d8, valid_d9;
  always @(posedge clk) begin
    if (!rst_n) begin
      valid_d1 <= 0; valid_d2 <= 0; valid_d3 <= 0; valid_d4 <= 0; valid_d5 <= 0;
      valid_d6 <= 0; valid_d7 <= 0; valid_d8 <= 0; valid_d9 <= 0;
    end else begin
      valid_d1 <= valid_in;   valid_d2 <= valid_d1;  valid_d3 <= valid_d2;  valid_d4 <= valid_d3;
      valid_d5 <= valid_d4;   valid_d6 <= valid_d5;  valid_d7 <= valid_d6;  valid_d8 <= valid_d7;  valid_d9 <= valid_d8;
    end
  end

  assign y = y_result;
  assign valid_out = valid_d9;

endmodule
