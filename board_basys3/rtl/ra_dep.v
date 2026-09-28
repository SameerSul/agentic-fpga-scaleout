module resadd (
  input                    clk,
  input                    rst_n,
  input      signed [7:0] a,
  input      signed [7:0] b,
  input      [17:0] scale_a,
  input      [17:0] scale_b,
  input      [6:0] shift,
  input                    valid_in,
  output reg signed [7:0] y,
  output reg               valid_out
);
  // Registered once per stream would do, but the inputs are held, so the
  // rounding constant is registered each cycle to keep it off the adder.
  reg signed [27:0] pa, pb, rnd_r, v;
  reg v1, v2;
  wire signed [27:0] rnd = (shift == 0) ? 28'sd0 : (28'sd1 <<< (shift - 1));
  wire signed [27:0] r = v >>> shift;
  always @(posedge clk) begin
    if (!rst_n) begin
      pa <= 0; pb <= 0; rnd_r <= 0; v <= 0; v1 <= 1'b0; v2 <= 1'b0;
      y <= 0; valid_out <= 1'b0;
    end else begin
      v1 <= valid_in;
      pa <= a * $signed({1'b0, scale_a});
      pb <= b * $signed({1'b0, scale_b});
      rnd_r <= rnd;
      v2 <= v1;
      v <= pa + pb + rnd_r;
      valid_out <= v2;
      if (r > 28'sd127) y <= 8'sd127;
      else if (r < -28'sd128) y <= -8'sd128;
      else y <= r[7:0];
    end
  end
endmodule
