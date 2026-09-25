module mac (
  input                     clk,
  input                     rst_n,
  input                     clear,
  input      signed [7:0] a,
  input      signed [7:0] b,
  input                     valid_in,
  output reg signed [27:0] acc,
  output reg                valid_out
);
  reg signed [15:0] prod;
  reg         vpipe;
  always @(posedge clk) begin
    if (!rst_n) begin
      prod      <= 0;
      vpipe     <= 1'b0;
      acc       <= 0;
      valid_out <= 1'b0;
    end else begin
      prod  <= a * b;
      vpipe <= valid_in;
      if (clear) begin
        acc <= 28'd0;
        valid_out <= 1'b0;
      end else begin
        if (vpipe) acc <= acc + prod;
        valid_out <= vpipe;
      end
    end
  end
endmodule
