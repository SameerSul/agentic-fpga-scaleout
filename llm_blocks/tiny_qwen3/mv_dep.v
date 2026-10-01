module matvec(
  input clk,
  input rst_n,
  input start,
  input [8:0] depth,
  input [8:0] cols,
  output [8:0] a_addr,
  output [17:0] w_addr,
  output mac_valid,
  output mac_clear,
  output col_valid,
  output [8:0] col_index,
  output busy
);

  reg [8:0] col_count;
  reg [9:0] cycle_in_col;
  reg busy_r;

  always @(posedge clk) begin
    if (~rst_n) begin
      col_count <= 0;
      cycle_in_col <= 0;
      busy_r <= 0;
    end else if (~busy_r) begin
      if (start) begin
        col_count <= 0;
        cycle_in_col <= 0;
        busy_r <= 1;
      end
    end else begin
      if (cycle_in_col == {1'b0, depth} + 10'd5) begin
        cycle_in_col <= 0;
        if (col_count == cols - 1) begin
          busy_r <= 0;
        end else begin
          col_count <= col_count + 1;
        end
      end else begin
        cycle_in_col <= cycle_in_col + 1;
      end
    end
  end

  assign a_addr = (busy_r && cycle_in_col < {1'b0, depth}) ? cycle_in_col[8:0] : 9'b0;
  assign w_addr = (busy_r && cycle_in_col < {1'b0, depth}) ? col_count * depth + cycle_in_col : 18'b0;
  assign mac_valid = busy_r && cycle_in_col >= 10'd1 && cycle_in_col <= {1'b0, depth};
  assign col_valid = busy_r && cycle_in_col == {1'b0, depth} + 10'd5;
  assign mac_clear = busy_r && cycle_in_col == {1'b0, depth} + 10'd5;
  assign col_index = col_count;
  assign busy = busy_r;

endmodule
