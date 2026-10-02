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

  reg [1:0] state, next_state;
  localparam IDLE = 0, ACTIVE = 1;

  reg [8:0] col_cnt, next_col_cnt;
  reg [9:0] step, next_step;
  reg [8:0] depth_reg, next_depth_reg;
  reg [8:0] cols_reg, next_cols_reg;

  wire [8:0] a_addr_wire;
  wire [17:0] w_addr_wire;
  wire mac_valid_wire;
  wire col_valid_wire;
  wire mac_clear_wire;
  wire [8:0] col_index_wire;

  assign a_addr = a_addr_wire;
  assign w_addr = w_addr_wire;
  assign mac_valid = mac_valid_wire;
  assign col_valid = col_valid_wire;
  assign mac_clear = mac_clear_wire;
  assign col_index = col_index_wire;
  assign busy = (state == ACTIVE);

  assign a_addr_wire = (state == ACTIVE && step < depth_reg) ? step : 9'b0;
  assign w_addr_wire = (state == ACTIVE && step < depth_reg) ? (col_cnt * depth_reg + step) : 18'b0;
  assign mac_valid_wire = (state == ACTIVE && step > 0 && step < (depth_reg + 1)) ? 1 : 0;
  assign col_valid_wire = (state == ACTIVE && step == (depth_reg + 4)) ? 1 : 0;
  assign mac_clear_wire = (state == ACTIVE && step == (depth_reg + 5)) ? 1 : 0;
  assign col_index_wire = col_cnt;

  always @(posedge clk) begin
    if (~rst_n) begin
      state <= IDLE;
      col_cnt <= 0;
      step <= 0;
      depth_reg <= 0;
      cols_reg <= 0;
    end else begin
      state <= next_state;
      col_cnt <= next_col_cnt;
      step <= next_step;
      depth_reg <= next_depth_reg;
      cols_reg <= next_cols_reg;
    end
  end

  always @(*) begin
    next_state = state;
    next_col_cnt = col_cnt;
    next_step = step + 1;
    next_depth_reg = depth_reg;
    next_cols_reg = cols_reg;

    case (state)
      IDLE: begin
        if (start) begin
          next_state = ACTIVE;
          next_col_cnt = 0;
          next_step = 0;
          next_depth_reg = depth;
          next_cols_reg = cols;
        end else begin
          next_step = 0;
        end
      end
      ACTIVE: begin
        if (step == (depth_reg + 5)) begin
          next_step = 0;
          if (col_cnt + 1 < cols_reg) begin
            next_col_cnt = col_cnt + 1;
          end else begin
            next_state = IDLE;
          end
        end
      end
    endcase
  end

endmodule
