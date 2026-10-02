module projn(
  input clk,
  input rst_n,
  input start,
  input [8:0] depth,
  input [8:0] cols,
  input [17:0] scale,
  input [6:0] shift,
  output [8:0] a_addr,
  input signed [15:0] a_data,
  output [11:0] w_addr,
  input [255:0] w_data,
  output [8:0] c_addr,
  input [56:0] c_data,
  output o_valid,
  output [8:0] o_index,
  output signed [15:0] o_data,
  output busy
);

  wire signed [15:0] w_lane [15:0];
  genvar i;
  generate
    for (i = 0; i < 16; i = i + 1) begin : lanes
      assign w_lane[i] = $signed(w_data[16*i+15:16*i]);
    end
  endgenerate

  wire signed [31:0] col_bias = c_data[56:25];
  wire [6:0] col_shift = c_data[24:18];
  wire [17:0] col_scale = c_data[17:0];

  wire [15:0] mac_vout;
  wire signed [31:0] mac_acc [15:0];
  reg [15:0] mac_clear;
  reg mac_vin;

  generate
    for (i = 0; i < 16; i = i + 1) begin : macs
      mac mac_i(
        .clk(clk),
        .rst_n(rst_n),
        .clear(mac_clear[i]),
        .a(a_data),
        .b(w_lane[i]),
        .valid_in(mac_vin),
        .acc(mac_acc[i]),
        .valid_out(mac_vout[i])
      );
    end
  endgenerate

  reg signed [31:0] rq_acc;
  reg [17:0] rq_scale;
  reg [6:0] rq_shift;
  reg rq_vin;
  wire signed [15:0] rq_out;
  wire rq_vout;

  requant rq_i(
    .clk(clk),
    .rst_n(rst_n),
    .acc_in(rq_acc),
    .scale(rq_scale),
    .shift(rq_shift),
    .valid_in(rq_vin),
    .q_out(rq_out),
    .sat(),
    .valid_out(rq_vout)
  );

  reg [8:0] a_addr_r, c_addr_r;
  reg [11:0] w_addr_r;

  assign a_addr = a_addr_r;
  assign w_addr = w_addr_r;
  assign c_addr = c_addr_r;

  reg o_valid_r;
  reg [8:0] o_index_r;
  reg signed [15:0] o_data_r;

  assign o_valid = o_valid_r;
  assign o_index = o_index_r;
  assign o_data = o_data_r;

  reg [3:0] state;
  localparam IDLE = 0, ROW_ADDR = 1, ROW_WAIT = 2, ROW_FEED = 3, COL_ADDR = 4, COL_WAIT = 5, COL_RQ_LOAD_WAIT = 6, COL_RQ_LOAD = 7, COL_WAIT_RQ = 8;

  reg [8:0] group, row, col, num_groups;
  
  assign busy = (state != IDLE);

  always @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      o_valid_r <= 0;
      mac_vin <= 0;
      rq_vin <= 0;
      a_addr_r <= 0;
      w_addr_r <= 0;
      c_addr_r <= 0;
      o_index_r <= 0;
      o_data_r <= 0;
      mac_clear <= 0;
      group <= 0;
      row <= 0;
      col <= 0;
      num_groups <= 0;
    end else begin
      o_valid_r <= 0;
      mac_clear <= 0;
      mac_vin <= 0;
      rq_vin <= 0;

      case (state)
        IDLE: begin
          if (start) begin
            group <= 0;
            row <= 0;
            col <= 0;
            num_groups <= (cols + 15) >> 4;
            mac_clear <= 16'hFFFF;
            state <= ROW_ADDR;
          end
        end

        ROW_ADDR: begin
          if (row < depth) begin
            a_addr_r <= row;
            w_addr_r <= group * depth + row;
            row <= row + 1;
            state <= ROW_WAIT;
          end else begin
            row <= 0;
            col <= 0;
            state <= COL_ADDR;
          end
        end

        ROW_WAIT: begin
          state <= ROW_FEED;
        end

        ROW_FEED: begin
          mac_vin <= 1;
          state <= ROW_ADDR;
        end

        COL_ADDR: begin
          if (col < 16 && col + group * 16 < cols) begin
            c_addr_r <= col + group * 16;
            state <= COL_WAIT;
          end else begin
            if (group + 1 < num_groups) begin
              group <= group + 1;
              row <= 0;
              col <= 0;
              mac_clear <= 16'hFFFF;
              state <= ROW_ADDR;
            end else begin
              state <= IDLE;
            end
          end
        end

        COL_WAIT: begin
          state <= COL_RQ_LOAD_WAIT;
        end

        COL_RQ_LOAD_WAIT: begin
          state <= COL_RQ_LOAD;
        end

        COL_RQ_LOAD: begin
          rq_acc <= mac_acc[col] + col_bias;
          rq_scale <= col_scale;
          rq_shift <= col_shift;
          rq_vin <= 1;
          state <= COL_WAIT_RQ;
        end

        COL_WAIT_RQ: begin
          if (rq_vout) begin
            o_valid_r <= 1;
            o_index_r <= col + group * 16;
            o_data_r <= rq_out;
            col <= col + 1;
            state <= COL_ADDR;
          end
        end
      endcase
    end
  end

endmodule
