module projn(
  input clk, rst_n, start,
  input [8:0] depth, cols,
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

  localparam IDLE = 0, RUN = 1;
  
  reg [0:0] state;
  reg [8:0] col_group;
  reg [9:0] group_cycle;
  reg [56:0] col_params [15:0];
  
  wire [8:0] num_groups = (cols + 15) >> 4;
  
  wire [15:0] mac_valid_out;
  wire signed [31:0] mac_acc [15:0];
  
  wire [15:0] requant_valid_out;
  wire signed [15:0] requant_q_out [15:0];
  
  always @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      col_group <= 0;
      group_cycle <= 0;
    end else begin
      if (state == RUN && group_cycle >= 1 && group_cycle <= 16) begin
        col_params[group_cycle - 1] <= c_data;
      end
      
      if (state == IDLE && start) begin
        state <= RUN;
        col_group <= 0;
        group_cycle <= 0;
      end else if (state == RUN) begin
        if (group_cycle < depth + 30) begin
          group_cycle <= group_cycle + 1;
        end else begin
          if (col_group == num_groups - 1) begin
            state <= IDLE;
          end else begin
            col_group <= col_group + 1;
            group_cycle <= 0;
          end
        end
      end
    end
  end
  
  wire [8:0] row_addr = (group_cycle >= 1 && group_cycle <= depth && state == RUN) ? (group_cycle - 1) : 0;
  
  assign a_addr = row_addr;
  assign w_addr = col_group * depth + row_addr;
  assign c_addr = (state == RUN && group_cycle < 16) ? (col_group * 16 + group_cycle) : 0;
  
  wire [4:0] output_lane = group_cycle - (depth + 14);
  wire [4:0] safe_output_lane = (group_cycle >= depth + 14 && group_cycle <= depth + 29) ? output_lane : 5'b0;
  
  assign o_valid = (group_cycle >= depth + 14 && group_cycle <= depth + 29 && state == RUN) &&
                   requant_valid_out[safe_output_lane] && (col_group * 16 + safe_output_lane < cols);
  assign o_index = col_group * 16 + safe_output_lane;
  assign o_data = requant_q_out[safe_output_lane];
  assign busy = (state == RUN);
  
  genvar i;
  generate
    for (i = 0; i < 16; i = i + 1) begin : mac_gen
      mac mac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .clear((group_cycle == 0 && state == RUN)),
        .a(a_data),
        .b(w_data[16*i+15:16*i]),
        .valid_in((group_cycle >= 2 && group_cycle <= depth + 1 && state == RUN)),
        .acc(mac_acc[i]),
        .valid_out(mac_valid_out[i])
      );
    end
  endgenerate
  
  generate
    for (i = 0; i < 16; i = i + 1) begin : requant_gen
      wire [56:0] params = col_params[i];
      wire signed [31:0] bias = params[56:25];
      wire [6:0] shift_val = params[24:18];
      wire [17:0] scale_val = params[17:0];
      wire signed [31:0] acc_plus_bias = mac_acc[i] + bias;
      
      requant req_inst (
        .clk(clk),
        .rst_n(rst_n),
        .acc_in(acc_plus_bias),
        .scale(scale_val),
        .shift(shift_val),
        .valid_in((group_cycle == (depth + 5 + i)) && (state == RUN)),
        .q_out(requant_q_out[i]),
        .sat(),
        .valid_out(requant_valid_out[i])
      );
    end
  endgenerate
  
endmodule
