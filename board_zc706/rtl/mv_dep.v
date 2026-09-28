module matvec (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [12:0] depth,
  input      [12:0] cols,
  output reg [12:0] a_addr,
  output reg [25:0] w_addr,
  output                   mac_valid,
  output reg               mac_clear,
  output reg               col_valid,
  output reg [12:0] col_index,
  output reg               busy
);
  // Memory reads are asynchronous: data is expected in the same cycle as
  // the address, which is what a LUT RAM gives and what keeps the
  // address and the MAC's valid in step without another pipeline stage.
  localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DRAIN = 2'd2, S_EMIT = 2'd3;
  reg [1:0] state;
  reg [12:0] row;
  reg [12:0] col;
  reg [3:0] drain;
  reg issue, vdly;
  // A running base rather than col*depth. The multiply is the obvious
  // way to write it and it put a 12 by 12 multiplier in the control
  // path: the generic library hid that at 134 MHz and real place and
  // route came back at 69.6 against a 100 MHz target. The base only
  // ever advances by depth, so an adder does the same job.
  reg [25:0] col_base;
  assign mac_valid = vdly;

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      row       <= 0;
      col       <= 0;
      drain     <= 0;
      a_addr    <= 0;
      w_addr    <= 0;
      col_base  <= 0;
      issue     <= 1'b0;
      vdly      <= 1'b0;
      mac_clear <= 1'b0;
      col_valid <= 1'b0;
      col_index <= 0;
      busy      <= 1'b0;
    end else begin
      vdly      <= issue;
      issue     <= 1'b0;
      mac_clear <= 1'b0;
      col_valid <= 1'b0;
      case (state)
        S_IDLE: begin
          if (start && depth != 0 && cols != 0) begin
            state  <= S_RUN;
            row    <= 0;
            col    <= 0;
            busy     <= 1'b1;
            a_addr   <= 0;
            w_addr   <= 0;
            col_base <= 0;
            issue  <= 1'b1;
          end
        end
        S_RUN: begin
          if (row + 1 == depth) begin
            state <= S_DRAIN;
            drain <= 4;
          end else begin
            row       <= row + 1;
            a_addr    <= row + 1;
            w_addr    <= w_addr + 1;
            issue     <= 1'b1;
          end
        end
        S_DRAIN: begin
          if (drain == 0) begin
            state     <= S_EMIT;
            col_valid <= 1'b1;
            col_index <= col;
          end else begin
            drain <= drain - 1;
          end
        end
        S_EMIT: begin
      if (state == S_EMIT) mac_clear <= 1'b1;
          if (col + 1 == cols) begin
            state <= S_IDLE;
            busy  <= 1'b0;
          end else begin
            col       <= col + 1;
            row       <= 0;
            a_addr    <= 0;
            col_base  <= col_base + depth;
            w_addr    <= col_base + depth;
            state     <= S_RUN;
            issue     <= 1'b1;
          end
        end
      endcase
    end
  end
endmodule
