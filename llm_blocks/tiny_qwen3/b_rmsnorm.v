module rmsnorm (
  input clk,
  input rst_n,
  input start,
  input [39:0] eps,
  input [17:0] scale_o,
  input [6:0] shift_o,
  output reg [5:0] x_addr,
  input signed [15:0] x_data,
  output reg [5:0] g_addr,
  input signed [15:0] g_data,
  output reg o_valid,
  output reg [5:0] o_index,
  output signed [15:0] o_data,
  output reg busy,
  output reg [39:0] ssq,
  output reg [16:0] rs_m,
  output reg [5:0] rs_e
);

  localparam IDLE = 0, READ = 1, WAIT_RSQRT = 2, PROCESS = 3;

  reg [2:0] state;
  reg [6:0] read_idx;

  reg signed [15:0] x_store [0:63];
  reg signed [15:0] g_store [0:63];

  reg [6:0] proc_idx;

  // Pipeline register for squaring to break the long combinational path
  reg [31:0] x_sq_reg;
  reg x_sq_valid;

  // Stage A0: fetch x/g from store arrays
  reg signed [15:0] x_fetch_reg;
  reg signed [15:0] g_fetch_reg;
  reg xg_fetch_valid;
  reg [5:0] xg_fetch_index;

  // Stage A1: x*g
  reg signed [31:0] xg_prod_reg;
  reg xg_valid_reg;
  reg [5:0] xg_index_reg;

  // Stage B: xg * m
  reg signed [48:0] xgm_prod_reg;
  reg xgm_valid_reg;
  reg [5:0] xgm_index_reg;

  // Stage C: shift xgm >>> (e+2)
  reg signed [31:0] t_val_reg;
  reg t_valid_reg;
  reg [5:0] t_index_reg;

  // Requant pipeline index tracking (9 stages)
  reg [5:0] index_pipe [0:8];
  reg valid_pipe [0:8];
  integer i;

  wire [16:0] rsqrt_y;
  wire [5:0] rsqrt_e;
  wire rsqrt_valid_out;

  rsqrt rsqrt_inst (
    .clk(clk),
    .rst_n(rst_n),
    .x(ssq),
    .valid_in(state == WAIT_RSQRT),
    .y(rsqrt_y),
    .e(rsqrt_e),
    .valid_out(rsqrt_valid_out)
  );

  wire signed [48:0] xgm_prod_comb = xg_prod_reg * $signed({1'b0, rs_m});
  wire [6:0] t_shift = rs_e + 7'd2;
  wire signed [31:0] t_val_comb = xgm_prod_reg >>> t_shift;

  wire signed [15:0] requant_q;
  wire requant_sat;
  wire requant_valid_out;

  requant requant_inst (
    .clk(clk),
    .rst_n(rst_n),
    .acc_in(t_val_reg),
    .scale(scale_o),
    .shift(shift_o),
    .valid_in(t_valid_reg),
    .q_out(requant_q),
    .sat(requant_sat),
    .valid_out(requant_valid_out)
  );

  reg signed [15:0] o_data_reg;
  assign o_data = o_data_reg;

  always @(posedge clk) begin
    if (!rst_n) begin
      state <= IDLE;
      busy <= 0;
      o_valid <= 0;
      read_idx <= 0;
      proc_idx <= 0;
      x_addr <= 0;
      g_addr <= 0;
      ssq <= 0;
      rs_m <= 0;
      rs_e <= 0;
      o_index <= 0;
      o_data_reg <= 0;
      x_sq_reg <= 0;
      x_sq_valid <= 0;
      x_fetch_reg <= 0;
      g_fetch_reg <= 0;
      xg_fetch_valid <= 0;
      xg_fetch_index <= 0;
      xg_prod_reg <= 0;
      xg_valid_reg <= 0;
      xg_index_reg <= 0;
      xgm_prod_reg <= 0;
      xgm_valid_reg <= 0;
      xgm_index_reg <= 0;
      t_val_reg <= 0;
      t_valid_reg <= 0;
      t_index_reg <= 0;
      for (i = 0; i < 9; i = i + 1) begin
        valid_pipe[i] <= 0;
        index_pipe[i] <= 0;
      end
    end else begin
      // Stage A0: fetch from store (registered read from arrays)
      if (state == PROCESS && proc_idx < 64) begin
        x_fetch_reg    <= x_store[proc_idx[5:0]];
        g_fetch_reg    <= g_store[proc_idx[5:0]];
        xg_fetch_valid <= 1;
        xg_fetch_index <= proc_idx[5:0];
      end else begin
        x_fetch_reg    <= 0;
        g_fetch_reg    <= 0;
        xg_fetch_valid <= 0;
        xg_fetch_index <= 0;
      end

      // Stage A1: multiply fetched x and g
      if (xg_fetch_valid) begin
        xg_prod_reg  <= x_fetch_reg * g_fetch_reg;
        xg_valid_reg <= 1;
        xg_index_reg <= xg_fetch_index;
      end else begin
        xg_prod_reg  <= 0;
        xg_valid_reg <= 0;
        xg_index_reg <= 0;
      end

      // Stage B: register xg*m
      xgm_prod_reg  <= xg_valid_reg ? xgm_prod_comb : 49'sd0;
      xgm_valid_reg <= xg_valid_reg;
      xgm_index_reg <= xg_index_reg;

      // Stage C: register shifted value
      t_val_reg   <= xgm_valid_reg ? t_val_comb : 32'sd0;
      t_valid_reg <= xgm_valid_reg;
      t_index_reg <= xgm_index_reg;

      // Shift requant index pipeline
      valid_pipe[0] <= t_valid_reg;
      index_pipe[0] <= t_index_reg;
      for (i = 1; i < 9; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        index_pipe[i] <= index_pipe[i - 1];
      end

      // Pipeline stage 1: register x_data squared (breaks long timing path)
      if (state == READ && read_idx >= 2 && read_idx <= 65) begin
        x_sq_reg   <= x_data * x_data;
        x_sq_valid <= 1;
      end else begin
        x_sq_reg   <= 0;
        x_sq_valid <= 0;
      end

      case (state)
        IDLE: begin
          if (start) begin
            state    <= READ;
            busy     <= 1;
            read_idx <= 0;
            ssq      <= eps;
            o_valid  <= 0;
          end
        end

        READ: begin
          if (read_idx < 65) begin
            x_addr   <= read_idx[5:0];
            g_addr   <= read_idx[5:0];
            read_idx <= read_idx + 1;
          end else if (read_idx < 66) begin
            read_idx <= read_idx + 1;
          end else begin
            state <= WAIT_RSQRT;
          end

          // Store x and g when data is valid (registered read: data[i] at read_idx = i+2)
          if (read_idx >= 2 && read_idx <= 65) begin
            x_store[read_idx - 2] <= x_data;
            g_store[read_idx - 2] <= g_data;
          end

          // Accumulate ssq one cycle after squaring (pipeline stage 2)
          if (x_sq_valid) begin
            ssq <= ssq + x_sq_reg;
          end
        end

        WAIT_RSQRT: begin
          if (rsqrt_valid_out) begin
            rs_m     <= rsqrt_y;
            rs_e     <= rsqrt_e;
            state    <= PROCESS;
            proc_idx <= 0;
          end
        end

        PROCESS: begin
          if (proc_idx < 64) begin
            proc_idx <= proc_idx + 1;
          end else if (!xg_fetch_valid && !xg_valid_reg && !xgm_valid_reg && !t_valid_reg && !valid_pipe[8]) begin
            state <= IDLE;
            busy  <= 0;
          end
        end
      endcase

      if (requant_valid_out) begin
        o_data_reg <= requant_q;
        o_index    <= index_pipe[8];
        o_valid    <= 1;
      end else begin
        o_valid <= 0;
      end
    end
  end

endmodule
