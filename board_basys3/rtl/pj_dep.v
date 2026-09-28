module proj (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [6:0] depth,
  input      [6:0] cols,
  input      [17:0] scale,
  input      [6:0] shift,
  output     [6:0] a_addr,
  input      signed [7:0] a_data,
  output     [13:0] w_addr,
  input      signed [7:0] w_data,
  output reg               o_valid,
  output reg [6:0] o_index,
  output reg signed [7:0] o_data,
  output reg               busy
);
  reg  mv_start;
  wire mv_valid, mv_clear, mv_colv, mv_busy;
  wire [6:0] mv_coli;
  matvec mv (.clk(clk), .rst_n(rst_n), .start(mv_start), .depth(depth),
             .cols(cols), .a_addr(a_addr), .w_addr(w_addr),
             .mac_valid(mv_valid), .mac_clear(mv_clear),
             .col_valid(mv_colv), .col_index(mv_coli), .busy(mv_busy));
  wire signed [23:0] acc;
  wire mac_vout;
  mac mc (.clk(clk), .rst_n(rst_n), .clear(mv_clear), .a(a_data),
          .b(w_data), .valid_in(mv_valid), .acc(acc),
          .valid_out(mac_vout));
  reg  rq_vin;
  reg  signed [23:0] rq_acc;
  reg  [6:0] rq_idx;
  wire signed [7:0] rq_q;
  wire rq_sat, rq_vout;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc), .scale(scale),
              .shift(shift), .valid_in(rq_vin), .q_out(rq_q),
              .sat(rq_sat), .valid_out(rq_vout));
  reg [6:0] idx_pipe [0:5];
  reg [7:0] outst;
  reg ran;
  integer k;

  always @(posedge clk) begin
    if (!rst_n) begin
      mv_start <= 1'b0; rq_vin <= 1'b0; rq_acc <= 0; rq_idx <= 0;
      outst <= 0; ran <= 1'b0; busy <= 1'b0; o_valid <= 1'b0;
      o_index <= 0; o_data <= 0;
      for (k = 0; k <= 5; k = k + 1) idx_pipe[k] <= 0;
    end else begin
      mv_start <= 1'b0;
      rq_vin   <= 1'b0;
      o_valid  <= 1'b0;
      idx_pipe[0] <= rq_idx;
      for (k = 1; k <= 5; k = k + 1) idx_pipe[k] <= idx_pipe[k-1];
      if (mv_busy) ran <= 1'b1;
      case ({rq_vin, rq_vout})
        2'b10: outst <= outst + 1;
        2'b01: outst <= outst - 1;
        default: ;
      endcase
      if (mv_colv) begin
        rq_acc <= acc; rq_idx <= mv_coli; rq_vin <= 1'b1;
      end
      if (rq_vout) begin
        o_valid <= 1'b1; o_index <= idx_pipe[5]; o_data <= rq_q;
      end
      if (!busy && start) begin
        busy <= 1'b1; ran <= 1'b0; mv_start <= 1'b1;
      end else if (busy && ran && !mv_busy && outst == 0 && !mv_colv
                   && !rq_vin) begin
        busy <= 1'b0;
      end
    end
  end
endmodule
