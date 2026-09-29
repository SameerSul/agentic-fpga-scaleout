module projn (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [11:0] depth,
  input      [11:0] cols,
  input      [17:0] scale,
  input      [6:0] shift,
  output     [11:0] a_addr,
  input      signed [15:0] a_data,
  output     [18:0] w_addr,
  input      [511:0] w_data,
  output reg [11:0] c_addr,
  input      [60:0] c_data,
  output reg               o_valid,
  output reg [11:0] o_index,
  output reg signed [15:0] o_data,
  output reg               busy
);
  reg issuing;
  reg [11:0] r;
  reg [11:0] g0;          // first column of the group being issued
  reg [18:0] wbase;        // g * depth
  assign a_addr = r;
  assign w_addr = wbase + r;

  // The valid for each row trails its address by the read, and the last
  // row's flag trails the MACs by their stages, when every lane's sum is
  // complete.
  reg v1;
  reg [4:0] lastp;
  reg [11:0] gp [0:4];
  reg mclr;
  wire signed [35:0] acc [0:31];
  genvar j;
  generate
    for (j = 0; j < 32; j = j + 1) begin : lane
      mac mc (.clk(clk), .rst_n(rst_n), .clear(mclr), .a(a_data),
              .b(w_data[16*j +: 16]), .valid_in(v1), .acc(acc[j]), .valid_out());
    end
  endgenerate

  // Shadow bank and the drain through one requantizer.
  reg signed [35:0] shadow [0:31];
  reg [11:0] cap0;
  reg [5:0] dj, dn;
  reg  rq_vin;
  reg  signed [35:0] rq_acc;
  reg  [11:0] rq_idx;
  wire signed [15:0] rq_q;
  wire rq_sat, rq_vout;
  reg [4:0] dsel, dselb;
  reg dva, dvb;
  reg [11:0] didx, didxb;
  reg [17:0] rq_scale_r;
  reg [6:0] rq_shift_r;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc), .scale(rq_scale_r),
              .shift(rq_shift_r), .valid_in(rq_vin), .q_out(rq_q),
              .sat(rq_sat), .valid_out(rq_vout));
  reg [11:0] idx_pipe [0:10];
  reg [7:0] outst;
  reg pend;                    // a finished group waits for the drain
  integer k;
  wire [11:0] left = cols - cap0;
  wire [11:0] next0 = g0 + 32;

  always @(posedge clk) begin
    if (!rst_n) begin
      issuing <= 1'b0; r <= 0; g0 <= 0; wbase <= 0; v1 <= 1'b0;
      lastp <= 0; mclr <= 1'b0; cap0 <= 0; dj <= 0; dn <= 0;
      rq_vin <= 1'b0; rq_acc <= 0; rq_idx <= 0; outst <= 0; pend <= 1'b0;
      busy <= 1'b0; o_valid <= 1'b0; o_index <= 0; o_data <= 0;
      c_addr <= 0; dsel <= 0; dva <= 1'b0; didx <= 0;
      dselb <= 0; dvb <= 1'b0; didxb <= 0;
      rq_scale_r <= 0; rq_shift_r <= 0;
      for (k = 0; k <= 10; k = k + 1) idx_pipe[k] <= 0;
      for (k = 0; k <= 4; k = k + 1) gp[k] <= 0;
      for (k = 0; k < 32; k = k + 1) shadow[k] <= 0;
    end else begin
      mclr <= 1'b0;
      rq_vin <= 1'b0;
      o_valid <= 1'b0;
      v1 <= issuing;
      lastp <= {lastp[3:0], issuing && r == depth - 1};
      gp[0] <= g0;
      for (k = 1; k <= 4; k = k + 1) gp[k] <= gp[k-1];
      idx_pipe[0] <= rq_idx;
      for (k = 1; k <= 10; k = k + 1) idx_pipe[k] <= idx_pipe[k-1];
      case ({rq_vin, rq_vout})
        2'b10: outst <= outst + 1;
        2'b01: outst <= outst - 1;
        default: ;
      endcase
      if (rq_vout) begin
        o_valid <= 1'b1; o_index <= idx_pipe[10]; o_data <= rq_q;
      end

      // Issue rows for the current group; at its last row, stop and wait
      // for the sums.
      if (issuing) begin
        if (r == depth - 1) begin
          issuing <= 1'b0; r <= 0;
        end else begin
          r <= r + 1;
        end
      end

      // A group's sums are complete: take them once the drain is free,
      // clear the lanes, and start the next group in the same cycle.
      if (lastp[4] || pend) begin
        // The shadow bank is only free once its last column has left the
        // drain, parameter read included.
        if (dj == dn && !dva && !dvb) begin
          pend <= 1'b0;
          for (k = 0; k < 32; k = k + 1) shadow[k] <= acc[k];
          cap0 <= gp[4];
          dj <= 0;
          dn <= (cols - gp[4] < 32) ? cols - gp[4] : 32;
          mclr <= 1'b1;
          if (next0 < cols) begin
            g0 <= next0; wbase <= wbase + depth; issuing <= 1'b1;
          end
        end else begin
          pend <= 1'b1;
        end
      end

      if (dj != dn) begin
        c_addr <= cap0 + dj; dsel <= dj[4:0]; dva <= 1'b1;
        didx <= cap0 + dj; dj <= dj + 1;
      end else dva <= 1'b0;
      dvb <= dva; dselb <= dsel; didxb <= didx;
      if (dvb) begin
        rq_acc <= shadow[dselb] + $signed(c_data[60:25]);
        rq_shift_r <= c_data[24:18]; rq_scale_r <= c_data[17:0];
        rq_idx <= didxb; rq_vin <= 1'b1;
      end

      if (!busy && start) begin
        busy <= 1'b1; issuing <= 1'b1; r <= 0; g0 <= 0; wbase <= 0;
        dj <= 0; dn <= 0; pend <= 1'b0;
      end else if (busy && !issuing && !v1 && lastp == 0 && !pend
                   && dj == dn && outst == 0 && !rq_vin && !mclr && !dva && !dvb
                   && g0 + 32 >= cols) begin
        busy <= 1'b0;
      end
    end
  end
endmodule
