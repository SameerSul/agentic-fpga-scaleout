module softmax (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [5:0] n,
  output reg [4:0] s_addr,
  input      signed [20:0] s_data,
  output reg               w_valid,
  output reg [4:0] w_index,
  output reg [15:0] w_data,
  output reg               busy
);
  // Four phases. The exponential is only defined for non-positive
  // arguments, which is why the row maximum is found first and
  // subtracted: that is what makes every argument non-positive.
  localparam P_IDLE = 3'd0, P_MAX = 3'd1, P_EXP = 3'd2,
             P_RCP  = 3'd3, P_OUT = 3'd4;
  reg [2:0] ph;
  reg [5:0] iss, col;
  reg signed [20:0] mx;
  reg [19:0] sum;
  reg [16:0] rm;
  reg [4:0] rk;
  reg [15:0] buf_mem [0:31];
  reg [15:0] bq;
  // Both s_addr and the memory read are registered, so the datum for an
  // address assigned at a posedge is valid two cycles later, not one. A
  // single valid flag compares stale data on the first element of every
  // pass, which is the kind of thing that passes a one-element row.
  reg v1, v2;

  // score - max is never positive, but it can be far below the
  // exponential's range. Anything below -2**12 has an exponential of
  // zero in the output format anyway, so clamping there is exact.
  wire signed [21:0] dfull = s_data - mx;
  wire signed [12:0] sub_max = (dfull < -22'sd4096) ? -13'sd4096
                                  : dfull[12:0];
  // Registered before the exponential: the wide subtract, the clamp and
  // the exponential's first stage missed 100 MHz by 0.42 ns together.
  reg  signed [12:0] sub_r;
  always @(posedge clk) sub_r <= sub_max;
  reg  e_vin;
  wire [15:0] e_y;
  wire e_vout;
  expu eu (.clk(clk), .rst_n(rst_n), .x(sub_r), .valid_in(e_vin),
           .y(e_y), .valid_out(e_vout));

  reg  r_vin;
  wire [16:0] r_y;
  wire [4:0] r_k;
  wire r_vout;
  recip ru (.clk(clk), .rst_n(rst_n), .x(sum), .valid_in(r_vin),
            .y(r_y), .k(r_k), .valid_out(r_vout));

  // The multiply, the variable shift and the saturate together miss
  // the clock by a tenth of a nanosecond, so the product is registered
  // between them and the output valid follows it.
  reg  [47:0] prod_r;
  reg  [4:0] idx_r;
  reg            ov1;
  wire [47:0] shifted = prod_r >> (36 - rk - 15);
  wire [15:0] wsat = (shifted > 32768) ? 16'd32768 : shifted[15:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      ph <= P_IDLE; iss <= 0; col <= 0; mx <= 0; sum <= 0;
      rm <= 0; rk <= 0; s_addr <= 0; e_vin <= 1'b0; r_vin <= 1'b0;
      w_valid <= 1'b0; w_index <= 0; w_data <= 0; busy <= 1'b0;
      v1 <= 1'b0; v2 <= 1'b0; bq <= 0;
      prod_r <= 0; idx_r <= 0; ov1 <= 1'b0;
    end else begin
      e_vin   <= 1'b0;
      r_vin   <= 1'b0;
      w_valid <= 1'b0;
      v1      <= 1'b0;
      v2      <= v1;
      ov1     <= 1'b0;
      bq      <= buf_mem[s_addr[4:0]];
      case (ph)
        P_IDLE: if (start && n != 0) begin
          ph <= P_MAX; iss <= 1; col <= 0; s_addr <= 0;
          busy <= 1'b1; v1 <= 1'b1;
          mx <= 21'sh100000;
        end
        P_MAX: begin
          // Reads are registered, so the datum for an address arrives
          // the cycle after it is issued: iss leads col by one.
          if (col != n) begin
            if (iss != n) begin
              s_addr <= iss[4:0]; iss <= iss + 1; v1 <= 1'b1;
            end
            if (v2) begin
              if (s_data > mx) mx <= s_data;
              col <= col + 1;
            end
          end else begin
            // Address zero is issued by this transition, so the next
            // one to issue is one. Resetting iss to zero here reads
            // element zero twice and sums its exponential twice.
            ph <= P_EXP; iss <= 1; col <= 0; s_addr <= 0; v1 <= 1'b1;
            sum <= 0;
          end
        end
        P_EXP: begin
          if (iss != n) begin
            s_addr <= iss[4:0]; iss <= iss + 1; v1 <= 1'b1;
          end
          // s_data holds an element in the cycle v2 is high, and sub_r
          // holds its clamped difference one cycle later. e_vin is
          // registered from v2, so it is high in exactly that cycle.
          // Driving it from v1, as before sub_r existed, feeds the
          // exponential the previous element's difference.
          e_vin <= v2;
          if (e_vout) begin
            buf_mem[col[4:0]] <= e_y;
            sum <= sum + e_y;
            col <= col + 1;
          end
          if (col == n) begin
            ph <= P_RCP; r_vin <= 1'b1;
          end
        end
        P_RCP: if (r_vout) begin
          rm <= r_y; rk <= r_k;
          ph <= P_OUT; iss <= 1; col <= 0; s_addr <= 0; v1 <= 1'b1;
        end
        P_OUT: begin
          if (iss != n) begin
            s_addr <= iss[4:0]; iss <= iss + 1; v1 <= 1'b1;
          end
          if (v2) begin
            prod_r <= bq * rm;
            idx_r  <= col[4:0];
            ov1    <= 1'b1;
            col    <= col + 1;
          end
          if (ov1) begin
            w_valid <= 1'b1;
            w_index <= idx_r;
            w_data  <= wsat;
          end
          if (col == n && !ov1 && !v2) begin
            ph <= P_IDLE; busy <= 1'b0;
          end
        end
      endcase
    end
  end
endmodule
