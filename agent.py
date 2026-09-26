"""Design agent. Rule-based stand-in for an LLM: propose(spec, feedback_history)
returns Verilog source derived from parsed tool feedback, never from an
iteration counter. To upgrade, replace RuleBasedAgent with a class whose
propose() sends spec plus feedback to an LLM and returns its Verilog.

Two block types are handled, dispatched on spec name: the fabric endpoint
(crc32_endpoint) and the compute chiplet (any MAC spec, including the ones
specgen.py derives from the model, at whatever data and accumulator widths
the derivation chose). Each carries its own seeded first-cut bugs so the demo
shows the feedback loop converging on both."""

import specgen

FIX_WIDTH = "widen_product_register"
FIX_CLEAR = "implement_sync_clear"
FIX_XOR = "apply_final_inversion"
FIX_SATURATE = "saturate_instead_of_wrap"
FIX_LUT = "interpolate_the_fractional_part"
FIX_NORM = "normalise_before_the_table"
FIX_EVEN = "align_the_exponent_to_an_even_boundary"
FIX_CLRCOL = "clear_the_accumulator_between_columns"
FIX_MEMLAT = "delay_valid_for_the_memory_read"
FIX_REGRD = "register_the_read_port"
FIX_SUBMAX = "subtract_the_row_maximum"
FIX_CHAIN = "take_the_second_depth_from_the_first_count"
# One definition, in specgen, so the declared depth and the generated
# depth cannot disagree.
WIDE_ADD_BITS = specgen.WIDE_ADD_BITS
ACC_SLICE_BITS = specgen.ACC_SLICE_BITS


class RuleBasedAgent:
    def propose(self, spec, feedback_history):
        fixes = self.diagnose(spec, feedback_history)
        # Dispatch on the block's top module, not its name: specgen derives
        # names that carry the parameters (mac8_gpt2_124m, crc32_endpoint_16B)
        # and those change with the model and the link rate.
        if spec["top_module"] == "crc32":
            return self.render_crc(spec, fixes), sorted(fixes)
        if spec["top_module"] == "requant":
            return self.render_requant(spec, fixes), sorted(fixes)
        if spec["top_module"] == "expu":
            return self.render_exp(spec, fixes), sorted(fixes)
        if spec["top_module"] == "recip":
            return self.render_recip(spec, fixes), sorted(fixes)
        if spec["top_module"] == "rsqrt":
            return self.render_rsqrt(spec, fixes), sorted(fixes)
        if spec["top_module"] == "matvec":
            return self.render_matvec(spec, fixes), sorted(fixes)
        if spec["top_module"] == "wmem":
            return self.render_wmem(spec, fixes), sorted(fixes)
        if spec["top_module"] == "softmax":
            return self.render_softmax(spec, fixes), sorted(fixes)
        if spec["top_module"] == "mlp":
            return self.render_mlp(spec, fixes), sorted(fixes)
        return self.render_mac(spec, fixes), sorted(fixes)

    def diagnose(self, spec, history):
        """Map parsed feedback to fix intents, mimicking how an LLM would read
        tool output. MAC: clear-test mismatches implicate the clear path,
        other accumulator mismatches implicate datapath width. CRC: a wrong
        checksum on every frame with correct relative behavior implicates the
        standard final inversion."""
        fixes = set()
        for fb in history:
            if fb.get("stage") != "sim":
                continue
            for m in fb.get("mismatches", []):
                if "expected_crc" in m:
                    fixes.add(FIX_XOR)
                elif "clear" in m.get("test", "").lower():
                    fixes.add(FIX_CLEAR)
                elif "expected_y" in m and "out" in m:
                    # The layer's only seeded bug: the second matmul's
                    # reduction length came from an input rather than
                    # from what the first matmul produced.
                    fixes.add(FIX_CHAIN)
                elif "expected_w" in m:
                    # The sequencer's only seeded bug: raw scores fed to
                    # the exponential instead of score minus the row max.
                    fixes.add(FIX_SUBMAX)
                elif spec["top_module"] == "wmem" and "col" in m:
                    # The memory's only seeded bug: a combinational read
                    # delivers data a cycle early and the sequencer
                    # reduces the wrong element.
                    fixes.add(FIX_REGRD)
                elif "col" in m:
                    # Which column failed says which bug it is, which is
                    # the same inference a human makes here. A wrong
                    # column zero means the very first product was
                    # wrong, so the data was not there yet: the valid
                    # was not delayed for the memory read. A correct
                    # column zero with a wrong one after it means the
                    # accumulator was never cleared between them.
                    if m.get("col") == "0":
                        fixes.add(FIX_MEMLAT)
                    else:
                        fixes.add(FIX_CLRCOL)
                elif "expected_e" in m:
                    # The inverse square root's only seeded bug: the
                    # normalisation was aligned to an odd boundary.
                    fixes.add(FIX_EVEN)
                elif "expected_k" in m:
                    # The reciprocal's only seeded bug: the table was
                    # indexed before normalisation.
                    fixes.add(FIX_NORM)
                elif "expected_y" in m:
                    # The exponential's only seeded bug: the fractional
                    # part of the exponent was thrown away.
                    fixes.add(FIX_LUT)
                elif "expected_q" in m:
                    # The requantizer's only seeded bug: it wrapped where
                    # it had to saturate.
                    fixes.add(FIX_SATURATE)
                elif m.get("got_acc") != m.get("expected_acc"):
                    fixes.add(FIX_WIDTH)
        return fixes

    def render_mlp(self, spec, fixes):
        """MLP layer: two matmuls with a requantize and rectify between.

        It owns the activation buffer and drives the matmul sequencer
        and the requantizer, which is the routing that was missing: the
        first matmul's outputs become the second's inputs, in the other
        bank.

        The seeded first-cut bug takes the second matmul's reduction
        length from an input instead of from the first matmul's column
        count. Those are the same number by construction, so the design
        looks right and works whenever a caller happens to pass
        consistent values.
        """
        p = spec["parameters"]
        dw, aw = p["data_width"], p["acc_width"]
        bw, bank = p["bank_width"], p["bank"]
        mw, sw = p["scale_width"], p["shift_width"]
        dep_w, col_w, addr_w = (p["depth_width"], p["col_width"],
                                p["addr_width"])
        d2 = "cols1_r" if FIX_CHAIN in fixes else "depth1"
        return """module mlp (
  input                    clk,
  input                    rst_n,
  input                    load_valid,
  input      signed [{dwm}:0] load_data,
  input                    start,
  input      [{depwm}:0] depth1,
  input      [{colwm}:0] cols1,
  input      [{colwm}:0] cols2,
  input      [{mwm}:0] scale1,
  input      [{swm}:0] shift1,
  input      [{mwm}:0] scale2,
  input      [{swm}:0] shift2,
  output     [{addrwm}:0] w_addr,
  input      signed [{dwm}:0] w_data,
  output reg               o_valid,
  output reg [{bwm}:0] o_index,
  output reg signed [{dwm}:0] o_data,
  output reg               busy
);
  // Two banks of activations. The first matmul reads bank zero and
  // writes bank one; the second reads bank one. That alternation is
  // the routing this block exists to do.
  reg signed [{dwm}:0] act [0:{bank2m}];
  reg [{bwm}:0] wptr;
  reg bank;

  reg [1:0] st;
  localparam S_IDLE = 2'd0, S_M1 = 2'd1, S_M2 = 2'd2, S_DONE = 2'd3;

  reg  mv_start;
  reg  [{depwm}:0] mv_depth;
  reg  [{colwm}:0] mv_cols;
  wire [{depwm}:0] mv_a_addr;
  wire [{addrwm}:0] mv_w_addr;
  wire mv_valid, mv_clear, mv_colv, mv_busy;
  wire [{colwm}:0] mv_coli;
  reg  [{addrwm}:0] w_base;
  reg  [{colwm}:0] cols1_r;

  assign w_addr = w_base + mv_w_addr;

  // The activation the matmul is reading, from whichever bank is the
  // source for the current matmul.
  reg signed [{dwm}:0] a_data;
  always @(posedge clk)
    a_data <= act[{{bank, mv_a_addr[{bwm}:0]}}];

  matvec mv (.clk(clk), .rst_n(rst_n), .start(mv_start),
             .depth(mv_depth), .cols(mv_cols), .a_addr(mv_a_addr),
             .w_addr(mv_w_addr), .mac_valid(mv_valid),
             .mac_clear(mv_clear), .col_valid(mv_colv),
             .col_index(mv_coli), .busy(mv_busy));

  wire signed [{awm}:0] acc;
  wire mac_vout;
  mac mc (.clk(clk), .rst_n(rst_n), .clear(mv_clear), .a(a_data),
          .b(w_data), .valid_in(mv_valid), .acc(acc),
          .valid_out(mac_vout));

  reg  rq_vin;
  reg  signed [{awm}:0] rq_acc;
  reg  [{bwm}:0] rq_idx;
  wire signed [{dwm}:0] rq_q;
  wire rq_sat, rq_vout;
  reg  [{mwm}:0] rq_scale;
  reg  [{swm}:0] rq_shift;
  requant rq (.clk(clk), .rst_n(rst_n), .acc_in(rq_acc),
              .scale(rq_scale), .shift(rq_shift), .valid_in(rq_vin),
              .q_out(rq_q), .sat(rq_sat), .valid_out(rq_vout));

  // The requantizer's index has to travel with its data, so it is
  // pushed through a shift register of the same depth.
  reg [{bwm}:0] idx_pipe [0:{rqsm}];
  integer k;
  // Outstanding requantizations. The requantizer is several stages
  // deep, so "the matmul has finished" is not "the results have
  // landed": advancing on the former writes the first matmul's last
  // outputs after the second has already read that bank.
  reg [7:0] outst;
  // "Not busy" is true before a matmul has started as well as after it
  // has finished, so completion has to mean "it ran and then stopped".
  reg ran;

  always @(posedge clk) begin
    if (!rst_n) begin
      wptr <= 0; bank <= 1'b0; st <= S_IDLE; mv_start <= 1'b0;
      mv_depth <= 0; mv_cols <= 0; w_base <= 0; cols1_r <= 0;
      rq_vin <= 1'b0; rq_acc <= 0; rq_idx <= 0; rq_scale <= 0;
      rq_shift <= 0; o_valid <= 1'b0; o_index <= 0; o_data <= 0;
      busy <= 1'b0; outst <= 0; ran <= 1'b0;
      for (k = 0; k <= {rqsm}; k = k + 1) idx_pipe[k] <= 0;
    end else begin
      mv_start <= 1'b0;
      rq_vin   <= 1'b0;
      o_valid  <= 1'b0;
      idx_pipe[0] <= rq_idx;
      for (k = 1; k <= {rqsm}; k = k + 1)
        idx_pipe[k] <= idx_pipe[k-1];

      if (mv_busy) ran <= 1'b1;

      case ({{mv_colv, rq_vout}})
        2'b10: outst <= outst + 1;
        2'b01: outst <= outst - 1;
        default: ;
      endcase

      if (load_valid && !busy) begin
        act[{{1'b0, wptr}}] <= load_data;
        wptr <= wptr + 1;
      end

      // Every finished column goes straight into the requantizer.
      if (mv_colv) begin
        rq_acc   <= acc;
        rq_idx   <= mv_coli[{bwm}:0];
        rq_vin   <= 1'b1;
        rq_scale <= (st == S_M1) ? scale1 : scale2;
        rq_shift <= (st == S_M1) ? shift1 : shift2;
      end

      case (st)
        S_IDLE: if (start) begin
          st <= S_M1; busy <= 1'b1; bank <= 1'b0; ran <= 1'b0;
          mv_depth <= depth1; mv_cols <= cols1; cols1_r <= cols1;
          w_base <= 0; mv_start <= 1'b1;
        end
        S_M1: begin
          // Rectify here: a negative activation after requantization is
          // clamped, which is what makes this an MLP rather than two
          // bare matmuls.
          if (rq_vout)
            act[{{1'b1, idx_pipe[{rqsm}]}}] <=
                (rq_q[{dwm}] ? {dw}'sd0 : rq_q);
          if (ran && !mv_busy && outst == 0 && !mv_colv) begin
            st <= S_M2; bank <= 1'b1; ran <= 1'b0;
            mv_depth <= {d2};
            mv_cols  <= cols2;
            w_base   <= w_base + (depth1 * cols1);
            mv_start <= 1'b1;
          end
        end
        S_M2: begin
          if (rq_vout) begin
            o_valid <= 1'b1;
            o_index <= idx_pipe[{rqsm}];
            o_data  <= rq_q;
          end
          if (ran && !mv_busy && outst == 0 && !mv_colv) begin
            st <= S_IDLE; busy <= 1'b0; wptr <= 0; ran <= 1'b0;
          end
        end
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
""".format(dwm=dw - 1, dw=dw, awm=aw - 1, bwm=bw - 1, bank2m=2 * bank - 1,
           mwm=mw - 1, swm=sw - 1, depwm=dep_w - 1, colwm=col_w - 1,
           addrwm=addr_w - 1, rqsm=p["requant_stages"] - 1, d2=d2)

    def render_softmax(self, spec, fixes):
        """Softmax sequencer: max pass, exponential pass, one reciprocal,
        normalising multiply.

        This block instantiates the generated exponential and reciprocal
        units rather than reimplementing them, so it is the first
        generated block that contains others. Both are driven through
        their valid handshakes, which is what lets the collect index
        trail the issue index without counting pipeline stages here.

        The seeded first-cut bug is feeding raw scores to the
        exponential instead of the score minus the row maximum. The
        exponential is only defined for non-positive arguments, so a
        positive score produces nonsense, and the bug is invisible on
        any row whose scores are all negative.
        """
        p = spec["parameters"]
        nw, sw = p["index_width"], p["score_width"]
        ww, wf = p["weight_width"], p["weight_frac"]
        rw, riw = p["recip_out_width"], p["recip_in_width"]
        bias = p["shift_bias"]
        kw = max(4, riw.bit_length())
        pw = ww + wf + rw
        sub = "sub_max" if FIX_SUBMAX in fixes else "s_data"
        return """module softmax (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [{nw}:0] n,
  output reg [{nwm}:0] s_addr,
  input      signed [{swm}:0] s_data,
  output reg               w_valid,
  output reg [{nwm}:0] w_index,
  output reg [{wwm}:0] w_data,
  output reg               busy
);
  // Four phases. The exponential is only defined for non-positive
  // arguments, which is why the row maximum is found first and
  // subtracted: that is what makes every argument non-positive.
  localparam P_IDLE = 3'd0, P_MAX = 3'd1, P_EXP = 3'd2,
             P_RCP  = 3'd3, P_OUT = 3'd4;
  reg [2:0] ph;
  reg [{nw}:0] iss, col;
  reg signed [{swm}:0] mx;
  reg [{riwm}:0] sum;
  reg [{rwm}:0] rm;
  reg [{kwm}:0] rk;
  reg [{wwm}:0] buf_mem [0:{capm}];
  reg [{wwm}:0] bq;
  // Both s_addr and the memory read are registered, so the datum for an
  // address assigned at a posedge is valid two cycles later, not one. A
  // single valid flag compares stale data on the first element of every
  // pass, which is the kind of thing that passes a one-element row.
  reg v1, v2;

  wire signed [{swm}:0] sub_max = s_data - mx;
  reg  e_vin;
  wire [{wwm}:0] e_y;
  wire e_vout;
  expu eu (.clk(clk), .rst_n(rst_n), .x({sub}), .valid_in(e_vin),
           .y(e_y), .valid_out(e_vout));

  reg  r_vin;
  wire [{rwm}:0] r_y;
  wire [{kwm}:0] r_k;
  wire r_vout;
  recip ru (.clk(clk), .rst_n(rst_n), .x(sum), .valid_in(r_vin),
            .y(r_y), .k(r_k), .valid_out(r_vout));

  // The multiply, the variable shift and the saturate together miss
  // the clock by a tenth of a nanosecond, so the product is registered
  // between them and the output valid follows it.
  reg  [{pwm}:0] prod_r;
  reg  [{nwm}:0] idx_r;
  reg            ov1;
  wire [{pwm}:0] shifted = prod_r >> ({bias} - rk - {wf});
  wire [{wwm}:0] wsat = (shifted > {one}) ? {ww}'d{one} : shifted[{wwm}:0];

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
      bq      <= buf_mem[s_addr[{nwm}:0]];
      case (ph)
        P_IDLE: if (start && n != 0) begin
          ph <= P_MAX; iss <= 1; col <= 0; s_addr <= 0;
          busy <= 1'b1; v1 <= 1'b1;
          mx <= {sw}'sh{minv};
        end
        P_MAX: begin
          // Reads are registered, so the datum for an address arrives
          // the cycle after it is issued: iss leads col by one.
          if (col != n) begin
            if (iss != n) begin
              s_addr <= iss[{nwm}:0]; iss <= iss + 1; v1 <= 1'b1;
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
            s_addr <= iss[{nwm}:0]; iss <= iss + 1; v1 <= 1'b1;
          end
          // e_vin is registered, so driving it from v1 makes it high
          // in the same cycle v2 is, which is the cycle s_data holds
          // the value. Gating it on v2 asserts it a cycle late and the
          // exponential consumes the next element instead.
          e_vin <= v1;
          if (e_vout) begin
            buf_mem[col[{nwm}:0]] <= e_y;
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
            s_addr <= iss[{nwm}:0]; iss <= iss + 1; v1 <= 1'b1;
          end
          if (v2) begin
            prod_r <= bq * rm;
            idx_r  <= col[{nwm}:0];
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
""".format(nw=nw, nwm=nw - 1, swm=sw - 1, wwm=ww - 1, ww=ww, wf=wf,
           rwm=rw - 1, riwm=riw - 1, kwm=kw - 1, pwm=pw - 1,
           capm=p["capacity"] - 1, bias=bias, sub=sub,
           sw=sw, one=(1 << wf), minv="%x" % (1 << (sw - 1)))

    def render_wmem(self, spec, fixes):
        """Weight tile memory with a streaming loader.

        The write port is sequential, which is how weights actually
        reach a device: a host or a DMA engine pushes them in order and
        the pointer advances. The read port is registered, because that
        is what block RAM gives and what the sequencer was built
        against.

        The seeded first-cut bug is a combinational read. It looks
        harmless and the memory alone behaves, but it delivers data a
        cycle early, so the sequencer multiplies the wrong element. It
        is only visible when the two run together, which is why this
        block's testbench is the whole subsystem.
        """
        p = spec["parameters"]
        dw, cap, aw = p["data_width"], p["capacity"], p["addr_width"]
        if FIX_REGRD in fixes:
            rd = ("  always @(posedge clk)\n"
                  "    rd_data <= mem[rd_addr];")
            decl = "  output reg signed [%d:0] rd_data" % (dw - 1)
        else:   # first cut: combinational read, a cycle too early
            rd = "  always @(*)\n    rd_data = mem[rd_addr];"
            decl = "  output reg signed [%d:0] rd_data" % (dw - 1)
        return """module wmem (
  input                    clk,
  input                    rst_n,
  input                    load_start,
  input                    load_valid,
  input      signed [{dwm}:0] load_data,
  output reg [{cntwm}:0] load_count,
  input      [{awm}:0] rd_addr,
{decl}
);
  reg signed [{dwm}:0] mem [0:{capm}];

  // Sequential write port: a host or a DMA engine pushes weights in
  // order and the pointer advances, saturating at capacity so an
  // overrun corrupts nothing.
  always @(posedge clk) begin
    if (!rst_n) begin
      load_count <= 0;
    end else if (load_start) begin
      load_count <= 0;
    end else if (load_valid && load_count != {cap}) begin
      mem[load_count[{awm}:0]] <= load_data;
      load_count <= load_count + 1;
    end
  end

  // Read port. Registered, one cycle behind rd_addr.
{rd}
endmodule
""".format(dwm=dw - 1, awm=aw - 1, cntwm=aw, capm=cap - 1, cap=cap,
           decl=decl, rd=rd)

    def render_matvec(self, spec, fixes):
        """Weight-streaming sequencer for the MAC chiplet.

        Walks a matrix column by column, drives the MAC for depth cycles,
        waits out that unit's pipeline, and flags the finished column.
        The drain count is the MAC's own latency, taken from the spec, so
        a change to the MAC's pipeline changes this block rather than
        silently desynchronising it.

        The seeded first-cut bug is forgetting to clear the accumulator
        between columns, so every column sums into the one before it.
        Column zero is then correct and every later column is wrong,
        which is the shape of bug that survives a one-column test.
        """
        p = spec["parameters"]
        dep_w, col_w, addr_w = (p["depth_width"], p["col_width"],
                                p["addr_width"])
        stages = p["mac_stages"]
        mlat = p.get("mem_latency", 1)
        clr = ("      if (state == S_EMIT) mac_clear <= 1'b1;"
               if FIX_CLRCOL in fixes else
               "      // first cut: no clear between columns")
        # Synchronous memory: the data for an address lands a cycle
        # later, so valid has to follow it. The first cut drives valid
        # with the address and multiplies whatever the memory held
        # before, which is right only for the very first element by
        # accident.
        vsel = "vdly" if FIX_MEMLAT in fixes else "issue"
        drain_n = stages + mlat
        return """module matvec (
  input                    clk,
  input                    rst_n,
  input                    start,
  input      [{depwm}:0] depth,
  input      [{colwm}:0] cols,
  output reg [{depwm}:0] a_addr,
  output reg [{addrwm}:0] w_addr,
  output                   mac_valid,
  output reg               mac_clear,
  output reg               col_valid,
  output reg [{colwm}:0] col_index,
  output reg               busy
);
  // Memory reads are asynchronous: data is expected in the same cycle as
  // the address, which is what a LUT RAM gives and what keeps the
  // address and the MAC's valid in step without another pipeline stage.
  localparam S_IDLE = 2'd0, S_RUN = 2'd1, S_DRAIN = 2'd2, S_EMIT = 2'd3;
  reg [1:0] state;
  reg [{depwm}:0] row;
  reg [{colwm}:0] col;
  reg [3:0] drain;
  reg issue, vdly;
  // A running base rather than col*depth. The multiply is the obvious
  // way to write it and it put a 12 by 12 multiplier in the control
  // path: the generic library hid that at 134 MHz and real place and
  // route came back at 69.6 against a 100 MHz target. The base only
  // ever advances by depth, so an adder does the same job.
  reg [{addrwm}:0] col_base;
  assign mac_valid = {vsel};

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
            drain <= {drain_n};
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
{clr}
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
""".format(depwm=dep_w - 1, colwm=col_w - 1, addrwm=addr_w - 1,
           drain_n=drain_n, clr=clr, vsel=vsel)

    def render_rsqrt(self, spec, fixes):
        """Inverse square root by even normalisation, table, and a halved
        exponent.

        A square root halves the exponent, so the input has to be
        normalised by an even number of bits or the halved exponent is
        not an integer. That makes the mantissa span two octaves instead
        of one, which is why the table covers [1,4).

        The seeded first-cut bug is aligning to an odd boundary, which
        leaves the result wrong by a factor of root two for half of all
        inputs and exactly right for the other half.
        """
        p = spec["parameters"]
        iw, ow, lb = p["in_width"], p["out_width"], p["lut_bits"]
        ew = max(4, iw.bit_length())
        align = iw - 2 if FIX_EVEN in fixes else iw - 1
        return """module rsqrt (
  input               clk,
  input               rst_n,
  input      [{iwm}:0] x,
  input               valid_in,
  output reg [{owm}:0] y,
  output reg [{ewm}:0] e,
  output reg          valid_out
);
  // Normalise by an even number of bits so the halved exponent is an
  // integer, which makes the mantissa span [1,4) and the table index the
  // top bits of it directly. The consumer applies the shift, so the
  // mantissa stays full width.
  function [{ewm}:0] msb;
    input [{iwm}:0] v;
    integer i;
    begin
      msb = 0;
      for (i = {iwm}; i >= 0; i = i - 1)
        if (v[i] && msb == 0) msb = i[{ewm}:0];
    end
  endfunction

  wire [{ewm}:0] e_w  = msb(x) >> 1;
  wire [{ewm}:0] s_w  = {align} - (e_w << 1);
  wire [{iwm}:0] xn_w = x << s_w;
  reg  [{iwm}:0] xn;
  wire [{owm}:0] lut_out;
  rsqrt_rom rom (.idx(xn[{hi}:{lo}]), .val(lut_out));
  reg  [{ewm}:0] e1, e2;
  reg  [{owm}:0] m;
  reg            vpipe, vpipe2;
  always @(posedge clk) begin
    if (!rst_n) begin
      xn        <= 0;
      e1        <= 0;
      e2        <= 0;
      m         <= 0;
      y         <= 0;
      e         <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      valid_out <= 1'b0;
    end else begin
      xn        <= xn_w;
      e1        <= e_w;
      vpipe     <= valid_in;
      m         <= lut_out;
      e2        <= e1;
      vpipe2    <= vpipe;
      y         <= m;
      e         <= e2;
      valid_out <= vpipe2;
    end
  end
endmodule
""".format(iwm=iw - 1, owm=ow - 1, ewm=ew - 1, ow=ow, align=align,
           hi=iw - 1, lo=iw - lb)

    def render_recip(self, spec, fixes):
        """Reciprocal by normalise, look up, and hand back the shift.

        The denominator is normalised into [1,2) so one table covers the
        whole range, and the mantissa is returned at full width with the
        shift beside it rather than pre-shifted, because the softmax
        denominator spans ten bits and a pre-shifted output would keep as
        few as five significant bits at the top of that range.

        The seeded first-cut bug is indexing the table with the raw input
        instead of the normalised one, which is right only when the input
        already has its high bit set.
        """
        p = spec["parameters"]
        iw, ow, lb = p["in_width"], p["out_width"], p["lut_bits"]
        kw = max(4, iw.bit_length())
        src = "xn" if FIX_NORM in fixes else "x"
        return """module recip (
  input               clk,
  input               rst_n,
  input      [{iwm}:0] x,
  input               valid_in,
  output reg [{owm}:0] y,
  output reg [{kwm}:0] k,
  output reg          valid_out
);
  // Normalise x into [1,2) so a single table covers every input, then
  // hand back that table entry and the normalisation count. The consumer
  // folds the shift into the multiply it was going to do anyway, which
  // keeps the mantissa at full width.
  function [{kwm}:0] lzc;
    input [{iwm}:0] v;
    integer i;
    begin
      lzc = {iw};
      for (i = {iwm}; i >= 0; i = i - 1)
        if (v[i] && lzc == {iw}) lzc = {iwm} - i;
    end
  endfunction

  wire [{kwm}:0] k_w  = lzc(x);
  wire [{iwm}:0] xn_w = x << k_w;
  reg  [{iwm}:0] xn;
  wire [{owm}:0] lut_out;
  recip_rom rom (.idx({src}[{hi}:{lo}]), .val(lut_out));
  reg  [{kwm}:0] k1, k2;
  reg  [{owm}:0] m;
  reg            vpipe, vpipe2;
  always @(posedge clk) begin
    if (!rst_n) begin
      xn        <= 0;
      k1        <= 0;
      k2        <= 0;
      m         <= 0;
      y         <= 0;
      k         <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      valid_out <= 1'b0;
    end else begin
      xn        <= xn_w;
      k1        <= k_w;
      vpipe     <= valid_in;
      m         <= lut_out;
      k2        <= k1;
      vpipe2    <= vpipe;
      y         <= m;
      k         <= k2;
      valid_out <= vpipe2;
    end
  end
endmodule
""".format(iwm=iw - 1, owm=ow - 1, kwm=kw - 1, iw=iw, ow=ow, src=src,
           hi=iw - 2, lo=iw - 1 - lb)

    def render_exp(self, spec, fixes):
        """Fixed-point exponential by table and shift.

        exp(x) = 2**(x*log2(e)). Split x*log2(e) into an integer part and
        a fraction: the fraction indexes a table of 2**f and the integer
        part, which is non-positive because softmax subtracts the row
        maximum first, is a right shift.

        The seeded first-cut bug is dropping the fractional part, so the
        unit returns a plain power of two and every value between them is
        wrong by up to a factor of two.

        A logical shift where the integer part wants an arithmetic one was
        tried as the seeded bug first and turned out to be unobservable
        here: the two differ only in the high bits, and only the low bits
        of the shift amount survive into sh, so they agree modulo its
        width. The same reasoning showed a separate flush-to-zero flag was
        dead logic, because clamping the shift already drives the output
        to zero. Both are gone; the shift is arithmetic because that is
        what it means, not because the testbench can tell.
        """
        p = spec["parameters"]
        iw, fi = p["in_width"], p["in_frac"]
        ow, fo, lb = p["out_width"], p["out_frac"], p["lut_bits"]
        tw = iw + 18
        shw = max(4, (tw).bit_length())
        # The table is a generated module, instantiated rather than
        # transcribed. See specgen.render_rom for why.
        return """module expu (
  input                     clk,
  input                     rst_n,
  input      signed [{iwm}:0] x,
  input                     valid_in,
  output reg        [{owm}:0] y,
  output reg                valid_out
);
  // exp(x) = 2**(x*log2(e)); the fraction of x*log2(e) indexes a table of
  // 2**f and its integer part is a right shift. x is non-positive, which
  // softmax guarantees by subtracting the row maximum, so the shift only
  // ever goes right.
  reg signed [{twm}:0] t;
  reg        [{owm}:0] m;
  reg        [{shm}:0] sh;
  reg                  vpipe, vpipe2;
  wire [{owm}:0] lut_out;
  exp_rom rom (.idx(t[{fim}:0]), .val(lut_out));
  wire signed [{twm}:0] prod = x * $signed({{1'b0, 18'd{log2e}}});
  wire signed [{twm}:0] tt   = prod >>> 16;
  wire signed [{twm}:0] n    = t >>> {fi};
  always @(posedge clk) begin
    if (!rst_n) begin
      t         <= 0;
      m         <= 0;
      sh        <= 0;
      vpipe     <= 1'b0;
      vpipe2    <= 1'b0;
      y         <= 0;
      valid_out <= 1'b0;
    end else begin
      t         <= tt;
      vpipe     <= valid_in;
      m         <= {mexpr};
      // Clamping the shift is what drives an underflowing input to zero:
      // the table entry is {ow} bits, so a shift of {fo1} empties it. A
      // separate flush flag was redundant.
      sh        <= (-n) > {fo} ? {fo1} : (-n);
      vpipe2    <= vpipe;
      y         <= m >> sh;
      valid_out <= vpipe2;
    end
  end
endmodule
""".format(iwm=iw - 1, owm=ow - 1, twm=tw - 1, shm=shw - 1, lbm=lb - 1,
           fim=fi - 1, fi=fi, fo=fo, fo1=fo + 1, ow=ow,
           one=1 << fo, log2e=specgen.LOG2E_Q16,
           mexpr="lut_out" if FIX_LUT in fixes
                 else ("%d'd%d" % (ow, 1 << fo)))

    def render_requant(self, spec, fixes):
        """Requantizer: scale, round, saturate.

        The structure is generated, not written, because every structure
        guessed at was wrong and OpenSTA named a different critical path
        each time: the rounding shifters, then the scale multiply, then the
        adder recombination. The final constraint is blunt. In this generic
        cell library, with no carry chain, a 64-bit add is about 13.4 ns
        against a 10 ns budget, and every wide add in the block violated,
        so no amount of rearranging helps. Wide adds are therefore split
        across two stages with the carry registered between them.

        Latency is the currency being spent, and it is nearly free here: a
        requantization happens once per dot product, so once per
        reduction_depth MACs, which is at least 64 and usually thousands.

        The seeded first-cut bug is wrapping instead of saturating, the
        mistake this block exists to prevent: a wrapped overflow flips the
        sign of a large activation and corrupts every later layer.
        """
        p = spec["parameters"]
        aw, dw = p["acc_width"], p["out_width"]
        mw, sw = p["scale_width"], p["shift_width"]
        n = p.get("scale_splits", 3)
        pw = aw + mw
        wide = pw > WIDE_ADD_BITS
        half_pt = pw // 2
        lo_s, hi_s = -(1 << (dw - 1)), (1 << (dw - 1)) - 1

        decls, reset = [], []
        stages = []          # each entry: list of "dst <= expr" strings

        def reg(name, width=None, init="0"):
            decls.append("  reg  signed [%d:0] %s;" % ((width or pw) - 1, name)
                         if width != 1 else "  reg  %s;" % name)
            reset.append("      %-10s <= %s;" % (name, init))

        def add_stage(pairs):
            stages.append(pairs)

        def emit_add(dst, x, y, into):
            """One add, as one stage when it fits and two when it does not.
            The split keeps the carry in a register between halves."""
            if not wide:
                reg(dst)
                into.append([(dst, "%s + %s" % (x, y))])
                return
            lo, hh, yh = dst + "_lo", dst + "_xh", dst + "_yh"
            decls.append("  reg  [%d:0] %s;" % (half_pt, lo))
            reset.append("      %-10s <= 0;" % lo)
            reg(hh)
            reg(yh)
            reg(dst)
            into.append([(lo, "{1'b0, %s[%d:0]} + {1'b0, %s[%d:0]}"
                          % (x, half_pt - 1, y, half_pt - 1)),
                         (hh, x), (yh, y)])
            into.append([(dst, "{%s[%d:%d] + %s[%d:%d] + %s[%d], %s[%d:0]}"
                          % (hh, pw - 1, half_pt, yh, pw - 1, half_pt,
                             lo, half_pt, lo, half_pt - 1))])

        # Stage 0: partial products, each already weighted so the tree
        # never carries a shift.
        #
        # Both operands are sliced, not just the scale. Slicing only the
        # scale left the critical path at acc_in to a partial product:
        # multiplying a 46-bit accumulator by even a 3-bit slice is two
        # wide adds of shifted copies, so the multiplicand's width is on
        # the path whatever the multiplier does. The high slice of acc_in
        # keeps the sign; the low slice is unsigned.
        an = 2 if aw > ACC_SLICE_BITS else 1
        sb = [(i * mw // n, ((i + 1) * mw // n) - 1) for i in range(n)]
        sb[-1] = (sb[-1][0], mw - 1)
        ab = [(i * aw // an, ((i + 1) * aw // an) - 1) for i in range(an)]
        ab[-1] = (ab[-1][0], aw - 1)
        cur, s0 = [], []
        for ai, (alo, ahi) in enumerate(ab):
            atop = (ai == an - 1)
            aterm = ("$signed(acc_in[%d:%d])" % (ahi, alo) if atop
                     else "$signed({1'b0, acc_in[%d:%d]})" % (ahi, alo))
            for si, (slo, shi) in enumerate(sb):
                nm = "pp%d_%d" % (ai, si)
                cur.append(nm)
                reg(nm)
                e = "%s * $signed({1'b0, scale[%d:%d]})" % (aterm, shi, slo)
                w = alo + slo
                s0.append((nm, "(%s) <<< %d" % (e, w) if w else e))
        add_stage(s0)

        # Pairwise reduction.
        lvl = 0
        per_add = 2 if wide else 1
        while len(cur) > 1:
            lvl += 1
            nxt, groups = [], []
            for i in range(0, len(cur) - 1, 2):
                dst = "s%d_%d" % (lvl, i // 2)
                nxt.append(dst)
                own = []                     # this add's own stage list
                emit_add(dst, cur[i], cur[i + 1], own)
                groups.append(own)
            if len(cur) % 2:
                # An odd term is carried forward, and must be delayed by
                # exactly as many stages as the adds beside it or it
                # arrives at the next level a cycle early.
                dst = "s%d_%d" % (lvl, len(cur) // 2)
                nxt.append(dst)
                own, src = [], cur[-1]
                for k in range(per_add):
                    nm = dst if k == per_add - 1 else "%s_d%d" % (dst, k)
                    reg(nm)
                    own.append([(nm, src)])
                    src = nm
                groups.append(own)
            # Every add in a level takes the same number of stages, so the
            # level's stages are the per-add stages merged position by
            # position.
            for k in range(per_add):
                merged = []
                for grp in groups:
                    if k < len(grp):
                        merged.extend(grp[k])
                add_stage(merged)
            cur = nxt
        rdepth = len(stages) - 1

        # Rounding add, then the output shift, then the saturate.
        built = []
        emit_add("summed", cur[0], "hf_last", built)
        for b in built:
            add_stage(b)
        # The saturate stays in the last stage. Precomputing the two
        # range flags beside the shift was tried, so the last stage would
        # be a small mux: it duplicates the barrel shifter three times and
        # took the real device from 97.88 MHz to 58.78. Measured, reverted.
        reg("shifted")
        add_stage([("shifted", "summed >>> sh_last")])
        nstage = len(stages) + 1           # + the saturate stage

        # Sideband pipelines, each exactly as deep as the point it is used.
        hf_depth = 1 + rdepth              # hf is consumed by the first
        sh_depth = len(stages) - 1         # sh by the shift stage
        for nm, depth, src in (("hf", hf_depth, "half_w"),
                               ("sh", sh_depth, "shift")):
            w = pw if nm == "hf" else sw
            names = ["%s%d" % (nm, i) for i in range(depth)]
            decls.append("  reg  %s[%d:0] %s;"
                         % ("signed " if nm == "hf" else "", w - 1,
                            ", ".join(names)))
            for x in names:
                reset.append("      %-10s <= 0;" % x)
            stages[0].append((names[0], src))
            for i in range(1, depth):
                stages[i].append((names[i], names[i - 1]))
        body = "\n".join(
            "\n".join("      %-10s <= %s;" % (d, e) for d, e in st)
            for st in stages)
        body = body.replace("hf_last", "hf%d" % (hf_depth - 1))
        body = body.replace("sh_last", "sh%d" % (sh_depth - 1))

        vnames = ["v%d" % i for i in range(nstage - 1)]
        decls.append("  reg  %s;" % ", ".join(vnames))
        for x in vnames:
            reset.append("      %-10s <= 1'b0;" % x)
        vlines = ["      %-10s <= valid_in;" % vnames[0]]
        vlines += ["      %-10s <= %s;" % (vnames[i], vnames[i - 1])
                   for i in range(1, len(vnames))]

        if FIX_SATURATE in fixes:
            clamp = """      if (shifted > %d'sd%d) begin
        q_out <= %d'sd%d;
        sat   <= 1'b1;
      end else if (shifted < -%d'sd%d) begin
        q_out <= -%d'sd%d;
        sat   <= 1'b1;
      end else begin
        q_out <= shifted[%d:0];
        sat   <= 1'b0;
      end""" % (pw, hi_s, dw, hi_s, pw, -lo_s, dw, -lo_s, dw - 1)
        else:   # first cut truncates, which wraps on overflow
            clamp = ("      q_out <= shifted[%d:0];\n"
                     "      sat   <= 1'b0;" % (dw - 1))

        return """module requant (
  input                     clk,
  input                     rst_n,
  input      signed [{awm}:0] acc_in,
  input             [{mwm}:0] scale,
  input             [{swm}:0] shift,
  input                     valid_in,
  output reg signed [{dwm}:0] q_out,
  output reg                sat,
  output reg                valid_out
);
  // scale is unsigned, so it is zero-extended before the signed multiply:
  // mixing a signed and an unsigned operand makes the whole expression
  // unsigned in Verilog and silently breaks every negative accumulator.
  // half depends only on shift, so it is built before the tree and never
  // sits on the rounding path.
{decls}
  wire signed [{pwm}:0] half_w = (shift == 0)
        ? {pw}'sd0 : ({pw}'sd1 <<< (shift - 1));
  always @(posedge clk) begin
    if (!rst_n) begin
{reset}
      q_out      <= 0;
      sat        <= 1'b0;
      valid_out  <= 1'b0;
    end else begin
{body}
{vlines}
{clamp}
      valid_out  <= {vlast};
    end
  end
endmodule
""".format(awm=aw - 1, mwm=mw - 1, swm=sw - 1, dwm=dw - 1, pwm=pw - 1,
           pw=pw, decls="\n".join(decls), reset="\n".join(reset),
           body=body, vlines="\n".join(vlines), clamp=clamp,
           vlast=vnames[-1])

    def render_mac(self, spec, fixes):
        p = spec["parameters"]
        dw, aw = p["data_width"], p["acc_width"]
        pw = 2 * dw if FIX_WIDTH in fixes else dw  # first cut truncates the product
        # Quantized weights are two's complement, so the operands, the
        # product and the accumulator are all signed and the product is
        # sign-extended on the way in. Declaring only some of them signed
        # is worse than declaring none: Verilog makes the whole expression
        # unsigned if any operand is, so a half-signed datapath silently
        # computes the unsigned answer.
        sg = "signed " if p.get("signed", True) else ""
        if FIX_CLEAR in fixes:
            acc_logic = """      if (clear) begin
        acc <= {aw}'d0;
        valid_out <= 1'b0;
      end else begin
        if (vpipe) acc <= acc + prod;
        valid_out <= vpipe;
      end""".format(aw=aw)
        else:  # first cut forgets the synchronous clear
            acc_logic = """      if (vpipe) acc <= acc + prod;
      valid_out <= vpipe;"""
        stages = p.get("pipeline_stages", 2)
        if stages >= 3:
            # Split the multiply. b's high half is signed, its low half is
            # not, so the two partial products are formed separately and
            # recombined a cycle later. Each multiplier is half as deep as
            # the full one, which is where the clock comes back.
            h = dw // 2
            decl = ("  reg {sg}[{p}:0] p_lo, p_hi;\n"
                    "  reg {sg}[{p}:0] prod;\n"
                    "  reg         vpipe, vpipe2;").format(sg=sg, p=pw - 1)
            reset = ("      p_lo      <= 0;\n"
                     "      p_hi      <= 0;\n"
                     "      prod      <= 0;\n"
                     "      vpipe     <= 1'b0;\n"
                     "      vpipe2    <= 1'b0;")
            drive = ("      p_lo   <= a * $signed({{1'b0, b[{hm}:0]}});\n"
                     "      p_hi   <= a * $signed(b[{d}:{h}]);\n"
                     "      prod   <= p_lo + (p_hi <<< {h});\n"
                     "      vpipe  <= valid_in;\n"
                     "      vpipe2 <= vpipe;").format(hm=h - 1, d=dw - 1, h=h)
            vq = "vpipe2"
        else:
            decl = ("  reg {sg}[{p}:0] prod;\n"
                    "  reg         vpipe;").format(sg=sg, p=pw - 1)
            reset = ("      prod      <= 0;\n"
                     "      vpipe     <= 1'b0;")
            drive = ("      prod  <= a * b;\n"
                     "      vpipe <= valid_in;")
            vq = "vpipe"
        acc_logic = acc_logic.replace("vpipe)", vq + ")").replace(
            "<= vpipe;", "<= " + vq + ";")
        return """module mac (
  input                     clk,
  input                     rst_n,
  input                     clear,
  input      {sg}[{d}:0] a,
  input      {sg}[{d}:0] b,
  input                     valid_in,
  output reg {sg}[{a}:0] acc,
  output reg                valid_out
);
{decl}
  always @(posedge clk) begin
    if (!rst_n) begin
{reset}
      acc       <= 0;
      valid_out <= 1'b0;
    end else begin
{drive}
{logic}
    end
  end
endmodule
""".format(d=dw - 1, a=aw - 1, logic=acc_logic, sg=sg,
           decl=decl, reset=reset, drive=drive)

    def render_crc_matrix(self, spec, fixes):
        """CRC32 next state as one XOR reduction per output bit.

        The step function is linear over GF(2), so the next state is the XOR
        of a fixed set of current-state and input bits. Written this way the
        combinational depth is logarithmic in the datapath width rather than
        linear in it, which is the difference between an endpoint that meets
        its clock at 32 bytes per cycle and one that misses by 11 ns. The
        selection matrices come from specgen, which derives them from the
        same polynomial the golden vectors use.
        """
        w = spec["parameters"]["bytes_per_cycle"]
        A, B = specgen.crc_matrix(w)
        out_expr = "nxt ^ 32'hFFFFFFFF" if FIX_XOR in fixes else "nxt"
        lines = []
        for i in range(32):
            terms = ["state[%d]" % j for j in range(32) if (A[j] >> i) & 1]
            terms += ["data[%d]" % k for k in range(8 * w) if (B[k] >> i) & 1]
            if not terms:
                lines.append("  assign nxt[%d] = 1'b0;" % i)
                continue
            body, cur = [], "  assign nxt[%d] = ^{" % i
            for t in terms:
                piece = t + ", "
                if len(cur) + len(piece) > 76:
                    body.append(cur)
                    cur = "      "
                cur += piece
            body.append(cur.rstrip(", ") + "};")
            lines.extend(body)
        return """module crc32 (
  input              clk,
  input              rst_n,
  input              clear,
  input      [{dm}:0] data,
  input              valid_in,
  output reg [31:0]  crc_out,
  output reg         valid_out
);
  // zlib/Ethernet CRC32, reflected polynomial 0xEDB88320, {w} byte(s) per
  // cycle, bytes consumed LSB-first. Next state is a GF(2) linear function
  // of the state and the input word, so each bit is a single XOR reduction
  // and the depth is logarithmic in the width rather than linear in it.
  reg  [31:0] state;
  wire [31:0] nxt;
{assigns}

  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= 32'hFFFFFFFF;
      crc_out   <= 32'd0;
      valid_out <= 1'b0;
    end else if (clear) begin
      state     <= 32'hFFFFFFFF;
      valid_out <= 1'b0;
    end else if (valid_in) begin
      state     <= nxt;
      crc_out   <= {out};
      valid_out <= 1'b1;
    end else begin
      valid_out <= 1'b0;
    end
  end
endmodule
""".format(out=out_expr, w=w, dm=8 * w - 1, assigns="\n".join(lines))

    def render_crc(self, spec, fixes):
        if spec["parameters"].get("architecture") == "matrix":
            return self.render_crc_matrix(spec, fixes)
        # Width-generic: the endpoint consumes bytes_per_cycle bytes per
        # clock, derived from the link rate it has to keep up with, so the
        # same agent covers a 1-byte 1GbE endpoint and a 16-byte 25GbE one.
        # First cut forgets the standard final inversion (crc_out = state
        # instead of state ^ 0xFFFFFFFF), a classic CRC32 bring-up bug: every
        # frame checksum is wrong by the same transformation, which the
        # golden-vector testbench catches on the first frame.
        w = spec["parameters"]["bytes_per_cycle"]
        out_expr = "nxt ^ 32'hFFFFFFFF" if FIX_XOR in fixes else "nxt"
        return """module crc32 (
  input              clk,
  input              rst_n,
  input              clear,
  input      [{dm}:0] data,
  input              valid_in,
  output reg [31:0]  crc_out,
  output reg         valid_out
);
  // zlib/Ethernet CRC32, reflected polynomial 0xEDB88320, {w} byte(s) per
  // cycle, bytes consumed LSB-first (little-endian packing on the wire).
  function [31:0] stepw;
    input [31:0] c;
    input [{dm}:0] d;
    integer i, k;
    reg [31:0] x;
    reg [7:0]  b;
    begin
      x = c;
      for (i = 0; i < {w}; i = i + 1) begin
        b = (d >> (8 * i));
        x = x ^ {{24'd0, b}};
        for (k = 0; k < 8; k = k + 1)
          x = (x >> 1) ^ (32'hEDB88320 & {{32{{x[0]}}}});
      end
      stepw = x;
    end
  endfunction

  reg [31:0] state;
  reg [31:0] nxt;
  always @(posedge clk) begin
    if (!rst_n) begin
      state     <= 32'hFFFFFFFF;
      crc_out   <= 32'd0;
      valid_out <= 1'b0;
    end else if (clear) begin
      state     <= 32'hFFFFFFFF;
      valid_out <= 1'b0;
    end else if (valid_in) begin
      nxt = stepw(state, data);
      state     <= nxt;
      crc_out   <= {out};
      valid_out <= 1'b1;
    end else begin
      valid_out <= 1'b0;
    end
  end
endmodule
""".format(out=out_expr, w=w, dm=8 * w - 1)
