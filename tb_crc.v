`timescale 1ns/1ps
// Self-checking testbench for the fabric endpoint CRC32 block.
// Golden CRC values were precomputed with Python zlib.crc32 on the byte
// streams below (32-bit words packed little-endian, bytes LSB-first), so a
// PASS here means the RTL is bit-compatible with the software link layer.
// Prints machine-parseable TB_FAIL / TB_PROFILE / TB_RESULT lines.
module tb_crc;
  reg clk = 0, rst_n = 0, clear = 0, valid_in = 0;
  reg [31:0] data = 0;
  wire [31:0] crc_out;
  wire valid_out;
  integer checks = 0, i;
  reg [127:0] testname;

  // Throughput profiling state
  integer cyc = 0;
  integer first_vin_cyc = 0, first_vout_cyc = 0, last_vout_cyc = 0;
  integer vout_count = 0;
  reg profiling = 0;

  crc32 dut (.clk(clk), .rst_n(rst_n), .clear(clear), .data(data),
             .valid_in(valid_in), .crc_out(crc_out), .valid_out(valid_out));

  always #5 clk = ~clk;

  always @(posedge clk) begin
    cyc = cyc + 1;
    if (profiling && valid_out) begin
      if (vout_count == 0) first_vout_cyc = cyc;
      last_vout_cyc = cyc;
      vout_count = vout_count + 1;
    end
  end

  task word(input [31:0] w);
    begin
      @(negedge clk);
      valid_in = 1; data = w; clear = 0;
    end
  endtask

  task gap;
    begin
      @(negedge clk);
      valid_in = 0; data = 0; clear = 0;
    end
  endtask

  task frame_clear;
    begin
      @(negedge clk);
      valid_in = 0; data = 0; clear = 1;
      @(negedge clk);
      clear = 0;
    end
  endtask

  // Check the registered CRC one settled cycle after the last word.
  task expect_crc(input [31:0] exp);
    begin
      gap;
      gap;
      checks = checks + 1;
      if (crc_out !== exp) begin
        $display("TB_FAIL test=%0s expected_crc=%0d got_crc=%0d",
                 testname, exp, crc_out);
        $display("TB_RESULT: FAIL");
        $finish;
      end
    end
  endtask

  initial begin
    testname = "reset";
    repeat (3) @(negedge clk);
    rst_n = 1;
    gap;

    // zlib.crc32(b"\x00\x00\x00\x00") = 0x2144DF1C
    testname = "zero_word";
    frame_clear;
    word(32'h00000000);
    expect_crc(32'h2144df1c);

    // zlib.crc32(b"1234") = 0x9BE3E0A3
    testname = "ascii_1234";
    frame_clear;
    word(32'h34333231);
    expect_crc(32'h9be3e0a3);

    // zlib.crc32(b"12345678") = 0x9AE0DAAF, two back-to-back words
    testname = "ascii_12345678";
    frame_clear;
    word(32'h34333231);
    word(32'h38373635);
    expect_crc(32'h9ae0daaf);

    // 32 pseudorandom bytes (seed 42), zlib.crc32 = 0x8316515F
    testname = "random_32B";
    frame_clear;
    word(32'h7d8c0c39); word(32'h2c344772); word(32'h2f0f10d8);
    word(32'h650d776f); word(32'h8ee570d6); word(32'haed85103);
    word(32'hac6e4f8e); word(32'h31c22f34);
    expect_crc(32'h8316515f);

    // Same frame again after a clear: state must fully reinitialize.
    testname = "clear_reinit";
    frame_clear;
    word(32'h34333231);
    expect_crc(32'h9be3e0a3);

    // Throughput burst: 256 back-to-back words (1 KB), one word per cycle.
    testname = "throughput_burst";
    frame_clear;
    profiling = 1;
    vout_count = 0;
    first_vin_cyc = cyc;
    for (i = 0; i < 256; i = i + 1)
      word(i * 32'h01010101 + 32'h00010203);
    gap; gap; gap;
    profiling = 0;
    $display("TB_PROFILE bytes=%0d span_cycles=%0d latency_cycles=%0d",
             vout_count * 4, last_vout_cyc - first_vout_cyc + 1,
             first_vout_cyc - first_vin_cyc);

    $display("TB_PASS checks=%0d", checks);
    $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
