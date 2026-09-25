read_liberty cells.lib
read_verilog netlist.v
link_design crc32
create_clock -name clk -period 12.8 [get_ports clk]
set_input_delay 0.5 -clock clk [get_ports {rst_n clear data valid_in}]
set_output_delay 0.5 -clock clk [all_outputs]
report_checks -path_delay max
report_worst_slack -max
exit
