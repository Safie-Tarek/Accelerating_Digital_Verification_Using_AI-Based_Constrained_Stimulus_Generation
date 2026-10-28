// Copy of "src codes/FIFO_top.sv" that also imports FIFO_rl_pkg, so FIFO_rl_test is elaborated and
// registered with the UVM factory. Select the test with +UVM_TESTNAME=FIFO_rl_test.
import uvm_pkg::*;
import FIFO_test_pkg::*;
import FIFO_rl_pkg::*;
`include "uvm_macros.svh"

module FIFO_rl_top();
    bit clk;
    always #5 clk = ~clk;
    FIFO_if FIFOif (clk);
    FIFO dut (FIFOif);
    bind FIFO FIFO_SVA assert_inst (FIFOif);

    initial begin
        uvm_config_db #(virtual FIFO_if)::set(null, "uvm_test_top", "FIFO_IF", FIFOif);
        run_test("FIFO_rl_test");
    end
endmodule
