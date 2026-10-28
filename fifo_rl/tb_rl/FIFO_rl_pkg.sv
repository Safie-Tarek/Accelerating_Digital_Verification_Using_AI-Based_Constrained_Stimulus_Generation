////////////////////////////////////////////////////////////////////////////////
// AI-team stand-in for the plan-based test (step 1), until the verification team
// delivers its own. It does NOT change anything in "src codes/": it reuses the
// original env, agent, coverage and reset sequence, and only adds
//   FIFO_plan_sequence  reads the plan file (+PLAN_FILE=<path>) and runs its blocks
//   FIFO_rl_test        reset, then the plan; writes result.txt (+RESULT_FILE=<path>)
//
// Plan file (written by fifo_rl/plan/plan_file.py):
//   PLAN id=12 seed=3141592653 version=1
//   ACT idx=0 name=WRITE wr=100 rd=0 rst=0 cycles=10
//   END_TEST
// One transaction = one clock cycle; blocks follow each other with no reset in between.
////////////////////////////////////////////////////////////////////////////////
package FIFO_rl_pkg;
    import uvm_pkg::*;
    import FIFO_seq_item_pkg::*;
    import FIFO_sequence_pkg::*;
    import FIFO_test_pkg::*;
    `include "uvm_macros.svh"

    class FIFO_plan_sequence extends uvm_sequence #(FIFO_seq_item);
        `uvm_object_utils(FIFO_plan_sequence)

        string plan_path;
        int n_cycles = 0;   // transactions sent by this sequence
        int n_blocks = 0;   // ACT lines run

        function new(string name = "FIFO_plan_sequence");
            super.new(name);
        endfunction

        // One block: `cycles` transactions with the block's write / read / reset percentages.
        task run_block(int wr, int rd, int rst, int cycles);
            FIFO_seq_item item;
            repeat (cycles) begin
                item = FIFO_seq_item::type_id::create("item");
                item.WR_EN_ON_DIST = wr;
                item.RD_EN_ON_DIST = rd;
                item.reset_con.constraint_mode(0);   // replace the fixed 10% reset by the block's rst %
                start_item(item);
                if (!item.randomize() with { rst_n dist {0 :/ rst, 1 :/ (100 - rst)}; })
                    `uvm_fatal("PLAN", "randomize failed")
                finish_item(item);
                n_cycles++;
            end
        endtask

        task body();
            int fd, idx, wr, rd, rst, cycles, got;
            string line, name;
            fd = $fopen(plan_path, "r");
            if (fd == 0)
                `uvm_fatal("PLAN", $sformatf("cannot open plan file '%s'", plan_path))
            while ($fgets(line, fd)) begin
                if (line.len() >= 8 && line.substr(0, 7) == "END_TEST")
                    break;
                if (line.len() < 4 || line.substr(0, 3) != "ACT ")
                    continue;   // PLAN header, empty lines
                got = $sscanf(line, "ACT idx=%d name=%s wr=%d rd=%d rst=%d cycles=%d", idx, name, wr, rd, rst, cycles);
                if (got != 6 || idx != n_blocks)
                    `uvm_fatal("PLAN", $sformatf("bad ACT line (expected idx %0d): %s", n_blocks, line))
                `uvm_info("PLAN", $sformatf("block %0d %s wr=%0d rd=%0d rst=%0d cycles=%0d",
                                            idx, name, wr, rd, rst, cycles), UVM_MEDIUM)
                run_block(wr, rd, rst, cycles);
                n_blocks++;
            end
            $fclose(fd);
        endtask
    endclass

    // Reuses FIFO_test's build_phase (env, config object, reset sequence); only the run differs.
    class FIFO_rl_test extends FIFO_test;
        `uvm_component_utils(FIFO_rl_test)

        FIFO_plan_sequence plan_seq;
        string result_path = "";

        function new(string name = "FIFO_rl_test", uvm_component parent = null);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            plan_seq = FIFO_plan_sequence::type_id::create("plan_seq");
            if (!$value$plusargs("PLAN_FILE=%s", plan_seq.plan_path))
                `uvm_fatal("PLAN", "missing +PLAN_FILE=<path>")
            void'($value$plusargs("RESULT_FILE=%s", result_path));
        endfunction

        task run_phase(uvm_phase phase);   // FIFO_test's 4-sequence run is NOT called
            phase.raise_objection(this);
            reset_seq.start(env.agt.sqr);     // the one reset cycle before the first block
            plan_seq.start(env.agt.sqr);
            #1ns;                             // let the last sample reach the coverage, before the next clock
            phase.drop_objection(this);
        endtask

        function void report_phase(uvm_phase phase);
            uvm_report_server server = uvm_report_server::get_server();
            int errors = server.get_severity_count(UVM_ERROR);
            int fatals = server.get_severity_count(UVM_FATAL);
            int fd;
            super.report_phase(phase);
            if (result_path == "")
                return;
            fd = $fopen(result_path, "w");
            if (fd == 0) begin
                `uvm_error("PLAN", $sformatf("cannot write result file '%s'", result_path))
                return;
            end
            $fdisplay(fd, "passed=%0d errors=%0d fatals=%0d cycles=%0d actions=%0d",
                      (errors == 0 && fatals == 0), errors, fatals, 1 + plan_seq.n_cycles, plan_seq.n_blocks);
            $fclose(fd);
        endfunction
    endclass
endpackage
