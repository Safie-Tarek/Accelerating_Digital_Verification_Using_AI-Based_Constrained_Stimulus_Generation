////////////////////////////////////////////////////////////////////////////////
// AI-team stand-in for the plan-based test (step 1), until the verification team
// delivers its own. It does NOT change anything in "src codes/": it reuses the
// original env, agent, coverage and reset sequence, and only adds
//   FIFO_plan_sequence  reads the plan file (+PLAN_FILE=<path>) and runs its blocks
//   FIFO_rl_test        reset, then the plan; writes result.txt (+RESULT_FILE=<path>)
//   FIFO_action_coverage (step 2, only with +ACTION_COV=1) one cg_action instance per
//                       block, act_<idx>, with the same 64 bins as cg_FIFO
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

    // Step 2: the same coverpoints, crosses and ignore_bins as cg_FIFO (src codes/FIFO_coverage.sv),
    // so every act_<idx> instance has the same 64 bins. One instance per plan block.
    covergroup cg_action (string inst_name) with function sample(bit wr_en, bit rd_en, bit wr_ack, bit overflow,
            bit full, bit empty, bit almostfull, bit almostempty, bit underflow);
        option.per_instance = 1;
        option.name = inst_name;
        cp_wr_en: coverpoint wr_en;
        cp_rd_en: coverpoint rd_en;
        cp_wr_ack: coverpoint wr_ack;
        cp_overflow: coverpoint overflow;
        cp_full: coverpoint full;
        cp_empty: coverpoint empty;
        cp_almostfull: coverpoint almostfull;
        cp_almostempty: coverpoint almostempty;
        cp_underflow: coverpoint underflow;
        cx_wr_ack: cross cp_wr_en, cp_rd_en, cp_wr_ack {
            ignore_bins auto_wr_ack_0 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {1} && binsof(cp_wr_ack) intersect {1};
            ignore_bins auto_wr_ack_1 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {0} && binsof(cp_wr_ack) intersect {1};
            ignore_bins auto_wr_ack_2 = binsof(cp_wr_en) intersect {1} && binsof(cp_rd_en) intersect {1} && binsof(cp_wr_ack) intersect {0};
        }
        cx_overflow: cross cp_wr_en, cp_rd_en, cp_overflow {
            ignore_bins auto_overflow_0 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {1} && binsof(cp_overflow) intersect {1};
            ignore_bins auto_overflow_1 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {0} && binsof(cp_overflow) intersect {1};
            ignore_bins auto_overflow_2 = binsof(cp_wr_en) intersect {1} && binsof(cp_rd_en) intersect {1} && binsof(cp_overflow) intersect {1};
        }
        cx_full: cross cp_wr_en, cp_rd_en, cp_full {
            ignore_bins auto_full_0 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {1} && binsof(cp_full) intersect {1};
            ignore_bins auto_full_1 = binsof(cp_wr_en) intersect {1} && binsof(cp_rd_en) intersect {1} && binsof(cp_full) intersect {1};
        }
        cx_empty: cross cp_wr_en, cp_rd_en, cp_empty;
        cx_almostfull: cross cp_wr_en, cp_rd_en, cp_almostfull;
        cx_almostempty: cross cp_wr_en, cp_rd_en, cp_almostempty;
        cx_underflow: cross cp_wr_en, cp_rd_en, cp_underflow {
            ignore_bins auto_underflow_0 = binsof(cp_wr_en) intersect {1} && binsof(cp_rd_en) intersect {0} && binsof(cp_underflow) intersect {1};
            ignore_bins auto_underflow_1 = binsof(cp_wr_en) intersect {0} && binsof(cp_rd_en) intersect {0} && binsof(cp_underflow) intersect {1};
        }
    endgroup

    // Step 2: gets every monitor sample (like FIFO_coverage) and samples the act_<idx> instance of
    // the block that sample belongs to. One transaction = one cycle, so sample n belongs to the
    // block whose cycles contain n: sample 0 is the test's reset cycle (no block), then block 0's
    // cycles, block 1's, ... The plan sequence calls add_block() for each ACT line before
    // sending that block's items, so the boundaries are always known in time.
    class FIFO_action_coverage extends uvm_subscriber #(FIFO_seq_item);
        `uvm_component_utils(FIFO_action_coverage)

        localparam int MAX_BLOCKS = 64;      // Questa has no queue of covergroups; plans are <= plan_max_len
        cg_action insts[MAX_BLOCKS];   // act_0, act_1, ...
        int n_blocks = 0;
        int block_end[$];     // sample number just after each block
        int n_samples = 0;
        int unmapped = 0;     // samples after the last block (must stay 0)

        function new(string name = "FIFO_action_coverage", uvm_component parent = null);
            super.new(name, parent);
        endfunction

        function void add_block(int cycles);
            int start = block_end.size() ? block_end[$] : 1;   // after the reset cycle
            if (n_blocks == MAX_BLOCKS)
                `uvm_fatal("PLAN", $sformatf("more than %0d blocks: raise MAX_BLOCKS", MAX_BLOCKS))
            insts[n_blocks] = new($sformatf("act_%0d", n_blocks));
            n_blocks++;
            block_end.push_back(start + cycles);
        endfunction

        function void write(FIFO_seq_item t);
            int k;
            if (n_samples > 0) begin
                for (k = 0; k < block_end.size() && n_samples >= block_end[k]; k++) ;
                if (k < n_blocks)
                    insts[k].sample(t.wr_en, t.rd_en, t.wr_ack, t.overflow, t.full, t.empty,
                                    t.almostfull, t.almostempty, t.underflow);
                else
                    unmapped++;
            end
            n_samples++;
        endfunction
    endclass

    class FIFO_plan_sequence extends uvm_sequence #(FIFO_seq_item);
        `uvm_object_utils(FIFO_plan_sequence)

        string plan_path;
        FIFO_action_coverage act_cov;   // step 2 (+ACTION_COV=1), else null
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
                if (act_cov != null)
                    act_cov.add_block(cycles);
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
        FIFO_action_coverage act_cov;   // only with +ACTION_COV=1
        string result_path = "";

        function new(string name = "FIFO_rl_test", uvm_component parent = null);
            super.new(name, parent);
        endfunction

        function void build_phase(uvm_phase phase);
            int action_cov = 0;
            super.build_phase(phase);
            plan_seq = FIFO_plan_sequence::type_id::create("plan_seq");
            if (!$value$plusargs("PLAN_FILE=%s", plan_seq.plan_path))
                `uvm_fatal("PLAN", "missing +PLAN_FILE=<path>")
            void'($value$plusargs("RESULT_FILE=%s", result_path));
            void'($value$plusargs("ACTION_COV=%d", action_cov));
            if (action_cov) begin
                act_cov = FIFO_action_coverage::type_id::create("act_cov", this);
                plan_seq.act_cov = act_cov;
            end
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            if (act_cov != null)
                env.agt.agt_ap.connect(act_cov.analysis_export);   // same samples as FIFO_coverage
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
            if (act_cov != null && act_cov.unmapped != 0)
                `uvm_error("PLAN", $sformatf("%0d samples after the last block: act_<idx> coverage is misaligned",
                                             act_cov.unmapped))
            errors = server.get_severity_count(UVM_ERROR);
            $fdisplay(fd, "passed=%0d errors=%0d fatals=%0d cycles=%0d actions=%0d",
                      (errors == 0 && fatals == 0), errors, fatals, 1 + plan_seq.n_cycles, plan_seq.n_blocks);
            $fclose(fd);
        endfunction
    endclass
endpackage
