"""Launcher on the Python fake FIFO: reads the plan file like the UVM test would, simulates it and
writes result.txt + a coverage report in Questa's format (a few ms per test, used for training).

It reads only what the UVM test reads (the four numbers of every ACT line), never the agent's
action ids, so it also checks that the plan file carries everything the testbench needs.
"""

import time
from pathlib import Path

import numpy as np

from fifo_rl import fake_fifo
from fifo_rl.coverage_model import BIN_NAMES
from fifo_rl.launchers.base import RunOutput, SimulatorLauncher
from fifo_rl.plan.actions import RESET_CYCLE, block_cycles
from fifo_rl.plan.plan_file import read_plan

COVERGROUP = "/FIFO_coverage_pkg/FIFO_coverage/cg_FIFO"


def format_report(counts):
    """A coverage report with the same item and bin lines as vcover's -cvg -details report.

    Input:  counts, hit count of each of the 64 bins in BIN_NAMES order.
    Output: the report text (questa.report_parser.hits_vector reads it like a real report).
    """
    lines = ["Coverage report written by fifo_rl FakeLauncher (vcover -cvg -details -nocompactcrossbins format)",
             f" TYPE {COVERGROUP}"]
    item = None
    for name, count in zip(BIN_NAMES, counts):
        bin_item, label = name.split("/", 1)
        if bin_item != item:
            item = bin_item
            lines.append(f"    {'Coverpoint' if item.startswith('cp_') else 'Cross'} {item}")
        lines.append(f"        bin {label:<40} {int(count):>8}          1    {'Covered' if count else 'ZERO'}")
    return "\n".join(lines) + "\n"


class FakeLauncher(SimulatorLauncher):
    name = "fake"

    def launch(self, plan_file, seed, run_dir):
        """Input:  plan_file, seed, run_dir (see SimulatorLauncher).
        Output: RunOutput pointing at run_dir/result.txt and run_dir/coverage.txt (no log).
        Logic:  1. read the plan file
                2. cycles = the test's reset cycle + every block drawn from its percentages with
                   one generator seeded by `seed` (same seed -> same test, like -sv_seed)
                3. simulate on the fake FIFO and count the hits of every bin
                4. write the report and result.txt (always passed: the fake FIFO has no bugs)
        """
        start = time.perf_counter()
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        plan = read_plan(plan_file)
        rng = np.random.default_rng(seed)
        cycles = [RESET_CYCLE]
        for block in plan.blocks:
            cycles += block_cycles(block.wr_pct, block.rd_pct, block.rst_pct, block.n_cycles, rng)
        counts = fake_fifo.bin_hit_counts(fake_fifo.simulate(cycles))

        report = run_dir / "coverage.txt"
        report.write_text(format_report(counts))
        result = run_dir / "result.txt"
        result.write_text(f"passed=1 errors=0 assert_fail=0 cycles={len(cycles)} actions={len(plan.blocks)}\n")
        return RunOutput(run_dir, Path(plan_file), report, result, None,
                         sim_time=time.perf_counter() - start, report_time=0.0)
