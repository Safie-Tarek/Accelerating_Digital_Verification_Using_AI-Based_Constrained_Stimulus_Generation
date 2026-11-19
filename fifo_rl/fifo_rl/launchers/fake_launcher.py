"""Launcher on the Python fake FIFO: reads the plan file like the UVM test would, simulates it and
writes result.txt + a coverage report in Questa's format (a few ms per test, used for training).

It reads only what the UVM test reads (the four numbers of every ACT line), never the agent's
action ids, so it also checks that the plan file carries everything the testbench needs.

Step 2 (per_action=True): the report also gets one covergroup instance per block, the way the
step-2 testbench will print cg_action with option.per_instance = 1:
    INSTANCE /FIFO_coverage_pkg/FIFO_coverage/cg_action/act_<idx>
with the same 64 bin lines, counting only that block's cycles (the test's first reset cycle
belongs to no block). The cg_FIFO section is unchanged, so step 1 reads the report as before.
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
ACTION_COVERGROUP = "/FIFO_coverage_pkg/FIFO_coverage/cg_action"


def _bin_lines(counts):
    """Input: 64 hit counts in BIN_NAMES order. Output: the Coverpoint/Cross and bin lines."""
    lines, item = [], None
    for name, count in zip(BIN_NAMES, counts):
        bin_item, label = name.split("/", 1)
        if bin_item != item:
            item = bin_item
            lines.append(f"    {'Coverpoint' if item.startswith('cp_') else 'Cross'} {item}")
        lines.append(f"        bin {label:<40} {int(count):>8}          1    {'Covered' if count else 'ZERO'}")
    return lines


def format_report(counts, action_counts=None):
    """A coverage report with the same item and bin lines as vcover's -cvg -details report.

    Input:  counts, hit count of each of the 64 bins in BIN_NAMES order;
            action_counts, None (step 1) or one such list per block, in idx order (step 2).
    Output: the report text (questa.report_parser.hits_vector reads it like a real report;
            hits_per_action reads the act_<idx> instances).
    """
    lines = ["Coverage report written by fifo_rl FakeLauncher (vcover -cvg -details -nocompactcrossbins format)",
             f" TYPE {COVERGROUP}"] + _bin_lines(counts)
    if action_counts is not None:
        lines.append(f" TYPE {ACTION_COVERGROUP}")
        for idx, block_counts in enumerate(action_counts):
            lines.append(f" INSTANCE {ACTION_COVERGROUP}/act_{idx}")
            lines += _bin_lines(block_counts)
    return "\n".join(lines) + "\n"


class FakeLauncher(SimulatorLauncher):
    name = "fake"

    def __init__(self, per_action=False):
        """Input: per_action, also write one act_<idx> instance per block (step 2). Output: nothing."""
        self.per_action = per_action

    def launch(self, plan_file, seed, run_dir):
        """Input:  plan_file, seed, run_dir (see SimulatorLauncher).
        Output: RunOutput pointing at run_dir/result.txt and run_dir/coverage.txt (no log).
        Logic:  1. read the plan file
                2. cycles = the test's reset cycle + every block drawn from its percentages with
                   one generator seeded by `seed` (same seed -> same test, like -sv_seed)
                3. simulate on the fake FIFO and count the hits of every bin
                   (per_action: also per block, over that block's own cycles)
                4. write the report and result.txt (always passed: the fake FIFO has no bugs)
        """
        start = time.perf_counter()
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        plan = read_plan(plan_file)
        rng = np.random.default_rng(seed)
        cycles, bounds = [RESET_CYCLE], []
        for block in plan.blocks:
            first = len(cycles)
            cycles += block_cycles(block.wr_pct, block.rd_pct, block.rst_pct, block.n_cycles, rng)
            bounds.append((first, len(cycles)))
        samples = fake_fifo.simulate(cycles)
        counts = fake_fifo.bin_hit_counts(samples)
        action_counts = [fake_fifo.bin_hit_counts(samples[a:b]) for a, b in bounds] if self.per_action else None

        report = run_dir / "coverage.txt"
        report.write_text(format_report(counts, action_counts))
        result = run_dir / "result.txt"
        result.write_text(f"passed=1 errors=0 assert_fail=0 cycles={len(cycles)} actions={len(plan.blocks)}\n")
        return RunOutput(run_dir, Path(plan_file), report, result, None,
                         sim_time=time.perf_counter() - start, report_time=0.0)
