"""Real-Questa checks of the plan-based test (fifo_rl/tb_rl + QuestaLauncher). Opt-in, ~1 minute:

    FIFO_RL_QUESTA=1 pytest -q tests/test_questa_real.py

Needs vsim/vcover/vlib/vlog on PATH (or FIFO_RL_QUESTA_BIN=<folder>) and the license variable set.
For deterministic plans the fake FIFO and Questa must give the SAME hit count on every one of the 64
bins: that is what makes training on the fake FIFO valid for the plan test.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from fifo_rl import fake_fifo
from fifo_rl.coverage_model import BIN_NAMES
from fifo_rl.launchers.questa_launcher import QuestaLauncher
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID, plan_cycles
from fifo_rl.plan.plan_file import TextPlanWriter
from fifo_rl.plan.stimulus_plan import StimulusPlan
from fifo_rl.questa.report_parser import parse_bin_hits
from fifo_rl.questa.result_parser import ResultParser

pytestmark = pytest.mark.skipif(os.environ.get("FIFO_RL_QUESTA") != "1", reason="set FIFO_RL_QUESTA=1 to run on Questa")

TB_LIST = Path(__file__).resolve().parent.parent / "tb_rl" / "rl_files.list"
PLANS = {
    "fill_hold": ("WRITE", "IDLE"),
    "nine_fixed_blocks": ("WRITE", "IDLE", "READ", "WRITE_READ", "IDLE", "WRITE", "WRITE_READ", "IDLE", "RESET"),
    "empty_test": (),
}


@pytest.fixture(scope="module")
def launcher(tmp_path_factory):
    work = tmp_path_factory.mktemp("questa")
    launcher = QuestaLauncher(work, TB_LIST, top="work.FIFO_rl_top", questa_bin=os.environ.get("FIFO_RL_QUESTA_BIN", ""))
    launcher.compile()
    return launcher


@pytest.mark.parametrize("name", PLANS)
def test_questa_matches_fake_fifo_bin_for_bin(launcher, name):
    plan = StimulusPlan(plan_id=0, seed=1, max_len=10)
    for block in PLANS[name]:
        plan.add(ACTION_INDEX[block], np.zeros(1))
    plan.add(END_TEST_ID, np.zeros(1))
    run_dir = launcher.work_dir / "tests" / name
    plan_file = TextPlanWriter().write(plan, run_dir / "plan.txt")
    out = launcher.launch(plan_file, seed=1, run_dir=run_dir)

    result = ResultParser().parse(out.result_file, out.log_file)
    assert result.passed and result.source == "result"
    assert result.cycles == plan.sim_cycles() and result.n_actions == plan.n_blocks

    questa = parse_bin_hits(out.coverage_report.read_text())
    fake = fake_fifo.bin_hit_counts(fake_fifo.simulate(plan_cycles(plan.action_ids())))
    assert {n: questa[n] for n in BIN_NAMES} == {n: int(c) for n, c in zip(BIN_NAMES, fake)}
