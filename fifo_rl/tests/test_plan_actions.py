"""Tests for fifo_rl/plan/actions.py: the 9 blocks and their cycles."""

import numpy as np
import pytest

from fifo_rl import fake_fifo
from fifo_rl.coverage_model import BIN_INDEX
from fifo_rl.plan.actions import (ACTION_INDEX, ACTIONS, END_TEST_ID, N_ACTIONS, RESET_CYCLE, block_cycles,
                                  get_action, plan_cycles, to_cycles)


def ids(*names):
    return [ACTION_INDEX[n] for n in names]


def test_nine_actions_with_ids_in_order():
    assert N_ACTIONS == 9
    assert [spec.id for spec in ACTIONS] == list(range(9))
    assert ACTIONS[END_TEST_ID].name == "END_TEST"
    assert ACTIONS[END_TEST_ID].is_end_test
    assert sum(spec.is_end_test for spec in ACTIONS) == 1


def test_end_test_runs_no_cycles():
    assert ACTIONS[END_TEST_ID].n_cycles == 0
    assert to_cycles(END_TEST_ID) == []


def test_blocks_have_no_reset_in_front():
    """Blocks follow each other inside one test: only RESET / RESET_TRAFFIC / BASELINE may reset."""
    for name in ("IDLE", "WRITE", "READ", "WRITE_READ"):
        assert all(rst_n == 1 for rst_n, _, _ in to_cycles(ACTION_INDEX[name]))


@pytest.mark.parametrize("name, cycle", [("IDLE", (1, 0, 0)), ("WRITE", (1, 1, 0)), ("READ", (1, 0, 1)),
                                         ("WRITE_READ", (1, 1, 1)), ("RESET", (0, 0, 0))])
def test_fixed_blocks(name, cycle):
    spec = get_action(ACTION_INDEX[name])
    assert spec.deterministic
    assert to_cycles(spec.id) == [cycle] * spec.n_cycles


def test_random_blocks_need_rng_and_repeat_with_the_same_seed():
    rand50 = ACTION_INDEX["RAND50"]
    with pytest.raises(ValueError):
        to_cycles(rand50)
    a = to_cycles(rand50, np.random.default_rng(5))
    assert a == to_cycles(rand50, np.random.default_rng(5))
    assert len(a) == get_action(rand50).n_cycles


def test_reset_traffic_holds_reset():
    cycles = to_cycles(ACTION_INDEX["RESET_TRAFFIC"], np.random.default_rng(0))
    assert all(rst_n == 0 for rst_n, _, _ in cycles)


def test_block_cycles_percentages():
    rng = np.random.default_rng(1)
    cycles = block_cycles(70, 30, 10, 20000, rng)
    rst, wr, rd = (np.mean([c[i] for c in cycles]) for i in range(3))
    assert rst == pytest.approx(0.9, abs=0.02)
    assert wr == pytest.approx(0.7, abs=0.02)
    assert rd == pytest.approx(0.3, abs=0.02)


def test_plan_cycles_starts_with_one_reset_and_chains_blocks():
    cycles = plan_cycles(ids("WRITE", "IDLE", "END_TEST"))
    assert cycles[0] == RESET_CYCLE
    assert cycles[1:] == [(1, 1, 0)] * 10 + [(1, 0, 0)] * 4


def test_fifo_keeps_its_level_between_blocks():
    """WRITE then IDLE holds a full FIFO: cx_full <0,0,1> (idle while full), impossible from empty."""
    idle_full = BIN_INDEX["cx_full/<auto[0],auto[0],auto[1]>"]
    assert fake_fifo.hits(plan_cycles(ids("WRITE", "IDLE")))[idle_full]
    assert not fake_fifo.hits(plan_cycles(ids("IDLE")))[idle_full]


def test_nine_blocks_can_close_all_bins_in_one_test():
    """A good plan exists within plan_max_len 10 (RESET_TRAFFIC is random: most seeds, not all)."""
    plan = ids("WRITE", "IDLE", "READ", "WRITE_READ", "IDLE", "WRITE", "WRITE_READ", "IDLE", "RESET_TRAFFIC")
    closed = sum(fake_fifo.hits(plan_cycles(plan, np.random.default_rng(s))).all() for s in range(100))
    assert closed >= 90


def test_get_action_rejects_bad_ids():
    for bad in (-1, 9):
        with pytest.raises(ValueError):
            get_action(bad)
