"""Tests for fifo_rl/actions.py: the 22 recipes and how they turn into stimulus."""

import numpy as np
import pytest

from fifo_rl.actions import (
    ACTION_INDEX,
    ACTIONS,
    FILL_CYCLE,
    N_ACTIONS,
    RESET_CYCLE,
    get_action,
    to_cycles,
    to_plusargs,
)


def rng(seed=0):
    """Helper. Input: a seed (int). Output: a numpy random Generator built from it."""
    return np.random.default_rng(seed)


def test_22_actions_with_unique_names():
    """There are exactly 22 actions, every name is different, and id == position == ACTION_INDEX[name]."""
    assert N_ACTIONS == 22
    assert len({spec.name for spec in ACTIONS}) == 22
    for i, spec in enumerate(ACTIONS):
        assert spec.id == i
        assert ACTION_INDEX[spec.name] == i


def test_id_order_is_level_then_pattern():
    """Ids follow id = 5 * level_index + pattern_index, then 20 and 21 (the order state.py's one-hot uses)."""
    assert ACTION_INDEX["L0_idle"] == 0
    assert ACTION_INDEX["L0_rand50"] == 4
    assert ACTION_INDEX["L1_wr"] == 6
    assert ACTION_INDEX["L7_idle"] == 10
    assert ACTION_INDEX["L8_idle"] == 15
    assert ACTION_INDEX["L8_rd"] == 17
    assert ACTION_INDEX["reset_with_traffic"] == 20
    assert ACTION_INDEX["baseline_random"] == 21


def test_recipe_values():
    """A few actions have exactly the agreed (level, wr %, rd %, rst %, cycles)."""
    spec = get_action(ACTION_INDEX["L8_idle"])
    assert (spec.level, spec.wr_pct, spec.rd_pct, spec.rst_pct, spec.k_cycles) == (8, 0, 0, 0, 10)
    spec = get_action(ACTION_INDEX["L7_wr_rd"])
    assert (spec.level, spec.wr_pct, spec.rd_pct, spec.rst_pct) == (7, 100, 100, 0)
    spec = get_action(ACTION_INDEX["reset_with_traffic"])
    assert (spec.level, spec.wr_pct, spec.rd_pct, spec.rst_pct, spec.k_cycles) == (0, 50, 50, 100, 16)
    spec = get_action(ACTION_INDEX["baseline_random"])
    assert (spec.level, spec.wr_pct, spec.rd_pct, spec.rst_pct, spec.k_cycles) == (0, 70, 30, 10, 18)


def test_16_deterministic_and_6_random():
    """Exactly these 6 actions use randomness; the other 16 are fixed."""
    random_names = {spec.name for spec in ACTIONS if not spec.deterministic}
    assert random_names == {
        "L0_rand50", "L1_rand50", "L7_rand50", "L8_rand50", "reset_with_traffic", "baseline_random",
    }


def test_every_run_starts_with_reset_then_fills():
    """Every run = 1 reset cycle, then `level` write-only cycles, total length 1 + level + K."""
    for spec in ACTIONS:
        cycles = to_cycles(spec.id, rng())
        assert len(cycles) == spec.n_cycles == 1 + spec.level + spec.k_cycles
        assert cycles[0] == RESET_CYCLE
        assert cycles[1 : 1 + spec.level] == [FILL_CYCLE] * spec.level


def test_fixed_patterns_give_exact_cycles():
    """The pattern part of fixed actions is exactly idle / read / write+read on every cycle."""
    pattern = to_cycles(ACTION_INDEX["L7_idle"])[8:]
    assert pattern == [(1, 0, 0)] * 10
    pattern = to_cycles(ACTION_INDEX["L8_rd"])[9:]
    assert pattern == [(1, 0, 1)] * 10
    pattern = to_cycles(ACTION_INDEX["L1_wr_rd"])[2:]
    assert pattern == [(1, 1, 1)] * 10


def test_deterministic_actions_ignore_the_seed():
    """Fixed actions give the same cycles with seed 1, seed 2 or no rng at all."""
    for spec in ACTIONS:
        if spec.deterministic:
            assert to_cycles(spec.id, rng(1)) == to_cycles(spec.id, rng(2)) == to_cycles(spec.id)


def test_random_actions_depend_on_the_seed_and_replay():
    """Random actions: the same seed replays the same run, different seeds give different runs."""
    for spec in ACTIONS:
        if not spec.deterministic:
            assert to_cycles(spec.id, rng(1)) == to_cycles(spec.id, rng(1))
            runs = {tuple(to_cycles(spec.id, rng(seed))) for seed in range(10)}
            assert len(runs) > 1


def test_reset_with_traffic_holds_reset():
    """reset_with_traffic keeps rst_n = 0 on every cycle (needed for the reset-only bins)."""
    cycles = to_cycles(ACTION_INDEX["reset_with_traffic"], rng())
    assert all(rst_n == 0 for rst_n, _, _ in cycles)


def test_random_percentages_are_roughly_right():
    """Over many runs, baseline_random really gives ~10% reset, ~70% write, ~30% read."""
    cycles = []
    for seed in range(300):
        cycles += to_cycles(ACTION_INDEX["baseline_random"], rng(seed))[1:]
    rst_n, wr_en, rd_en = np.array(cycles).T
    assert (1 - rst_n).mean() == pytest.approx(0.10, abs=0.02)
    assert wr_en.mean() == pytest.approx(0.70, abs=0.02)
    assert rd_en.mean() == pytest.approx(0.30, abs=0.02)


def test_random_action_needs_an_rng():
    """Calling a random action without an rng is an error (it would not be reproducible)."""
    with pytest.raises(ValueError):
        to_cycles(ACTION_INDEX["L0_rand50"])


def test_plusargs():
    """to_plusargs gives the test name and the 5 recipe numbers as plusargs."""
    assert to_plusargs(ACTION_INDEX["L8_idle"]) == [
        "+UVM_TESTNAME=FIFO_rl_test",
        "+FILL_LEVEL=8",
        "+WR_PCT=0",
        "+RD_PCT=0",
        "+RST_PCT=0",
        "+NUM_CYCLES=10",
    ]
    assert "+RST_PCT=100" in to_plusargs(ACTION_INDEX["reset_with_traffic"])


def test_bad_action_id():
    """Ids outside 0..21 raise ValueError in every function."""
    for bad in (-1, 22):
        with pytest.raises(ValueError):
            get_action(bad)
        with pytest.raises(ValueError):
            to_cycles(bad)
        with pytest.raises(ValueError):
            to_plusargs(bad)
