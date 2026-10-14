"""Tests for fifo_rl/backends/fake_backend.py: each action, run on the Python FIFO, hits the right bins."""

import time

import numpy as np
import pytest

from fifo_rl.actions import ACTION_INDEX, ACTIONS, N_ACTIONS
from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.coverage_model import BIN_INDEX, N_BINS

backend = FakeBackend()


def bins_over_seeds(action_name, seeds=range(30)):
    """Helper.

    Input:  an action name and a list of seeds.
    Output: bool array of 64 - bins the action hits in at least one of those seeds.
    """
    action_id = ACTION_INDEX[action_name]
    return np.any([backend.run(action_id, seed) for seed in seeds], axis=0)


def test_returns_bool_vector_of_64():
    """run() returns Contract C's shape: 64 true/false values."""
    hit = backend.run(0, seed=0)
    assert hit.shape == (N_BINS,)
    assert hit.dtype == bool


def test_same_seed_same_bins():
    """Running any action twice with the same seed gives identical bins."""
    for spec in ACTIONS:
        assert (backend.run(spec.id, 7) == backend.run(spec.id, 7)).all()


def test_deterministic_actions_ignore_the_seed():
    """Fixed actions give identical bins for different seeds."""
    for spec in ACTIONS:
        if spec.deterministic:
            assert (backend.run(spec.id, 1) == backend.run(spec.id, 2)).all()


def test_l0_idle_hits_only_the_free_reset_bins():
    """L0_idle (reset then nothing) only ever gets the 16 free bins."""
    assert backend.run(ACTION_INDEX["L0_idle"], 0).sum() == 16


# Each action exists for at least one hard bin; check it really hits it.
TARGETS = {
    "L1_idle": ["cx_almostempty/<auto[0],auto[0],auto[1]>"],
    "L1_wr_rd": ["cx_almostempty/<auto[1],auto[1],auto[1]>"],
    "L0_rd": ["cx_underflow/<auto[0],auto[1],auto[1]>", "cx_empty/<auto[0],auto[1],auto[1]>"],
    "L7_idle": ["cx_almostfull/<auto[0],auto[0],auto[1]>"],
    "L7_rd": ["cx_almostempty/<auto[0],auto[1],auto[1]>"],
    "L7_wr_rd": ["cx_almostfull/<auto[1],auto[1],auto[1]>"],
    "L8_idle": ["cx_full/<auto[0],auto[0],auto[1]>"],
    "L8_wr": ["cx_overflow/<auto[1],auto[0],auto[1]>", "cx_wr_ack/<auto[1],auto[0],auto[0]>"],
    "L8_rd": ["cx_almostfull/<auto[0],auto[1],auto[1]>"],
    "reset_with_traffic": [
        "cx_empty/<auto[1],auto[0],auto[1]>",
        "cx_empty/<auto[1],auto[1],auto[1]>",
        "cx_underflow/<auto[1],auto[1],auto[1]>",
    ],
}


@pytest.mark.parametrize("action_name", TARGETS)
def test_action_hits_its_target_bins(action_name):
    """Each action in TARGETS hits the hard bins it was designed for."""
    hit = bins_over_seeds(action_name)  # random actions may miss a bin in one particular run
    for name in TARGETS[action_name]:
        assert hit[BIN_INDEX[name]], f"{action_name} should hit {name}"


def test_reset_only_bins_need_reset_with_traffic_or_baseline():
    """The 3 reset-only bins are never hit by the 20 level/pattern actions."""
    reset_only = [BIN_INDEX[name] for name in TARGETS["reset_with_traffic"]]
    for spec in ACTIONS:
        if spec.name not in ("reset_with_traffic", "baseline_random"):
            assert not bins_over_seeds(spec.name)[reset_only].any()


def test_reset_with_traffic_usually_gets_all_three_in_one_run():
    """One reset_with_traffic run gets all 3 reset-only bins in more than 90% of seeds."""
    reset_only = [BIN_INDEX[name] for name in TARGETS["reset_with_traffic"]]
    action_id = ACTION_INDEX["reset_with_traffic"]
    got_all = [backend.run(action_id, seed)[reset_only].all() for seed in range(200)]
    assert np.mean(got_all) > 0.9


def test_all_actions_together_cover_all_64_bins():
    """The action set is complete: together the 22 actions can reach every bin."""
    union = np.any([bins_over_seeds(spec.name) for spec in ACTIONS], axis=0)
    assert union.all()


def test_fast_enough_for_training():
    """1000 runs take on average less than 5 ms each (measured ~0.4 ms; target < 1 ms)."""
    start = time.perf_counter()
    for i in range(1000):
        backend.run(i % N_ACTIONS, i)
    assert (time.perf_counter() - start) / 1000 < 0.005  # target < 1 ms; loose bound for slow CI
