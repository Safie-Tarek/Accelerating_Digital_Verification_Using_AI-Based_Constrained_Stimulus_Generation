"""Tests for fifo_rl/reward.py: the reward formula, its settings and its whole-episode properties."""

import numpy as np
import pytest

from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.coverage_model import N_BINS
from fifo_rl.envs.coverage_env import CoverageEnv
from fifo_rl.reward import LEGACY_REWARD, RewardConfig, compute_reward

RAW = RewardConfig(scale=1.0)  # default cost and bonus, but in plain "bins" units


def test_defaults():
    """Default settings: one run costs one bin, closing pays 10 bins, everything scaled by 1/64."""
    config = RewardConfig()
    assert config.step_cost == 1.0
    assert config.closure_bonus == 10.0
    assert config.scale == pytest.approx(1 / 64)


def test_new_bins_minus_step_cost_scaled():
    """A normal run: (new bins - 1) / 64."""
    assert compute_reward(3, False) == pytest.approx((3 - 1) / 64)
    assert compute_reward(37, False) == pytest.approx(36 / 64)


def test_run_with_nothing_new_costs_one_bin():
    """A run that finds nothing earns exactly -step_cost (scaled): a wasted run is punished once."""
    assert compute_reward(0, False) == pytest.approx(-1 / 64)


def test_closing_run_gets_the_bonus():
    """The run that completes coverage earns its new bins - cost + bonus."""
    assert compute_reward(4, True) == pytest.approx((4 - 1 + 10) / 64)


def test_unscaled_values_are_in_bins():
    """With scale = 1 the reward reads directly in bins."""
    assert compute_reward(5, False, RAW) == pytest.approx(4.0)
    assert compute_reward(1, True, RAW) == pytest.approx(10.0)


def test_more_new_bins_is_always_better():
    """Within one step, the ranking of runs depends only on new bins (the cost is the same for all)."""
    rewards = [compute_reward(n, False) for n in range(N_BINS + 1)]
    assert all(a < b for a, b in zip(rewards, rewards[1:]))


def test_closing_beats_almost_closing():
    """Getting the last bin and closing is worth far more than one ordinary bin."""
    assert compute_reward(1, True) - compute_reward(0, False) > 10 * (compute_reward(1, False) - compute_reward(0, False))


def test_legacy_reward_is_the_old_formula():
    """LEGACY_REWARD reproduces the first agreed reward: new bins - 0.1, no bonus, no scaling."""
    assert compute_reward(3, False, LEGACY_REWARD) == pytest.approx(2.9)
    assert compute_reward(3, True, LEGACY_REWARD) == pytest.approx(2.9)


def test_bad_settings_are_rejected():
    """Negative cost or bonus, or a scale <= 0, raise ValueError."""
    with pytest.raises(ValueError):
        RewardConfig(step_cost=-1)
    with pytest.raises(ValueError):
        RewardConfig(closure_bonus=-1)
    with pytest.raises(ValueError):
        RewardConfig(scale=0)


def test_negative_new_bins_is_rejected():
    """new_bins can never be negative (bins never get un-hit)."""
    with pytest.raises(ValueError):
        compute_reward(-1, False)


def play(env, rng):
    """Helper.

    Input:  an env and a numpy Generator.
    Output: (list of rewards, number of runs T, closed?) for one episode of random actions.
    """
    env.reset()
    rewards, done, info = [], False, {}
    while not done:
        _, reward, done, info = env.step(int(rng.integers(env.n_actions)))
        rewards.append(reward)
    return rewards, len(rewards), info["all_covered"]


def test_episode_return_is_64_minus_runs_plus_bonus():
    """Key property: an episode that closes in T runs sums (unscaled) to exactly 64 - T + 10.

    So every closing episode gets the same 64 bins, and the only way to score more is fewer runs.
    """
    env = CoverageEnv(FakeBackend(), budget=60, reward=RAW, seed=5)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        rewards, runs, closed = play(env, rng)
        if closed:
            assert sum(rewards) == pytest.approx(N_BINS - runs + 10)
            checked += 1
    assert checked > 10


def test_episode_return_without_closing_has_no_bonus():
    """An episode cut by the budget sums to (bins reached) - T, with no bonus."""
    env = CoverageEnv(FakeBackend(), budget=2, reward=RAW, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        rewards, runs, closed = play(env, rng)
        assert not closed  # 2 random runs never reach 64/64
        assert sum(rewards) == pytest.approx(env.encoder.checklist.sum() - runs)


def test_fewer_runs_means_higher_return():
    """Scaled returns: closing in 5 runs beats closing in 20, which beats stopping at 63/64."""
    def episode_return(new_bins_per_run, closed):
        """Input: new bins of each run, whether the last run closed. Output: the scaled return."""
        last = len(new_bins_per_run) - 1
        return sum(compute_reward(n, closed and i == last) for i, n in enumerate(new_bins_per_run))

    fast = episode_return([40, 10, 8, 4, 2], closed=True)
    slow = episode_return([40, 10, 8, 4] + [0] * 15 + [2], closed=True)
    stuck = episode_return([40, 10, 8, 4, 1] + [0] * 15, closed=False)
    assert fast > slow > stuck
    assert fast == pytest.approx((64 - 5 + 10) / 64)
