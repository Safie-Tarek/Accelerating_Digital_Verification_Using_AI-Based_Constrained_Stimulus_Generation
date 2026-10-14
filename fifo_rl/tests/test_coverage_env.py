"""Tests for fifo_rl/envs/coverage_env.py: state, reward, done and seeds of the game."""

import numpy as np
import pytest

from fifo_rl.actions import ACTION_INDEX, N_ACTIONS
from fifo_rl.backends.base import Backend
from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.envs.coverage_env import CoverageEnv
from fifo_rl.reward import LEGACY_REWARD


class ScriptedBackend(Backend):
    """Fake backend for tests: returns pre-set bin vectors and records the (action, seed) of every run."""

    def __init__(self, *runs):
        """Input: the bin vectors to return, in order (one per run)."""
        self.runs = list(runs)
        self.calls = []

    def run(self, action_id, seed):
        """Input: action id and seed. Output: the next pre-set bin vector. Logic: also log the call."""
        self.calls.append((action_id, seed))
        return self.runs.pop(0)


def bins(*indices):
    """Helper. Input: bin indices. Output: bool array of 64 with only those bins True."""
    hit = np.zeros(64, dtype=bool)
    hit[list(indices)] = True
    return hit


def test_reset_gives_fresh_87_state():
    """reset() returns 87 numbers: no bins, budget 1.0, no last action."""
    env = CoverageEnv(FakeBackend(), budget=20)
    state = env.reset()
    assert env.state_size == 87
    assert env.n_actions == N_ACTIONS == 22
    assert state.shape == (87,)
    assert state[:64].sum() == 0
    assert state[64] == 1.0
    assert state[65:].sum() == 0


def test_step_before_reset_fails():
    """step() before reset() is an error."""
    with pytest.raises(RuntimeError):
        CoverageEnv(FakeBackend()).step(0)


def test_reward_uses_compute_reward():
    """Reward = (new bins this episode - 1) / 64; state and info follow the run."""
    env = CoverageEnv(ScriptedBackend(bins(0, 1, 2), bins(2, 3)), budget=5)
    env.reset()
    state, reward, done, info = env.step(4)
    assert reward == pytest.approx((3 - 1) / 64)
    assert info["new_bins"] == 3 and info["total_bins"] == 3 and info["run_bins"] == 3
    assert not done and not info["closed_now"]
    state, reward, done, info = env.step(5)
    assert reward == pytest.approx((1 - 1) / 64)  # bin 2 was already hit, only bin 3 is new
    assert info["total_bins"] == 4
    assert state[:64].sum() == 4
    assert state[65 + 5] == 1.0 and state[65:].sum() == 1
    assert info["action"] == "L1_idle"


def test_run_with_nothing_new_is_penalised():
    """A run that adds no bin earns -step_cost (scaled)."""
    env = CoverageEnv(ScriptedBackend(bins(0), bins(0)), budget=5)
    env.reset()
    env.step(0)
    _, reward, _, info = env.step(0)
    assert info["new_bins"] == 0
    assert reward == pytest.approx(-1 / 64)


def test_custom_reward_config():
    """The env passes its RewardConfig to compute_reward (here: the old new-bins - 0.1 reward)."""
    env = CoverageEnv(ScriptedBackend(bins(0, 1, 2)), budget=5, reward=LEGACY_REWARD)
    env.reset()
    _, reward, _, _ = env.step(0)
    assert reward == pytest.approx(3 - 0.1)


def test_done_when_budget_used():
    """The episode ends after `budget` runs; stepping again is an error."""
    env = CoverageEnv(ScriptedBackend(bins(0), bins(1)), budget=2)
    env.reset()
    assert not env.step(0)[2]
    state, _, done, info = env.step(0)
    assert done and info["runs_left"] == 0 and state[64] == 0.0
    with pytest.raises(RuntimeError):
        env.step(0)


def test_done_and_bonus_when_all_covered():
    """The episode ends when all 64 bins are covered, and only that run gets the closure bonus."""
    env = CoverageEnv(ScriptedBackend(bins(*range(60)), np.ones(64, dtype=bool)), budget=10)
    env.reset()
    _, reward, done, info = env.step(0)
    assert not done and not info["closed_now"]
    assert reward == pytest.approx((60 - 1) / 64)
    _, reward, done, info = env.step(1)
    assert done and info["all_covered"] and info["closed_now"]
    assert reward == pytest.approx((4 - 1 + 10) / 64)


def test_every_run_gets_a_new_reproducible_seed():
    """6 runs get 6 different 32-bit seeds; the same base seed repeats them, another base seed does not."""

    def seeds(base_seed):
        """Input: base seed. Output: the seeds of 2 episodes x 3 runs."""
        backend = ScriptedBackend(*[bins()] * 6)
        env = CoverageEnv(backend, budget=3, seed=base_seed)
        for _ in range(2):
            env.reset()
            for _ in range(3):
                env.step(0)
        return [seed for _, seed in backend.calls]

    first = seeds(0)
    assert len(set(first)) == 6
    assert seeds(0) == first
    assert seeds(1) != first
    assert all(0 <= seed < 2**32 for seed in first)


def test_bad_action():
    """An action id outside 0..21 raises ValueError."""
    env = CoverageEnv(FakeBackend())
    env.reset()
    with pytest.raises(ValueError):
        env.step(N_ACTIONS)


def test_full_episode_on_fake_backend():
    """End to end: a sensible sequence of actions on the fake FIFO reaches 64/64 and ends the episode."""
    env = CoverageEnv(FakeBackend(), budget=40, seed=3)
    env.reset()
    plan = ["L8_rand50", "reset_with_traffic", "L8_rd", "L0_rand50", "L1_idle", "L8_idle", "L7_idle",
            "L7_wr_rd", "L1_wr_rd", "L7_rd", "baseline_random"]
    done, total = False, 0
    for name in plan * 3:
        _, reward, done, info = env.step(ACTION_INDEX[name])
        assert info["total_bins"] >= total
        total = info["total_bins"]
        if done:
            break
    assert done and info["all_covered"] and total == 64
