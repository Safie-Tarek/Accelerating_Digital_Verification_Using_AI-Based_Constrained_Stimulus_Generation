"""Tests for fifo_rl/evaluation.py and the scripts that use it."""

import csv
import json

import numpy as np
import pytest

from fifo_rl.agents.random_agent import FixedActionAgent, RandomAgent
from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.coverage_model import N_BINS
from fifo_rl.envs.coverage_env import CoverageEnv
from fifo_rl.evaluation import (
    EpisodeResult,
    constrained_random_stream,
    coverage_curve,
    cycles_to_close,
    long_random_cycles_to_close,
    run_episode,
    run_episodes,
    summarize,
)
from fifo_rl.reward import RewardConfig
from fifo_rl.utils.logger import CsvLogger


def test_episode_stops_at_closure_or_budget():
    """An episode ends exactly when the env says done: 64/64 or the budget used up."""
    env = CoverageEnv(FakeBackend(), budget=40, seed=0)
    for res in run_episodes(env, RandomAgent(seed=0), 30):
        assert 1 <= res.runs <= 40
        assert res.closed == (res.final_bins == N_BINS)
        if not res.closed:
            assert res.runs == 40


def test_total_bins_add_up_and_never_drop():
    """total_bins is the running sum of new_bins and never decreases."""
    env = CoverageEnv(FakeBackend(), budget=20, seed=1)
    res = run_episode(env, RandomAgent(seed=1))
    assert res.total_bins == list(np.cumsum(res.new_bins))
    assert all(a <= b for a, b in zip(res.total_bins, res.total_bins[1:]))


def test_closed_episode_return_is_64_minus_runs_plus_bonus():
    """With unscaled rewards a closed episode returns exactly 64 - T + 10 (see reward.py)."""
    env = CoverageEnv(FakeBackend(), budget=40, seed=2, reward=RewardConfig(scale=1.0))
    closed = [r for r in run_episodes(env, RandomAgent(seed=2), 20) if r.closed]
    assert closed
    for res in closed:
        assert res.episode_return == pytest.approx(N_BINS - res.runs + 10)


def test_same_seeds_same_results():
    """Same env seed + same agent seed -> identical episodes."""
    def play():
        env = CoverageEnv(FakeBackend(), budget=20, seed=5)
        return [(r.actions, r.total_bins) for r in run_episodes(env, RandomAgent(seed=5), 5)]
    assert play() == play()


def test_coverage_curve_pads_with_final_value():
    """A closed episode stays at 64 after closing; an unclosed one keeps its last value."""
    closed = EpisodeResult(actions=[0, 1], new_bins=[60, 4], total_bins=[60, 64], rewards=[0, 0], closed=True)
    short = EpisodeResult(actions=[0, 1, 2], new_bins=[30, 3, 0], total_bins=[30, 33, 33], rewards=[0, 0, 0])
    curve = coverage_curve([closed, short], budget=4)
    assert curve.tolist() == [[60, 64, 64, 64], [30, 33, 33, 33]]
    with pytest.raises(ValueError):
        coverage_curve([short], budget=2)


def test_summarize_numbers():
    """Closure rate, runs-to-close over closed episodes only, runs_used counts failures as the budget."""
    closed = EpisodeResult(actions=[0, 1], new_bins=[60, 4], total_bins=[60, 64], rewards=[1.0, 2.0], closed=True)
    failed = EpisodeResult(actions=[2, 2, 2], new_bins=[30, 3, 0], total_bins=[30, 33, 33], rewards=[0, 0, 0])
    s = summarize([closed, failed], budget=3, n_actions=22)
    assert s["closure_rate"] == 0.5
    assert s["runs_to_close"]["mean"] == 2 and s["runs_to_close_all"] == [2]
    assert s["runs_used"]["mean"] == 2.5
    assert s["return"]["mean"] == 1.5
    assert s["curve_mean"] == [45.0, 48.5, 48.5]
    assert s["action_counts"][2] == 3 and sum(s["action_counts"]) == 5
    json.dumps(s)  # must be JSON-ready


def test_summarize_with_nothing_closed():
    """No closed episode -> runs_to_close stats are None, not a crash."""
    failed = EpisodeResult(actions=[0], new_bins=[30], total_bins=[30], rewards=[0.0])
    assert summarize([failed], budget=1, n_actions=22)["runs_to_close"]["mean"] is None


def test_logger_gets_one_row_per_run(tmp_path):
    """run_episodes with a logger writes one CSV row per run, with the agent's name."""
    env = CoverageEnv(FakeBackend(), budget=10, seed=0)
    with CsvLogger(tmp_path / "log.csv") as logger:
        results = run_episodes(env, FixedActionAgent(), 3, logger)
    with open(tmp_path / "log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(r.runs for r in results)
    assert {row["agent"] for row in rows} == {"fixed_baseline_random"}
    assert [row["episode"] for row in rows][:1] == ["0"]


def test_constrained_random_stream_follows_fifo_test():
    """With the prefix: reset, 100 write-only, 100 read-only, then mixed traffic."""
    stream = constrained_random_stream(np.random.default_rng(0))
    cycles = [next(stream) for _ in range(2201)]
    assert cycles[0] == (0, 0, 0)
    assert all(wr == 1 and rd == 0 for _, wr, rd in cycles[1:101])
    assert all(wr == 0 and rd == 1 for _, wr, rd in cycles[101:201])
    rest = np.array(cycles[201:])
    assert rest[:, 1].mean() == pytest.approx(0.70, abs=0.04)   # wr 70 %
    assert rest[:, 2].mean() == pytest.approx(0.30, abs=0.04)   # rd 30 %
    assert 1 - rest[:, 0].mean() == pytest.approx(0.10, abs=0.03)  # reset 10 %


def test_cycles_to_close_simple_cases():
    """A stream that can never reach full never closes; a stream must start with reset."""
    assert cycles_to_close(iter([(0, 0, 0)] + [(1, 0, 0)] * 50)) is None
    with pytest.raises(ValueError):
        cycles_to_close(iter([(1, 1, 0)]))


def test_long_random_closes_like_questa():
    """The FIFO_test copy closes in the range measured on real Questa (64/64 by 400-500 cycles on seeds 1-5)."""
    res = long_random_cycles_to_close(20, base_seed=0, directed_prefix=True)
    assert res["closure_rate"] == 1.0
    assert 250 <= res["cycles_to_close"]["median"] <= 800
