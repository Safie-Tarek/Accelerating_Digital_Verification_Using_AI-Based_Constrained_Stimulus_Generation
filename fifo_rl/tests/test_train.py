"""Smoke tests for scripts/train.py, the dqn option of scripts/evaluate.py and the training plots.

They use StubAgent (Contract E interface, no learning), so they don't wait for the real DQNAgent.
"""

import csv
import json
import sys
import types

import pytest

from fifo_rl.agents.stub_agent import StubAgent
from fifo_rl.config import load_config
from fifo_rl.evaluation import GreedyWrapper
from scripts import evaluate, plot, train


def _small_cfg(**overrides):
    return load_config(overrides={"n_episodes": 6, "eval_every": 3, "eval_episodes": 5, "buffer_capacity": 50,
                                  "eps_decay_steps": 40, **overrides})


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_train_writes_everything(tmp_path):
    cfg = _small_cfg()
    result = train.train(cfg, tmp_path, StubAgent, verbose=False)
    for name in ("config.yaml", "steps.csv", "eval.csv", "last.pt", "best.pt", "stub_summary.json"):
        assert (tmp_path / name).exists(), name

    steps = _rows(tmp_path / "steps.csv")
    assert len(steps) == result["steps"]
    assert {r["episode"] for r in steps} == {str(e) for e in range(6)}
    assert all(r["loss"] == "" for r in steps)          # the stub never learns
    eps = [float(r["epsilon"]) for r in steps]
    assert eps[0] < 1.0 and eps[-1] == pytest.approx(0.05) and eps == sorted(eps, reverse=True)

    evals = _rows(tmp_path / "eval.csv")
    assert [int(r["episode"]) for r in evals] == [3, 6]
    assert evals[0]["best"] == "1"

    summary = json.loads((tmp_path / "stub_summary.json").read_text())
    assert summary["agent"] == "stub" and summary["budget"] == cfg["budget_eval"]
    assert summary["n_episodes"] == cfg["eval_episodes"] and 0 < summary["coverage_auc"] <= 1
    assert json.loads((tmp_path / "last.pt").read_text())["steps"] == result["steps"]


def test_last_partial_block_is_evaluated(tmp_path):
    """n_episodes not a multiple of eval_every: the last episode is still evaluated."""
    train.train(_small_cfg(n_episodes=5), tmp_path, StubAgent, verbose=False)
    assert [int(r["episode"]) for r in _rows(tmp_path / "eval.csv")] == [3, 5]


def test_validation_episodes_are_fixed():
    """Every evaluation replays the same episodes: same agent seed -> identical summaries."""
    cfg = _small_cfg()
    assert train.evaluate(StubAgent(seed=1), cfg) == train.evaluate(StubAgent(seed=1), cfg)


def test_main_cli(tmp_path, capsys):
    train.main(["--agent", "stub", "--out", str(tmp_path), "--n-episodes", "4", "--eval-every", "2",
                "--eval-episodes", "3"])
    assert "best: closes" in capsys.readouterr().out
    assert len(_rows(tmp_path / "eval.csv")) == 2


def test_greedy_wrapper_forces_greedy():
    calls = []

    class Spy:
        name = "spy"

        def act(self, state, greedy=False):
            calls.append(greedy)
            return 0

    wrapped = GreedyWrapper(Spy())
    assert wrapped.name == "spy" and wrapped.act(None) == 0 and calls == [True]


def test_evaluate_dqn_checkpoint(tmp_path, monkeypatch):
    """evaluate.py --agents dqn loads DQNAgent.load(checkpoint); StubAgent stands in for it."""
    train.train(_small_cfg(), tmp_path / "train", StubAgent, verbose=False)
    monkeypatch.setitem(sys.modules, "fifo_rl.agents.dqn", types.SimpleNamespace(DQNAgent=StubAgent))
    evaluate.main(["--agents", "dqn", "--checkpoint", str(tmp_path / "train" / "best.pt"), "--episodes", "4",
                   "--long-seeds", "0", "--out", str(tmp_path / "eval")])
    summary = json.loads((tmp_path / "eval" / "stub_summary.json").read_text())
    assert summary["n_episodes"] == 4 and summary["budget"] == 40


def test_evaluate_dqn_needs_checkpoint():
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--agents", "dqn"])


def test_training_plots(tmp_path):
    folders = []
    for seed in (0, 1):
        folders.append(tmp_path / f"stub_seed{seed}")
        train.train(_small_cfg(seed=seed), folders[-1], StubAgent, verbose=False)
    groups = plot.load_training_runs(folders)
    assert list(groups) == ["stub"] and len(groups["stub"]) == 2
    plot.main(["--train", *map(str, folders), "--train-out", str(tmp_path / "plots")])
    assert (tmp_path / "plots" / "learning_curve.png").stat().st_size > 0
    assert (tmp_path / "plots" / "loss_epsilon.png").stat().st_size > 0
