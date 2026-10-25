"""Tests for fifo_rl/agents/schedules.py and fifo_rl/config.py."""

import argparse
import math

import pytest

from fifo_rl.agents.schedules import ExpEpsilon, LinearEpsilon, make_schedule
from fifo_rl.config import add_config_args, config_from_args, load_config, save_config


def test_linear_end_points_and_midpoint():
    s = LinearEpsilon(1.0, 0.05, 1000)
    assert s.value(0) == 1.0
    assert s.value(500) == pytest.approx(0.525)
    assert s.value(1000) == pytest.approx(0.05)
    assert s.value(10**6) == pytest.approx(0.05)
    assert s.value(-5) == 1.0


def test_linear_never_increases():
    s = LinearEpsilon(1.0, 0.05, 300)
    values = [s.value(t) for t in range(400)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_exp_decay():
    s = ExpEpsilon(1.0, 0.05, 1000)
    assert s.value(0) == 1.0
    assert s.value(1000) == pytest.approx(0.05 + 0.95 / math.e)
    assert 0.05 < s.value(20_000) < 0.0501


@pytest.mark.parametrize("cls", [LinearEpsilon, ExpEpsilon])
def test_bad_schedule_arguments(cls):
    with pytest.raises(ValueError):
        cls(0.05, 1.0, 100)   # end above start
    with pytest.raises(ValueError):
        cls(1.0, 0.05, 0)


def test_make_schedule_from_config():
    cfg = load_config()
    assert isinstance(make_schedule(cfg), LinearEpsilon)
    assert isinstance(make_schedule({**cfg, "eps_schedule": "exp"}), ExpEpsilon)
    with pytest.raises(ValueError):
        make_schedule({**cfg, "eps_schedule": "cosine"})


def test_default_config_has_contract_keys():
    cfg = load_config()
    for key in ("hidden", "lr", "gamma", "batch_size", "buffer_capacity", "learning_starts", "train_freq",
                "target_sync", "double", "eps_start", "eps_end", "eps_decay_steps", "budget_train",
                "budget_eval", "n_episodes", "eval_every", "eval_episodes", "seed"):
        assert key in cfg
    assert cfg["hidden"] == [128, 128] and isinstance(cfg["lr"], float) and cfg["double"] is True


def test_cli_overrides_are_typed():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args(["--double", "false", "--seed", "3", "--lr", "3e-4",
                                              "--hidden", "[64, 32]", "--eps-schedule", "exp"]))
    assert cfg["double"] is False and cfg["seed"] == 3 and cfg["lr"] == 3e-4
    assert cfg["hidden"] == [64, 32] and cfg["eps_schedule"] == "exp"
    assert cfg["gamma"] == load_config()["gamma"]  # untouched keys keep the default


def test_config_file_on_top_of_defaults(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text("n_episodes: 7\ndouble: false\n")
    cfg = load_config(path, {"seed": 9})
    assert cfg["n_episodes"] == 7 and cfg["double"] is False and cfg["seed"] == 9
    assert cfg["batch_size"] == 64
    save_config(cfg, tmp_path / "copy.yaml")
    assert load_config(tmp_path / "copy.yaml") == cfg


def test_unknown_keys_rejected(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("gama: 0.5\n")
    with pytest.raises(ValueError, match="gama"):
        load_config(path)
    with pytest.raises(ValueError, match="nope"):
        load_config(overrides={"nope": 1})
