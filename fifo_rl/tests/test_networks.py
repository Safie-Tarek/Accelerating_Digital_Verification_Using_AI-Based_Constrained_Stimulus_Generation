"""Tests for fifo_rl/agents/networks.py: the Q-network's shape and layers."""

import pytest
import torch
from torch import nn

from fifo_rl.agents.networks import QNetwork
from fifo_rl.config import load_config


def test_scores_one_number_per_action():
    """A batch of 5 states of 87 numbers gives 5 rows of 22 scores."""
    net = QNetwork(87, [128, 128], 22)
    scores = net(torch.zeros(5, 87))
    assert scores.shape == (5, 22)
    assert scores.dtype == torch.float32


def test_single_state_gives_one_row():
    net = QNetwork(87, [128, 128], 22)
    assert net(torch.zeros(87)).shape == (22,)


def test_layers_follow_the_hidden_sizes():
    """[128, 128] -> 87->128, ReLU, 128->128, ReLU, 128->22."""
    net = QNetwork(87, [128, 128], 22)
    linears = [m for m in net.layers if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(87, 128), (128, 128), (128, 22)]
    assert sum(isinstance(m, nn.ReLU) for m in net.layers) == 2
    assert isinstance(net.layers[-1], nn.Linear)  # no activation on the scores


def test_other_sizes_need_no_code_change():
    """The sizes are only arguments: a new state or action design just passes other numbers."""
    net = QNetwork(10, [16], 3)
    assert net(torch.zeros(4, 10)).shape == (4, 3)
    assert QNetwork(87, [], 22)(torch.zeros(1, 87)).shape == (1, 22)  # no hidden layer at all


def test_default_config_builds_the_planned_network():
    cfg = load_config()
    net = QNetwork(87, cfg["hidden"], 22)
    n_weights = sum(p.numel() for p in net.parameters())
    assert n_weights == (87 * 128 + 128) + (128 * 128 + 128) + (128 * 22 + 22)  # 30,486


def test_same_seed_same_weights():
    torch.manual_seed(0)
    a = QNetwork(87, [128, 128], 22)
    torch.manual_seed(0)
    b = QNetwork(87, [128, 128], 22)
    x = torch.rand(3, 87)
    assert torch.equal(a(x), b(x))


def test_bad_sizes_are_rejected():
    with pytest.raises(ValueError):
        QNetwork(0, [128], 22)
    with pytest.raises(ValueError):
        QNetwork(87, [128], 0)
    with pytest.raises(ValueError):
        QNetwork(87, [128, 0], 22)


def test_grad_clip_is_in_the_config():
    """The agent clips gradients at cfg["grad_clip"]; unknown keys are rejected, so it must exist."""
    assert load_config()["grad_clip"] == 10.0
