"""Tests for fifo_rl/agents/random_agent.py: the two baseline agents."""

import numpy as np
import pytest

from fifo_rl.actions import ACTION_INDEX, N_ACTIONS
from fifo_rl.agents.random_agent import FixedActionAgent, RandomAgent

STATE = np.zeros(87, dtype=np.float32)


def test_random_agent_picks_every_action_in_range():
    """Over many picks every one of the 22 actions appears, and nothing outside 0..21."""
    agent = RandomAgent(seed=0)
    picks = [agent.act(STATE) for _ in range(5000)]
    assert set(picks) == set(range(N_ACTIONS))
    assert all(isinstance(a, int) for a in picks)


def test_random_agent_is_roughly_uniform():
    """Each action gets about 1/22 of the picks (within 20%)."""
    agent = RandomAgent(seed=1)
    counts = np.bincount([agent.act(STATE) for _ in range(22_000)], minlength=N_ACTIONS)
    assert counts.min() > 800 and counts.max() < 1200


def test_random_agent_same_seed_same_choices():
    """Same seed -> same sequence of actions (reproducible experiments)."""
    a, b = RandomAgent(seed=7), RandomAgent(seed=7)
    assert [a.act(STATE) for _ in range(50)] == [b.act(STATE) for _ in range(50)]


def test_random_agent_ignores_the_state():
    """The baseline does not look at coverage: a different state gives the same choices."""
    a, b = RandomAgent(seed=3), RandomAgent(seed=3)
    assert [a.act(STATE) for _ in range(20)] == [b.act(np.ones(87)) for _ in range(20)]


def test_fixed_agent_defaults_to_baseline_random():
    """The default fixed agent repeats action 21 (the plan's 70/30/10 recipe)."""
    agent = FixedActionAgent()
    assert agent.action_id == ACTION_INDEX["baseline_random"] == 21
    assert agent.name == "fixed_baseline_random"
    assert {agent.act(STATE) for _ in range(10)} == {21}


def test_fixed_agent_other_action_and_bad_id():
    """Any valid id can be fixed; an invalid one is refused."""
    assert FixedActionAgent(15).name == "fixed_L8_idle"
    with pytest.raises(ValueError):
        FixedActionAgent(22)
