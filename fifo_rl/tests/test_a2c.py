"""Tests for fifo_rl/agents/a2c.py: the legality mask, the two networks and choosing a block."""

import numpy as np
import torch

from fifo_rl.agents.a2c import A2CAgent, legal_mask
from fifo_rl.campaign.campaign_store import CampaignStore
from fifo_rl.evaluation import GreedyWrapper
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID, N_ACTIONS
from fifo_rl.plan.state_encoder import StateEncoder
from fifo_rl.plan.stimulus_plan import StimulusPlan

WRITE = ACTION_INDEX["WRITE"]
STATE_DIM = 93
PLAN_MAX_LEN = 10


def a2c_cfg(**overrides):
    """The Contract F settings, small hidden layers so the tests run in milliseconds."""
    return {"gamma": 0.99, "a2c_hidden": [32, 32], "a2c_lr": 7e-4, "a2c_grad_clip": 0.5, "vf_coef": 0.5,
            "ent_coef": 0.01, "gae_lambda": 0.95, "a2c_rollout_tests": 4, "a2c_normalize_adv": True, **overrides}


def make_agent(seed=0, **overrides):
    return A2CAgent(STATE_DIM, N_ACTIONS, a2c_cfg(**overrides), seed=seed)


def real_state(n_blocks):
    """The state the real StateEncoder gives after n_blocks WRITE blocks in the current plan."""
    encoder, store = StateEncoder(PLAN_MAX_LEN), CampaignStore(30)
    plan = StimulusPlan(0, 0, PLAN_MAX_LEN)
    for _ in range(n_blocks):
        plan.add(WRITE, encoder.encode(store, plan))
    return encoder.encode(store, plan)


def same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# --- the legality mask --------------------------------------------------------------------------

def test_mask_allows_everything_until_the_plan_is_full():
    for n_blocks in (0, 1, PLAN_MAX_LEN - 1):
        assert legal_mask(real_state(n_blocks), N_ACTIONS).all()


def test_mask_allows_only_end_test_in_a_full_plan():
    mask = legal_mask(real_state(PLAN_MAX_LEN), N_ACTIONS)
    assert mask[END_TEST_ID] and mask.sum() == 1


def test_mask_works_row_by_row_on_a_batch():
    batch = np.stack([real_state(0), real_state(PLAN_MAX_LEN), real_state(3)])
    mask = legal_mask(batch, N_ACTIONS)
    assert mask.shape == (3, N_ACTIONS)
    assert mask[0].all() and mask[2].all() and mask[1].tolist() == [i == END_TEST_ID for i in range(N_ACTIONS)]


# --- the networks -------------------------------------------------------------------------------

def test_name_epsilon_counters_and_shapes():
    agent = make_agent()
    assert agent.name == "a2c" and agent.epsilon == 0.0 and agent.steps == 0 and agent.updates == 0
    batch = np.random.default_rng(0).random((5, STATE_DIM)).astype(np.float32)
    probs = agent.probabilities(batch)
    assert probs.shape == (5, N_ACTIONS) and np.allclose(probs.sum(axis=1), 1.0) and (probs > 0).all()
    assert agent.probabilities(batch[0]).shape == (N_ACTIONS,)
    assert agent.values(batch).shape == (5,) and isinstance(agent.values(batch[0]), float)


def test_full_plan_gives_end_test_probability_1():
    agent = make_agent()
    probs = agent.probabilities(real_state(PLAN_MAX_LEN))
    assert probs[END_TEST_ID] == 1.0 and probs.sum() == 1.0


# --- choosing a block ---------------------------------------------------------------------------

def test_illegal_blocks_are_never_chosen():
    agent = make_agent()
    full = real_state(PLAN_MAX_LEN)
    assert {agent.act(full) for _ in range(200)} == {END_TEST_ID}
    assert agent.act(full, greedy=True) == END_TEST_ID


def test_sampling_tries_every_block():
    agent = make_agent()
    state = real_state(0)
    assert {agent.act(state) for _ in range(2000)} == set(range(N_ACTIONS))


def test_greedy_is_the_most_likely_block_every_time():
    agent = make_agent()
    state = real_state(2)
    best = int(np.argmax(agent.probabilities(state)))
    assert all(agent.act(state, greedy=True) == best for _ in range(20))
    assert GreedyWrapper(agent).act(state) == best        # the wrapper evaluation uses


def test_same_seed_same_agent_other_seed_other_agent():
    a, b, c = make_agent(seed=3), make_agent(seed=3), make_agent(seed=4)
    assert same_weights(a.actor, b.actor) and same_weights(a.critic, b.critic)
    state = real_state(1)
    assert [a.act(state) for _ in range(30)] == [b.act(state) for _ in range(30)]
    assert not same_weights(a.actor, c.actor)


def test_acting_does_not_change_the_networks():
    agent = make_agent()
    before = make_agent()
    for _ in range(50):
        agent.act(real_state(0))
    assert same_weights(agent.actor, before.actor) and same_weights(agent.critic, before.critic)
