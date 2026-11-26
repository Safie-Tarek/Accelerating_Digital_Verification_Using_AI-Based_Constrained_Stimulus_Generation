"""Tests for fifo_rl/agents/a2c.py: the legality mask, the two networks, choosing a block, learning."""

import copy

import numpy as np
import pytest
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


# --- learning -----------------------------------------------------------------------------------

def observe_test(agent, blocks, reward=0.1, done=False):
    """One finished test as CampaignRunner hands it over: the blocks, then END_TEST."""
    for k, block in enumerate(blocks):
        agent.observe(real_state(k), block, reward, real_state(k + 1), False)
    agent.observe(real_state(len(blocks)), END_TEST_ID, reward, real_state(0), done)


def test_observe_counts_and_learn_waits_for_whole_tests():
    agent = make_agent(a2c_rollout_tests=2)
    agent.observe(real_state(0), WRITE, 0.1, real_state(1), False)
    agent.observe(real_state(1), WRITE, 0.1, real_state(2), False)
    assert agent.steps == 2 and agent.learn() is None             # half a test: never update here
    agent.observe(real_state(2), END_TEST_ID, 0.1, real_state(0), False)
    assert agent.learn() is None and len(agent.rollout) == 3      # 1 test of the 2 needed
    observe_test(agent, [WRITE])
    loss = agent.learn()
    assert isinstance(loss, float) and agent.updates == 1 and len(agent.rollout) == 0
    assert set(agent.last_stats) == {"policy_loss", "value_loss", "entropy", "n_transitions"}
    assert agent.last_stats["n_transitions"] == 5


def test_campaign_end_updates_at_once():
    agent = make_agent(a2c_rollout_tests=4)
    observe_test(agent, [WRITE, WRITE], done=True)               # 1 test, but the campaign is over
    assert isinstance(agent.learn(), float) and agent.updates == 1


def test_forced_end_test_does_not_move_the_actor():
    """A full plan: PlanBuilder chose END_TEST, not the agent, so only the critic may learn from it."""
    agent = make_agent()
    actor_before, critic_before = copy.deepcopy(agent.actor), copy.deepcopy(agent.critic)
    agent.observe(real_state(PLAN_MAX_LEN), END_TEST_ID, 1.0, real_state(0), True)
    agent.learn()
    assert same_weights(agent.actor, actor_before)
    assert not same_weights(agent.critic, critic_before)


def repeat_one_choice(agent, reward, n_updates):
    """Learn n_updates times from: in state s, WRITE gave `reward`, and the campaign ended."""
    for _ in range(n_updates):
        agent.observe(real_state(0), WRITE, reward, real_state(1), True)
        agent.learn()


def test_a_block_that_did_better_than_expected_becomes_likelier():
    agent = make_agent(a2c_lr=1e-2, ent_coef=0.0)
    before = agent.probabilities(real_state(0))[WRITE]
    repeat_one_choice(agent, reward=1.0, n_updates=30)
    assert agent.probabilities(real_state(0))[WRITE] > before + 0.05


def test_a_block_that_did_worse_than_expected_becomes_rarer():
    agent = make_agent(a2c_lr=1e-2, ent_coef=0.0)
    before = agent.probabilities(real_state(0))[WRITE]
    repeat_one_choice(agent, reward=-1.0, n_updates=30)
    assert agent.probabilities(real_state(0))[WRITE] < before - 0.05


def test_the_critic_learns_what_comes_back():
    agent = make_agent(a2c_lr=1e-2)
    repeat_one_choice(agent, reward=1.0, n_updates=1)
    first = agent.last_stats["value_loss"]
    repeat_one_choice(agent, reward=1.0, n_updates=60)
    assert agent.last_stats["value_loss"] < first / 10
    assert abs(agent.values(real_state(0)) - 1.0) < 0.1          # V(s) -> the reward that came back


def test_entropy_bonus_keeps_the_dice_open():
    """With a big entropy weight the policy stays spread out even when one block keeps paying."""
    greedy, curious = make_agent(a2c_lr=1e-2, ent_coef=0.0), make_agent(a2c_lr=1e-2, ent_coef=1.0)
    repeat_one_choice(greedy, reward=1.0, n_updates=40)
    repeat_one_choice(curious, reward=1.0, n_updates=40)
    assert curious.probabilities(real_state(0))[WRITE] < greedy.probabilities(real_state(0))[WRITE]


def test_tiny_and_big_rollouts_give_finite_numbers():
    tiny = make_agent()
    tiny.observe(real_state(0), END_TEST_ID, 0.5, real_state(0), True)    # an empty test, campaign over
    assert np.isfinite(tiny.learn()) and all(np.isfinite(v) for v in tiny.last_stats.values())
    big = make_agent(a2c_rollout_tests=4)
    for _ in range(4):
        observe_test(big, [WRITE, WRITE, WRITE])                          # 16 transitions: normalized
    assert np.isfinite(big.learn()) and big.last_stats["n_transitions"] == 16


def trained_agent(n_updates=3, **overrides):
    """An agent that has already learned a little (so Adam's momentum is not zero)."""
    agent = make_agent(**overrides)
    for _ in range(n_updates):
        observe_test(agent, [WRITE, WRITE], reward=0.3, done=True)
        agent.learn()
    return agent


# --- saving and loading -------------------------------------------------------------------------

def test_save_then_load_gives_the_same_agent(tmp_path):
    agent = trained_agent()
    agent.save(tmp_path / "a2c.pt")
    loaded = A2CAgent.load(tmp_path / "a2c.pt")             # nothing but the file is needed
    batch = np.stack([real_state(k) for k in range(PLAN_MAX_LEN + 1)])
    np.testing.assert_array_equal(loaded.probabilities(batch), agent.probabilities(batch))
    np.testing.assert_array_equal(loaded.values(batch), agent.values(batch))
    assert [loaded.act(s, greedy=True) for s in batch] == [agent.act(s, greedy=True) for s in batch]
    assert (loaded.name, loaded.steps, loaded.updates, loaded.seed) == ("a2c", agent.steps, agent.updates, agent.seed)
    assert loaded.cfg == agent.cfg and len(loaded.rollout) == 0


def test_loaded_agent_keeps_learning_exactly_where_it_stopped(tmp_path):
    """Same next update for the original and the loaded agent: Adam's momentum was saved too."""
    agent = trained_agent()
    agent.save(tmp_path / "a2c.pt")
    loaded = A2CAgent.load(tmp_path / "a2c.pt")
    for a in (agent, loaded):
        observe_test(a, [WRITE, WRITE, WRITE], reward=0.7, done=True)
        assert isinstance(a.learn(), float)
    assert loaded.updates == agent.updates == 4
    assert same_weights(loaded.actor, agent.actor) and same_weights(loaded.critic, agent.critic)


def test_load_refuses_a_dqn_checkpoint(tmp_path):
    from fifo_rl.agents.dqn import DQNAgent
    from fifo_rl.config import load_config

    DQNAgent(STATE_DIM, N_ACTIONS, load_config()).save(tmp_path / "dqn.pt")
    with pytest.raises(ValueError, match="dqn"):
        A2CAgent.load(tmp_path / "dqn.pt")


def test_gradient_is_clipped():
    """A huge reward gives a huge gradient; what the optimizer receives is cut to a2c_grad_clip."""
    agent = make_agent(a2c_grad_clip=0.5)
    agent.observe(real_state(0), WRITE, 1e6, real_state(1), True)
    agent.learn()
    grads = [p.grad for p in list(agent.actor.parameters()) + list(agent.critic.parameters())]
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
    assert total_norm <= 0.5 + 1e-4
