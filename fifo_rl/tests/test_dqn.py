"""Tests for fifo_rl/agents/dqn.py: acting, learning, the DQN / DDQN target, saving and loading."""

import json

import numpy as np
import pytest
import torch

from fifo_rl.agents.dqn import DQNAgent
from fifo_rl.agents.replay_buffer import ReplayBuffer
from fifo_rl.agents.schedules import LinearEpsilon
from fifo_rl.config import load_config
from scripts import evaluate, train


def small_cfg(**overrides):
    """The default config with small numbers, so the tests run in milliseconds."""
    return load_config(overrides={"learning_starts": 8, "batch_size": 8, "target_sync": 1000, **overrides})


def filled_buffer(n=32, state_dim=87, n_actions=22, seed=0):
    """A buffer holding n random transitions."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(100, state_dim, seed=seed)
    for _ in range(n):
        buf.push(rng.random(state_dim), int(rng.integers(n_actions)), rng.random(), rng.random(state_dim), 0.0)
    return buf


def same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# --- acting -------------------------------------------------------------------------------------

def test_scores_and_action_shapes():
    agent = DQNAgent(87, 22, small_cfg())
    assert agent.q_values(np.zeros(87)).shape == (22,)
    assert agent.q_values(np.zeros((5, 87))).shape == (5, 22)
    action = agent.act(np.zeros(87, dtype=np.float32))
    assert isinstance(action, int) and 0 <= action < 22


def test_name_follows_double():
    assert DQNAgent(87, 22, small_cfg(double=True)).name == "ddqn"
    assert DQNAgent(87, 22, small_cfg(double=False)).name == "dqn"


def test_greedy_picks_the_best_score_and_never_explores():
    """Even with epsilon = 1 (always explore), greedy=True always takes the highest score."""
    always_explore = LinearEpsilon(1.0, 1.0, 1)
    agent = DQNAgent(87, 22, small_cfg(), epsilon_schedule=always_explore)
    state = np.random.default_rng(0).random(87).astype(np.float32)
    best = int(np.argmax(agent.q_values(state)))
    assert {agent.act(state, greedy=True) for _ in range(200)} == {best}


def test_epsilon_1_explores_every_action():
    agent = DQNAgent(87, 22, small_cfg(), epsilon_schedule=LinearEpsilon(1.0, 1.0, 1))
    picks = {agent.act(np.zeros(87, dtype=np.float32)) for _ in range(2000)}
    assert picks == set(range(22))


def test_epsilon_follows_the_schedule_and_is_0_without_one():
    agent = DQNAgent(87, 22, small_cfg(), buffer=filled_buffer(0), epsilon_schedule=LinearEpsilon(1.0, 0.0, 10))
    assert agent.epsilon == 1.0
    for _ in range(5):
        agent.observe(np.zeros(87), 0, 0.0, np.zeros(87), False)
    assert agent.steps == 5 and agent.epsilon == pytest.approx(0.5)
    assert DQNAgent(87, 22, small_cfg()).epsilon == 0.0


def test_observe_stores_the_turn():
    buf = ReplayBuffer(10, 87)
    agent = DQNAgent(87, 22, small_cfg(), buffer=buf)
    agent.observe(np.ones(87), 3, 0.5, np.zeros(87), True)
    assert len(buf) == 1 and agent.steps == 1
    batch = buf.sample(1)
    assert batch.actions[0] == 3 and batch.rewards[0] == 0.5 and batch.dones[0] == 1.0


def test_same_seed_same_agent():
    a, b = DQNAgent(87, 22, small_cfg(), seed=4), DQNAgent(87, 22, small_cfg(), seed=4)
    assert same_weights(a.online, b.online)
    c = DQNAgent(87, 22, small_cfg(), seed=5)
    assert not same_weights(a.online, c.online)


# --- the learning target: DQN vs DDQN, worked out by hand ---------------------------------------

def hand_set_agent(double):
    """2 numbers in, 2 actions out, no hidden layer, weights chosen by hand.

    For the next state s' = [1, 0]:
        online scores = [1, 2]  -> the online net's best next action is 1
        target scores = [5, 3]  -> the target net's best score is 5 (action 0)
    """
    agent = DQNAgent(2, 2, small_cfg(hidden=[], double=double))
    with torch.no_grad():
        agent.online.layers[0].weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
        agent.online.layers[0].bias.zero_()
        agent.target.layers[0].weight.copy_(torch.tensor([[5.0, 0.0], [3.0, 0.0]]))
        agent.target.layers[0].bias.zero_()
    return agent


def test_dqn_and_ddqn_targets_by_hand():
    r, s2 = torch.tensor([1.0]), torch.tensor([[1.0, 0.0]])
    not_done, done = torch.tensor([0.0]), torch.tensor([1.0])
    gamma = small_cfg()["gamma"]  # 0.9

    # DQN: the target net picks and scores the next action -> max(5, 3) = 5
    y = hand_set_agent(double=False).td_target(r, s2, not_done)
    assert y.item() == pytest.approx(1 + gamma * 5)  # 5.5

    # DDQN: the online net picks action 1, the target net scores it -> 3
    y = hand_set_agent(double=True).td_target(r, s2, not_done)
    assert y.item() == pytest.approx(1 + gamma * 3)  # 3.7

    # Last turn of an episode: no next state, the target is just the reward
    for double in (False, True):
        assert hand_set_agent(double).td_target(r, s2, done).item() == pytest.approx(1.0)


# --- learning ------------------------------------------------------------------------------------

def test_no_learning_before_learning_starts():
    buf = ReplayBuffer(100, 87)
    agent = DQNAgent(87, 22, small_cfg(learning_starts=10), buffer=buf)
    for i in range(10):
        assert agent.learn() is None
        agent.observe(np.zeros(87), 0, 0.0, np.zeros(87), False)
    assert isinstance(agent.learn(), float)
    assert agent.updates == 1
    assert DQNAgent(87, 22, small_cfg()).learn() is None  # no buffer at all


def test_target_net_is_copied_every_target_sync_steps():
    agent = DQNAgent(87, 22, small_cfg(target_sync=3), buffer=filled_buffer())
    assert same_weights(agent.online, agent.target)
    agent.learn()
    agent.learn()
    assert not same_weights(agent.online, agent.target)  # online moved, target still old
    agent.learn()
    assert same_weights(agent.online, agent.target)      # 3rd gradient step: copied
    agent.learn()
    assert not same_weights(agent.online, agent.target)


def test_loss_falls_on_a_fixed_batch():
    """The same 8 finished turns over and over: the network must learn to predict their rewards."""
    buf = ReplayBuffer(8, 87, seed=0)
    rng = np.random.default_rng(1)
    for _ in range(8):
        buf.push(rng.random(87), int(rng.integers(22)), rng.random(), rng.random(87), 1.0)
    agent = DQNAgent(87, 22, small_cfg(), buffer=buf)
    losses = [agent.learn() for _ in range(300)]
    assert np.mean(losses[-20:]) < 0.1 * np.mean(losses[:20])


def test_learn_only_changes_the_online_net():
    agent = DQNAgent(87, 22, small_cfg(), buffer=filled_buffer())
    before = {k: v.clone() for k, v in agent.target.state_dict().items()}
    agent.learn()
    assert all(torch.equal(before[k], v) for k, v in agent.target.state_dict().items())
    assert not same_weights(agent.online, agent.target)


# --- saving and loading --------------------------------------------------------------------------

def test_save_then_load_gives_the_same_agent(tmp_path):
    agent = DQNAgent(87, 22, small_cfg(double=False), buffer=filled_buffer(), seed=3)
    for _ in range(5):
        agent.observe(np.zeros(87), 1, 0.1, np.zeros(87), False)
        agent.learn()
    agent.save(tmp_path / "agent.pt")

    loaded = DQNAgent.load(tmp_path / "agent.pt")  # no cfg, buffer or schedule needed
    states = np.random.default_rng(0).random((4, 87)).astype(np.float32)
    assert np.array_equal(loaded.q_values(states), agent.q_values(states))
    assert same_weights(loaded.target, agent.target)
    assert (loaded.name, loaded.steps, loaded.updates, loaded.seed) == ("dqn", agent.steps, agent.updates, 3)
    assert loaded.epsilon == 0.0  # a loaded agent only plays


def test_loaded_agent_can_keep_learning(tmp_path):
    agent = DQNAgent(87, 22, small_cfg(), buffer=filled_buffer())
    agent.learn()
    agent.save(tmp_path / "agent.pt")
    loaded = DQNAgent.load(tmp_path / "agent.pt", buffer=filled_buffer(), epsilon_schedule=LinearEpsilon(1, 0.05, 10))
    assert isinstance(loaded.learn(), float) and loaded.updates == 2


# --- with Jana's training and evaluation scripts -------------------------------------------------

def test_train_and_evaluate_with_the_real_agent(tmp_path):
    cfg = small_cfg(n_episodes=6, eval_every=3, eval_episodes=5, buffer_capacity=200, learning_starts=20,
                    eps_decay_steps=50)
    result = train.train(cfg, tmp_path / "train", DQNAgent, verbose=False)
    assert (tmp_path / "train" / "best.pt").exists() and (tmp_path / "train" / "ddqn_summary.json").exists()
    with open(tmp_path / "train" / "steps.csv") as f:
        assert any(line.rstrip().split(",")[-1] for line in f.readlines()[1:])  # some loss was logged
    assert result["steps"] > cfg["learning_starts"]

    evaluate.main(["--agents", "dqn", "--checkpoint", str(tmp_path / "train" / "best.pt"), "--episodes", "4",
                   "--long-seeds", "0", "--out", str(tmp_path / "eval")])
    summary = json.loads((tmp_path / "eval" / "ddqn_summary.json").read_text())
    assert summary["n_episodes"] == 4 and summary["budget"] == 40
