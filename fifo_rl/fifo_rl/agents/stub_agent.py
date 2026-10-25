"""A stand-in for the DQNAgent with the full Contract E interface, but no learning.

Lets scripts/train.py, scripts/evaluate.py and their tests run before the real agent
(fifo_rl/agents/dqn.py) is merged:
    act(state, greedy=False)   random action (greedy=True too: there is nothing to exploit)
    observe(s, a, r, s2, done) pushes into the buffer and counts steps, like the real agent
    learn()                    always None (= "not learning yet")
    epsilon                    the schedule's value at the current step
    save(path) / load(path)    a small JSON file with the step count and seed
"""

import json
from pathlib import Path

import numpy as np

from fifo_rl.actions import N_ACTIONS


class StubAgent:
    name = "stub"

    def __init__(self, state_dim=87, n_actions=N_ACTIONS, cfg=None, buffer=None, epsilon_schedule=None, seed=0):
        """Input: same arguments as DQNAgent (cfg is ignored). Output: nothing."""
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.buffer = buffer
        self.epsilon_schedule = epsilon_schedule
        self.seed = seed
        self.steps = 0
        self.rng = np.random.default_rng(seed)

    @property
    def epsilon(self):
        return self.epsilon_schedule.value(self.steps) if self.epsilon_schedule is not None else 1.0

    def act(self, state, greedy=False):
        return int(self.rng.integers(self.n_actions))

    def observe(self, s, a, r, s2, done):
        if self.buffer is not None:
            self.buffer.push(s, a, r, s2, done)
        self.steps += 1

    def learn(self):
        return None

    def save(self, path):
        Path(path).write_text(json.dumps({"agent": self.name, "steps": self.steps, "seed": self.seed,
                                          "state_dim": self.state_dim, "n_actions": self.n_actions}))

    @classmethod
    def load(cls, path, buffer=None):
        data = json.loads(Path(path).read_text())
        agent = cls(data["state_dim"], data["n_actions"], buffer=buffer, seed=data["seed"])
        agent.steps = data["steps"]
        return agent
