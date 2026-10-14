"""The two baseline agents the DQN must beat.

Every agent (these two and the DQN later) has the same interface:
    agent.name          short string used in file names and plots
    agent.act(state)    -> action id (int) for the next simulation run

    RandomAgent        picks one of the 22 actions uniformly at random every run
                       = "an engineer who picks test scenarios without looking at coverage"
    FixedActionAgent   always picks the same action; by default baseline_random (id 21),
                       the verification plan's FIFO_4 recipe (wr 70 / rd 30 / rst 10)
                       = "plain constrained-random verification, one short run at a time"

Neither agent looks at the state: that is exactly what makes them baselines.
"""

import numpy as np

from fifo_rl.actions import ACTION_INDEX, N_ACTIONS, get_action


class RandomAgent:
    name = "random"

    def __init__(self, n_actions=N_ACTIONS, seed=0):
        """Input:  n_actions, how many actions to choose from (22);
                seed, seed of the agent's own random generator.
        Output: nothing.
        Logic:  the agent has its own generator, separate from the env's per-run seeds, so
                the choice of action and the randomness inside a run never share numbers.
        """
        if n_actions < 1:
            raise ValueError(f"n_actions must be at least 1, got {n_actions}")
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        """Input:  state (ignored).
        Output: an action id in 0 .. n_actions-1, each equally likely.
        """
        return int(self.rng.integers(self.n_actions))


class FixedActionAgent:
    def __init__(self, action_id=ACTION_INDEX["baseline_random"]):
        """Input:  action_id, the action to repeat every run (default 21 = baseline_random).
        Output: nothing.
        Logic:  check the id exists (get_action raises otherwise) and name the agent after
                the action, e.g. "fixed_baseline_random".
        """
        self.action_id = get_action(action_id).id
        self.name = f"fixed_{get_action(action_id).name}"

    def act(self, state):
        """Input: state (ignored). Output: always the same action id."""
        return self.action_id
