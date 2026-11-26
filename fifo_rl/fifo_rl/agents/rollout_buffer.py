"""Contract F: the on-policy memory the A2C agent learns from, and the advantage maths.

    buf = RolloutBuffer()
    buf.add(s, a, r, s2, done)          # the same five things CampaignRunner passes to observe()
    buf.n_tests                         # tests finished inside the buffer (one per END_TEST)
    buf.ended                           # True once a transition with done=True was added
    rollout = buf.arrays()              # Rollout of numpy arrays, see below
    buf.clear()                         # after every update: old tests are never reused

Unlike the DQN's ReplayBuffer (thousands of old transitions, sampled at random), this buffer
holds only the transitions since the last update, all chosen by the CURRENT policy. A2C learns
from them once and throws them away: that is what "on-policy" means.

compute_gae turns rewards and the critic's guesses into the two numbers A2C learns from:
    delta_t = r_t + gamma * V(s2_t) * (1 - done_t) - V(s_t)       surprise at step t
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}      advantage (GAE), walked backwards
    R_t     = A_t + V(s_t)                                        return = target for the critic
lam = 1 gives the plain discounted return minus V(s_t); lam = 0 gives just delta_t (one-step TD).
done = 1 marks the end of a campaign, so nothing flows from one campaign into the previous one.
"""

from typing import NamedTuple

import numpy as np

from fifo_rl.plan.actions import END_TEST_ID


class Rollout(NamedTuple):
    """T transitions in the order they happened, one row per transition."""
    states: np.ndarray       # float32 (T, state_dim)
    actions: np.ndarray      # int64   (T,)
    rewards: np.ndarray      # float32 (T,)
    next_states: np.ndarray  # float32 (T, state_dim)
    dones: np.ndarray        # float32 (T,)  1.0 = the campaign ended after this transition


class RolloutBuffer:
    def __init__(self, end_action=END_TEST_ID):
        """Input:  end_action, the action id that closes a test (END_TEST = 8); every transition
                with this action counts as one finished test.
        Output: nothing; an empty buffer.
        """
        self.end_action = end_action
        self.clear()

    def clear(self):
        """Input: nothing. Output: nothing. Logic: forget every transition (called after each update)."""
        self._states, self._actions, self._rewards, self._next_states, self._dones = [], [], [], [], []
        self._n_tests = 0
        self._ended = False

    def __len__(self):
        return len(self._actions)

    @property
    def n_tests(self):
        """Input: nothing. Output: how many tests finished inside the buffer (END_TEST transitions)."""
        return self._n_tests

    @property
    def ended(self):
        """Input: nothing. Output: True once a transition with done=True was added (campaign over)."""
        return self._ended

    def add(self, s, a, r, s2, done):
        """Remember one transition.

        Input:  state, action id, reward, next state, done flag (as given to agent.observe()).
        Output: nothing.
        Logic:  keep copies (the caller may reuse its arrays); an END_TEST action counts one more
                finished test; done=True marks the campaign as over.
        """
        self._states.append(np.array(s, dtype=np.float32))
        self._actions.append(int(a))
        self._rewards.append(float(r))
        self._next_states.append(np.array(s2, dtype=np.float32))
        self._dones.append(float(bool(done)))
        if int(a) == self.end_action:
            self._n_tests += 1
        if done:
            self._ended = True

    def arrays(self):
        """Input: nothing. Output: a Rollout of numpy arrays (copies) in the order added.
        Logic:  stack the lists; an empty buffer is an error (there is nothing to learn from).
        """
        if not self._actions:
            raise ValueError("the rollout buffer is empty")
        return Rollout(
            states=np.stack(self._states),
            actions=np.array(self._actions, dtype=np.int64),
            rewards=np.array(self._rewards, dtype=np.float32),
            next_states=np.stack(self._next_states),
            dones=np.array(self._dones, dtype=np.float32),
        )


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """Advantages and returns of one rollout (formulas at the top of this file).

    Input:  rewards, values = V(s_t), next_values = V(s2_t), dones - four 1-D arrays of the same
            length T, in the order the transitions happened; gamma and lam, both in [0, 1].
    Output: (advantages, returns), two float32 arrays of length T.
    Logic:  walk backwards from the last transition, carrying the running advantage; done = 1
            stops both the next value and the carried advantage, so campaigns stay separate.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if not rewards.shape == values.shape == next_values.shape == dones.shape or rewards.ndim != 1:
        raise ValueError(f"rewards, values, next_values and dones must be 1-D of the same length, got "
                         f"{rewards.shape}, {values.shape}, {next_values.shape}, {dones.shape}")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= lam <= 1.0):
        raise ValueError(f"gamma and lam must be in [0, 1], got gamma={gamma}, lam={lam}")
    advantages = np.zeros_like(rewards)
    carried = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        keep = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * keep - values[t]
        carried = delta + gamma * lam * keep * carried
        advantages[t] = carried
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)
