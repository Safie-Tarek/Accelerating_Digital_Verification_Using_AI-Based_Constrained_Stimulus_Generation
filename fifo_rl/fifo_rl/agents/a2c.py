"""The A2C (advantage actor-critic) agent (Contract F).

    agent = A2CAgent(state_dim=93, n_actions=9, cfg=cfg, seed=0)
    action = agent.act(state)                  # roll the policy's dice (greedy=True: most likely block)
    agent.observe(state, action, reward, next_state, done)        # step 3
    loss = agent.learn()                       # step 3: one update after whole tests, else None
    agent.save("best.pt");  agent = A2CAgent.load("best.pt")      # step 4

Two networks, both the plain MLP of networks.py (QNetwork):
    actor    state -> one score (logit) per block; softmax of the legal ones = the policy pi(a|s)
    critic   state -> one number V(s): how much reward the rest of the campaign should still bring

Legality mask: when the plan is full (state slot POSITION_SLOT = 1.0), PlanBuilder adds END_TEST
without asking the agent. The mask allows only END_TEST in such a state, so pi(END_TEST) = 1 and
log pi = 0: the actor is never pushed by a choice it did not make. No other rule is masked (reading
an empty FIFO is exactly what underflow needs), the same freedom the DQN has.

Settings come from the config: a2c_hidden, a2c_lr (and, from step 3 on, gamma, gae_lambda,
vf_coef, ent_coef, a2c_grad_clip, a2c_rollout_tests, a2c_normalize_adv).
"""

import numpy as np
import torch

from fifo_rl.agents.networks import QNetwork
from fifo_rl.agents.rollout_buffer import RolloutBuffer
from fifo_rl.plan.actions import END_TEST_ID

# fifo_rl/plan/state_encoder.py puts "blocks in this plan / plan_max_len" at state[-2]
POSITION_SLOT = -2
FULL_PLAN = 1.0 - 1e-6     # a position at or above this means the plan holds plan_max_len blocks
MASKED_LOGIT = -1e9        # score given to an illegal block: its probability becomes exactly 0


def legal_mask(states, n_actions, end_action=END_TEST_ID):
    """Which blocks may be chosen in each state.

    Input:  states, an array (B, state_dim) or one state (state_dim,); n_actions (9);
            end_action, the id of END_TEST (8).
    Output: a bool array (B, n_actions) or (n_actions,): True = allowed.
    Logic:  every block is allowed, except in a full-plan state (position slot = 1.0), where
            only END_TEST is.
    """
    states = np.asarray(states, dtype=np.float32)
    full = states[..., POSITION_SLOT] >= FULL_PLAN
    mask = np.ones(states.shape[:-1] + (n_actions,), dtype=bool)
    mask[full] = False
    mask[full, end_action] = True
    return mask


class A2CAgent:
    def __init__(self, state_dim, n_actions, cfg, seed=0):
        """Set up the agent.

        Input:  state_dim, length of a state (93); n_actions, number of blocks (9);
                cfg, the config dict (a2c_hidden, a2c_lr, ...); seed, makes the weights and the
                dice rolls repeatable.
        Output: nothing.
        Logic:  build the actor and the critic, one Adam optimizer over both, an empty rollout,
                and start both counters at 0.
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = dict(cfg)
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # dice rolls in act()
        torch.manual_seed(seed)                  # initial weights
        self.actor = QNetwork(state_dim, self.cfg["a2c_hidden"], n_actions)
        self.critic = QNetwork(state_dim, self.cfg["a2c_hidden"], 1)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=self.cfg["a2c_lr"])
        self.rollout = RolloutBuffer(end_action=END_TEST_ID)
        self.steps = 0      # transitions seen by observe()
        self.updates = 0    # updates done by learn()
        self.last_stats = {}

    @property
    def name(self):
        """Input: nothing. Output: "a2c" (used in file names and plots)."""
        return "a2c"

    @property
    def epsilon(self):
        """Input: nothing. Output: always 0.0. A2C explores by rolling its dice, not with epsilon;
        the property exists so the logs of scripts/run_campaign.py work unchanged."""
        return 0.0

    def masked_logits(self, states):
        """The actor's scores with the illegal blocks pushed to MASKED_LOGIT.

        Input:  states, a float tensor (B, state_dim).
        Output: a tensor (B, n_actions); keeps gradients (learn() needs them in step 3).
        """
        mask = torch.as_tensor(legal_mask(states.detach().numpy(), self.n_actions))
        return self.actor(states).masked_fill(~mask, MASKED_LOGIT)

    def probabilities(self, states):
        """The policy pi(a|s).

        Input:  states, an array (B, state_dim) or one state (state_dim,).
        Output: a float numpy array (B, n_actions) or (n_actions,); each row sums to 1, illegal
                blocks get exactly 0.
        """
        states = np.asarray(states, dtype=np.float32)
        single = states.ndim == 1
        with torch.no_grad():
            probs = torch.softmax(self.masked_logits(torch.as_tensor(np.atleast_2d(states))), dim=-1).numpy()
        return probs[0] if single else probs

    def values(self, states):
        """The critic's guess V(s).

        Input:  states, an array (B, state_dim) or one state (state_dim,).
        Output: a float numpy array (B,), or a float for one state.
        """
        states = np.asarray(states, dtype=np.float32)
        with torch.no_grad():
            v = self.critic(torch.as_tensor(np.atleast_2d(states))).squeeze(-1).numpy()
        return float(v[0]) if states.ndim == 1 else v

    def act(self, state, greedy=False):
        """Choose the next block.

        Input:  state, one state vector; greedy, True = always the most likely block (evaluation).
        Output: a block id 0 .. n_actions-1 (never an illegal one).
        Logic:  greedy -> argmax of pi; otherwise draw one block with the probabilities pi, using
                the agent's own seeded generator.
        """
        probs = self.probabilities(state).astype(np.float64)
        if greedy:
            return int(np.argmax(probs))
        return int(self.rng.choice(self.n_actions, p=probs / probs.sum()))
