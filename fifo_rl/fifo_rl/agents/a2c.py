"""The A2C (advantage actor-critic) agent (Contract F).

    agent = A2CAgent(state_dim=93, n_actions=9, cfg=cfg, seed=0)
    action = agent.act(state)                  # roll the policy's dice (greedy=True: most likely block)
    agent.observe(state, action, reward, next_state, done)
    loss = agent.learn()                       # one update after whole tests, else None
    agent.save("best.pt");  agent = A2CAgent.load("best.pt")

Two networks, both the plain MLP of networks.py (QNetwork):
    actor    state -> one score (logit) per block; softmax of the legal ones = the policy pi(a|s)
    critic   state -> one number V(s): how much reward the rest of the campaign should still bring

Legality mask: when the plan is full (state slot POSITION_SLOT = 1.0), PlanBuilder adds END_TEST
without asking the agent. The mask allows only END_TEST in such a state, so pi(END_TEST) = 1 and
log pi = 0: the actor is never pushed by a choice it did not make. No other rule is masked (reading
an empty FIFO is exactly what underflow needs), the same freedom the DQN has.

When it learns: CampaignRunner simulates a whole test first, then calls observe() + learn() for
every block of it. learn() waits until the rollout holds a2c_rollout_tests whole tests (each ends
with END_TEST) or the campaign ended, so an update never falls in the middle of a test: every
transition it learns from was chosen by the current policy (on-policy), and the runner needs no
change.

One update (learn):
    V(s), V(s2) from the critic -> compute_gae -> advantages A, returns R
    policy_loss = -mean(log pi(a|s) * A)        blocks that did better than expected get likelier
    value_loss  = mean((R - V(s))^2)            the critic's guesses move towards what happened
    entropy     = mean(entropy of pi(.|s))      kept high so the agent keeps trying blocks
    loss        = policy_loss + vf_coef * value_loss - ent_coef * entropy
    backpropagate, cut the gradient norm to a2c_grad_clip, one Adam step, empty the rollout

Settings come from the config: a2c_hidden, a2c_lr, gamma, gae_lambda, vf_coef, ent_coef,
a2c_grad_clip, a2c_rollout_tests, a2c_normalize_adv.
"""

import numpy as np
import torch
from torch import nn

from fifo_rl.agents.networks import QNetwork
from fifo_rl.agents.rollout_buffer import RolloutBuffer, compute_gae
from fifo_rl.plan.actions import END_TEST_ID

# fifo_rl/plan/state_encoder.py puts "blocks in this plan / plan_max_len" at state[-2]
POSITION_SLOT = -2
FULL_PLAN = 1.0 - 1e-6     # a position at or above this means the plan holds plan_max_len blocks
MASKED_LOGIT = -1e9        # score given to an illegal block: its probability becomes exactly 0
MIN_NORMALIZE = 8          # advantages are normalized only in rollouts of at least this many transitions


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

    def observe(self, s, a, r, s2, done):
        """Remember one transition.

        Input:  state, block id, reward, next state, done flag (what CampaignRunner passes).
        Output: nothing.
        Logic:  add it to the rollout and count the step.
        """
        self.rollout.add(s, a, r, s2, done)
        self.steps += 1

    def ready(self):
        """Input: nothing. Output: True when learn() would update now.
        Logic:  the rollout holds a2c_rollout_tests whole tests, or the campaign ended (done).
        """
        return len(self.rollout) > 0 and (self.rollout.n_tests >= self.cfg["a2c_rollout_tests"] or self.rollout.ended)

    def learn(self):
        """One A2C update on the whole rollout (formulas at the top of this file).

        Input:  nothing (uses the rollout).
        Output: the total loss (float), or None when the rollout is not ready yet.
        Logic:  1. wait for whole tests (ready())
                2. critic: V(s) with gradients, V(s2) without -> compute_gae -> A, R
                3. normalize A (mean 0, std 1) only if a2c_normalize_adv and the rollout is big
                   enough: one or two transitions would give NaN or noise
                4. masked policy -> log pi(a|s), entropy; the three losses -> the total loss
                5. backpropagate, cut the gradient norm, one Adam step
                6. keep the numbers in last_stats, empty the rollout
        """
        if not self.ready():
            return None
        batch = self.rollout.arrays()
        states = torch.as_tensor(batch.states)
        actions = torch.as_tensor(batch.actions)

        values = self.critic(states).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic(torch.as_tensor(batch.next_states)).squeeze(-1)
        advantages, returns = compute_gae(batch.rewards, values.detach().numpy(), next_values.numpy(), batch.dones,
                                          self.cfg["gamma"], self.cfg["gae_lambda"])
        advantages = torch.as_tensor(advantages)
        returns = torch.as_tensor(returns)
        if self.cfg["a2c_normalize_adv"] and len(advantages) >= MIN_NORMALIZE:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy = torch.distributions.Categorical(logits=self.masked_logits(states))
        log_probs = policy.log_prob(actions)
        entropy = policy.entropy().mean()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = ((returns - values) ** 2).mean()
        loss = policy_loss + self.cfg["vf_coef"] * value_loss - self.cfg["ent_coef"] * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                 self.cfg["a2c_grad_clip"])
        self.optimizer.step()

        self.updates += 1
        self.last_stats = {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()),
                           "entropy": float(entropy.item()), "n_transitions": len(batch.actions)}
        self.rollout.clear()
        return float(loss.item())

    def save(self, path):
        """Write a checkpoint.

        Input:  path of the .pt file.
        Output: nothing.
        Logic:  one dict with everything needed to rebuild the agent without the config file:
                cfg, sizes, seed, both counters, both networks and the optimizer state (Adam's
                momentum, so a loaded agent keeps learning exactly where it stopped). The rollout
                is not saved: after every update it is empty, and a loaded agent starts a new one.
        """
        torch.save({
            "agent": self.name,
            "cfg": self.cfg,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "steps": self.steps,
            "updates": self.updates,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        """Rebuild an agent from a checkpoint.

        Input:  path of a file written by save().
        Output: an A2CAgent with the saved weights, optimizer state and counters.
        Logic:  read the dict on the CPU (weights_only: tensors and plain values only, nothing
                else is executed); refuse a checkpoint of another agent (e.g. a DQN best.pt);
                build the agent from its cfg and sizes, then load the states.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("agent") != "a2c":
            raise ValueError(f"{path} is a {data.get('agent')!r} checkpoint, not an A2C one")
        agent = cls(data["state_dim"], data["n_actions"], data["cfg"], seed=data["seed"])
        agent.actor.load_state_dict(data["actor"])
        agent.critic.load_state_dict(data["critic"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.steps = data["steps"]
        agent.updates = data["updates"]
        return agent
