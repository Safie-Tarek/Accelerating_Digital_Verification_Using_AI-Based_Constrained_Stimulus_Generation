"""The DQN / DDQN agent (Contract E, docs/fifo_team_plan.md §6.2).

    agent = DQNAgent(state_dim=87, n_actions=22, cfg=cfg, buffer=buffer, epsilon_schedule=schedule, seed=0)
    action = agent.act(state)                   # explore with chance epsilon, else the best-scored action
    agent.observe(state, action, reward, next_state, done)
    loss = agent.learn()                        # one gradient step (None until learning_starts)
    agent.save("best.pt");  agent = DQNAgent.load("best.pt")

Two copies of the same QNetwork:
    online   learns every step; used to choose actions
    target   a frozen copy, refreshed every target_sync gradient steps; used to build the learning
             target, so the network does not chase a goal that moves with every step

Learning target for one remembered turn (s, a, r, s', done):
    DQN    y = r + gamma * (1 - done) * max_a' target(s')[a']
    DDQN   y = r + gamma * (1 - done) * target(s')[argmax_a' online(s')[a']]
DDQN lets the online net pick the next action and the target net score it; plain DQN lets the
target net do both, which tends to over-estimate scores. cfg["double"] chooses.

Settings come from the config (configs/default.yaml): hidden, lr, gamma, batch_size,
learning_starts, target_sync, grad_clip, double. train.py decides how often learn() is called
(train_freq); this file counts env steps (observe) and gradient steps (learn) itself.
"""

import numpy as np
import torch
from torch import nn

from fifo_rl.agents.networks import QNetwork


class DQNAgent:
    def __init__(self, state_dim, n_actions, cfg, buffer=None, epsilon_schedule=None, seed=0):
        """Set up the agent.

        Input:  state_dim, length of a state (87); n_actions, number of actions (22);
                cfg, the config dict; buffer, a ReplayBuffer (None = cannot learn, e.g. evaluation);
                epsilon_schedule, a schedule from schedules.py (None = never explore);
                seed, makes the network weights and the exploration repeatable.
        Output: nothing.
        Logic:  build the online net, copy it into the target net, create Adam and the Huber loss,
                and start both counters at 0.
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = dict(cfg)
        self.buffer = buffer
        self.epsilon_schedule = epsilon_schedule
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # exploration draws
        torch.manual_seed(seed)                  # initial weights
        self.online = QNetwork(state_dim, self.cfg["hidden"], n_actions)
        self.target = QNetwork(state_dim, self.cfg["hidden"], n_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg["lr"])
        self.loss_fn = nn.SmoothL1Loss()  # Huber: like squared error for small errors, linear for big ones
        self.steps = 0    # turns seen by observe()
        self.updates = 0  # gradient steps done by learn()

    @property
    def name(self):
        """Input: nothing. Output: "ddqn" or "dqn" (used in file names and plots)."""
        return "ddqn" if self.cfg["double"] else "dqn"

    @property
    def epsilon(self):
        """Chance of a random action now.

        Input:  nothing. Output: a float 0..1.
        Logic:  the schedule's value at the current step count; 0.0 without a schedule
                (a loaded agent used only for evaluation never explores).
        """
        if self.epsilon_schedule is None:
            return 0.0
        return float(self.epsilon_schedule.value(self.steps))

    def q_values(self, states):
        """Scores of every action.

        Input:  states, an array (B, state_dim) or one state (state_dim,).
        Output: a float numpy array (B, n_actions) or (n_actions,).
        Logic:  run the online net without recording gradients.
        """
        with torch.no_grad():
            return self.online(torch.as_tensor(np.asarray(states), dtype=torch.float32)).numpy()

    def act(self, state, greedy=False):
        """Choose the next action.

        Input:  state, one state vector; greedy, True = never explore (evaluation).
        Output: an action id 0 .. n_actions-1.
        Logic:  with chance epsilon (only when not greedy) a uniformly random action, so the agent
                keeps trying things; otherwise the action with the highest score.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def observe(self, s, a, r, s2, done):
        """Remember one turn.

        Input:  state, action, reward, next state, done flag.
        Output: nothing.
        Logic:  store it in the replay buffer (if there is one) and count the step; the step count
                drives epsilon.
        """
        if self.buffer is not None:
            self.buffer.push(s, a, r, s2, done)
        self.steps += 1

    def td_target(self, rewards, next_states, dones):
        """The learning target y for a batch (see the formulas at the top of this file).

        Input:  rewards (B,), next_states (B, state_dim), dones (B,) - float tensors.
        Output: a tensor (B,) with y = r + gamma * (1 - done) * value of the next state.
        Logic:  DDQN: the online net picks the best next action, the target net gives its score.
                DQN: the target net's highest score. done = 1 means no next state, so only r.
        """
        with torch.no_grad():
            next_target = self.target(next_states)
            if self.cfg["double"]:
                best = self.online(next_states).argmax(dim=1, keepdim=True)
                next_value = next_target.gather(1, best).squeeze(1)
            else:
                next_value = next_target.max(dim=1).values
            return rewards + self.cfg["gamma"] * (1.0 - dones) * next_value

    def learn(self):
        """One gradient step on a random batch of remembered turns.

        Input:  nothing (uses the buffer).
        Output: the loss (float), or None when there is nothing to learn from yet.
        Logic:  1. wait until the buffer holds learning_starts turns (and at least one batch)
                2. sample batch_size turns; score of the action that was taken: online(s)[a]
                3. target y from td_target(); Huber loss between the two
                4. backpropagate, cut the gradient norm to grad_clip, Adam step
                5. every target_sync gradient steps copy online -> target
        """
        if self.buffer is None or len(self.buffer) < max(self.cfg["learning_starts"], self.cfg["batch_size"]):
            return None
        batch = self.buffer.sample(self.cfg["batch_size"])
        states = torch.as_tensor(batch.states)
        actions = torch.as_tensor(batch.actions)
        rewards = torch.as_tensor(batch.rewards)
        next_states = torch.as_tensor(batch.next_states)
        dones = torch.as_tensor(batch.dones)

        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target = self.td_target(rewards, next_states, dones)
        loss = self.loss_fn(predicted, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg["grad_clip"])
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg["target_sync"] == 0:
            self.sync_target()
        return float(loss.item())

    def sync_target(self):
        """Input: nothing. Output: nothing. Logic: copy the online weights into the target net."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path):
        """Write a checkpoint.

        Input:  path of the .pt file.
        Output: nothing.
        Logic:  one dict with everything needed to rebuild the agent without the config file:
                cfg, sizes, seed, both counters, both networks and the optimizer state.
        """
        torch.save({
            "agent": self.name,
            "cfg": self.cfg,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "steps": self.steps,
            "updates": self.updates,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, buffer=None, epsilon_schedule=None):
        """Rebuild an agent from a checkpoint.

        Input:  path of a file written by save(); optionally a buffer and a schedule to go on
                training (without them the agent can only play, with epsilon 0).
        Output: a DQNAgent with the saved weights and counters.
        Logic:  read the dict on the CPU (weights_only: tensors and plain values only, nothing
                else is executed), build the agent from its cfg and sizes, then load the states.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(data["state_dim"], data["n_actions"], data["cfg"], buffer=buffer,
                    epsilon_schedule=epsilon_schedule, seed=data["seed"])
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.steps = data["steps"]
        agent.updates = data["updates"]
        return agent
