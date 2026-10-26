"""The neural network inside the DQN agent (Contract E, docs/fifo_team_plan.md §6.2).

    net = QNetwork(state_dim=87, hidden=[128, 128], n_actions=22)
    scores = net(states)        # (B, 87) float32 -> (B, 22): one score ("Q-value") per action

A Q-value is the network's guess of how much reward the episode will still bring if the agent
takes that action now and plays well afterwards. The agent picks the action with the highest one.

The layer sizes come from the arguments (cfg["hidden"] and the env's sizes), never written here,
so a new state or action design only changes the numbers passed in.
"""

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden, n_actions):
        """Build the layers.

        Input:  state_dim, length of a state vector (87);
                hidden, list of hidden layer sizes (cfg["hidden"], e.g. [128, 128]);
                n_actions, number of actions (22).
        Output: nothing.
        Logic:  one Linear + ReLU per hidden size, then a last Linear to n_actions with no
                activation (Q-values can be any number, also negative).
                [128, 128] gives 87 -> 128 -> ReLU -> 128 -> ReLU -> 22.
        """
        super().__init__()
        if state_dim < 1 or n_actions < 1:
            raise ValueError(f"state_dim and n_actions must be at least 1, got {state_dim} and {n_actions}")
        if any(size < 1 for size in hidden):
            raise ValueError(f"hidden sizes must be at least 1, got {list(hidden)}")
        layers = []
        size_in = state_dim
        for size in hidden:
            layers += [nn.Linear(size_in, size), nn.ReLU()]
            size_in = size
        layers.append(nn.Linear(size_in, n_actions))
        self.layers = nn.Sequential(*layers)
        self.state_dim = state_dim
        self.n_actions = n_actions

    def forward(self, states):
        """Score every action.

        Input:  states, a float tensor (B, state_dim) - or (state_dim,) for a single state.
        Output: a float tensor (B, n_actions) - or (n_actions,) for a single state.
        Logic:  pass the states through the layers.
        """
        return self.layers(states)
