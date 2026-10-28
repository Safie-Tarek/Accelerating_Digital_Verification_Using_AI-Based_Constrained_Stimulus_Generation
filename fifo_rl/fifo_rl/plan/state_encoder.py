"""State encoder of the plan-based design: what the agent sees before choosing each block.

Layout (93 numbers with the defaults: 64 bins, 3 blocks of history, 9 actions):
    [0:64]   covered    1.0 if the bin was hit in any earlier test of this campaign
    [64:91]  history    the last 3 blocks of THIS plan, one-hot 9 each, newest first
                        (all zeros where the plan is still shorter than 3)
    [91]     position   blocks already in this plan / plan_max_len
    [92]     tests left tests still allowed in the campaign / campaign budget

The history is there because the FIFO keeps its fill level between blocks: "WRITE, then ?" is a
different situation from "?" at the start of a test. Unlike fifo_rl/state.py this encoder keeps
no memory of its own: the campaign's coverage lives in CampaignStore, the plan in StimulusPlan.
"""

import numpy as np

from fifo_rl.coverage_model import N_BINS
from fifo_rl.plan.actions import N_ACTIONS


class StateEncoder:
    def __init__(self, plan_max_len, n_history=3, n_actions=N_ACTIONS, n_bins=N_BINS):
        """Input:  plan_max_len, blocks per plan (>= 1); n_history, how many last blocks to show
                (>= 0); n_actions, size of each one-hot (9); n_bins, coverage bins (64).
        Output: nothing.
        """
        if plan_max_len < 1:
            raise ValueError(f"plan_max_len must be at least 1, got {plan_max_len}")
        if n_history < 0:
            raise ValueError(f"n_history must be >= 0, got {n_history}")
        self.plan_max_len = plan_max_len
        self.n_history = n_history
        self.n_actions = n_actions
        self.n_bins = n_bins

    @property
    def size(self):
        """Input: nothing. Output: length of the state vector (64 + 3*9 + 2 = 93 by default)."""
        return self.n_bins + self.n_history * self.n_actions + 2

    def encode(self, store, plan=None):
        """The state before the next choice.

        Input:  store, the CampaignStore (covered, tests_left, budget);
                plan, the StimulusPlan built so far in this test (None = a test not started yet).
        Output: float32 vector of `size` numbers, all between 0 and 1.
        Logic:  copy the covered bins; set one 1 per remembered block at
                n_bins + j*n_actions + action_id (j = 0 for the newest); write the plan position
                and the share of tests left.
        """
        covered = np.asarray(store.covered, dtype=bool)
        if covered.shape != (self.n_bins,):
            raise ValueError(f"store.covered must have shape ({self.n_bins},), got {covered.shape}")
        state = np.zeros(self.size, dtype=np.float32)
        state[: self.n_bins] = covered
        history = plan.last_actions(self.n_history) if plan is not None else []
        for j, action_id in enumerate(history):
            state[self.n_bins + j * self.n_actions + action_id] = 1.0
        n_blocks = plan.n_blocks if plan is not None else 0
        state[-2] = min(n_blocks / self.plan_max_len, 1.0)
        state[-1] = store.tests_left / store.budget
        return state
