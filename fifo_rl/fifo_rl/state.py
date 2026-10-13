"""State encoder: turns "what is covered so far" into the vector the agent reads.

Layout of the vector (87 numbers with the defaults):
    [0:64]   checklist  - 1.0 if that bin was hit in any run of this episode, else 0.0
    [64]     budget     - runs_left / budget (1.0 at the start, 0.0 when no runs are left)
    [65:87]  last move  - one-hot of the last action (all zeros before the first run)

Reward is not computed here: update() only reports how many bins are new.
"""

import numpy as np

from fifo_rl.coverage_model import N_BINS


class StateEncoder:
    def __init__(self, budget, n_actions=22, n_bins=N_BINS, include_last_action=True):
        if not isinstance(budget, (int, np.integer)) or budget < 1:
            raise ValueError(f"budget must be a whole number of runs, at least 1, got {budget}")
        if n_actions < 1:
            raise ValueError(f"n_actions must be at least 1, got {n_actions}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be at least 1, got {n_bins}")
        self.budget = budget
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.include_last_action = include_last_action
        self.reset()

    @property
    def size(self):
        """Length of the state vector."""
        return self.n_bins + 1 + (self.n_actions if self.include_last_action else 0)

    @property
    def checklist(self):
        """Copy of the bins hit so far (bool array of n_bins)."""
        return self._checklist.copy()

    @property
    def runs_left(self):
        return self._runs_left

    @property
    def all_covered(self):
        return bool(self._checklist.all())

    def reset(self):
        """Start a new episode: nothing covered, full budget, no last action."""
        self._checklist = np.zeros(self.n_bins, dtype=bool)
        self._runs_left = self.budget
        self._last_action = None
        return self.encode()

    def update(self, run_hits, action_id):
        """Record one finished run.

        run_hits:  n_bins flags, True where that run hit the bin.
        action_id: the action that produced the run (0 .. n_actions-1).
        Returns (state, n_new), n_new = bins hit for the first time in this episode.
        """
        if self._runs_left == 0:
            raise ValueError("budget is used up; call reset() to start a new episode")
        run_hits = np.asarray(run_hits)
        if run_hits.shape != (self.n_bins,):
            raise ValueError(f"run_hits must have shape ({self.n_bins},), got {run_hits.shape}")
        if not 0 <= action_id < self.n_actions:
            raise ValueError(f"action_id must be in 0..{self.n_actions - 1}, got {action_id}")

        run_hits = run_hits.astype(bool)
        n_new = int((run_hits & ~self._checklist).sum())
        self._checklist |= run_hits
        self._runs_left -= 1
        self._last_action = action_id
        return self.encode(), n_new

    def encode(self):
        """Current state as a float32 vector; every value is between 0 and 1."""
        state = np.zeros(self.size, dtype=np.float32)
        state[: self.n_bins] = self._checklist
        state[self.n_bins] = self._runs_left / self.budget
        if self.include_last_action and self._last_action is not None:
            state[self.n_bins + 1 + self._last_action] = 1.0
        return state
