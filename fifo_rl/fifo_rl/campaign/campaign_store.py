"""The campaign's memory: what is truly covered, how many tests are left, what happened so far.

A campaign = up to `budget` tests (simulations) aiming at 64/64 = one DQN episode. Only commit()
changes the coverage, once per finished test, with the bins read from that test's report; the
StateEncoder and the RewardAttributor only read it.
"""

from dataclasses import dataclass

import numpy as np

from fifo_rl.coverage_model import N_BINS


@dataclass(frozen=True, eq=False)
class TestSummary:
    __test__ = False      # its name starts with "Test": tell pytest it is not a test class
    campaign: int
    test_no: int
    seed: int
    plan: object          # StimulusPlan
    result: object        # RunResult
    attribution: object   # AttributionResult
    output: object        # RunOutput
    total_bins: int       # bins covered by the campaign after this test


class CampaignStore:
    def __init__(self, budget, n_bins=N_BINS):
        """Input: budget, tests per campaign (whole number >= 1); n_bins, coverage bins (64)."""
        if not isinstance(budget, (int, np.integer)) or budget < 1:
            raise ValueError(f"budget must be a whole number of tests, at least 1, got {budget}")
        self.budget = budget
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        """Input: nothing. Output: nothing. Logic: nothing covered, no tests done, empty history."""
        self._covered = np.zeros(self.n_bins, dtype=bool)
        self.tests_done = 0
        self.history = []

    @property
    def covered(self):
        """Input: nothing. Output: copy of the bool[64] covered so far."""
        return self._covered.copy()

    @property
    def n_covered(self):
        """Input: nothing. Output: number of bins covered so far."""
        return int(self._covered.sum())

    @property
    def tests_left(self):
        """Input: nothing. Output: tests still allowed in this campaign."""
        return self.budget - self.tests_done

    def all_covered(self):
        """Input: nothing. Output: True when every bin is covered."""
        return bool(self._covered.all())

    def done(self):
        """Input: nothing. Output: True when the campaign is over (all covered or no tests left)."""
        return self.all_covered() or self.tests_left == 0

    def commit(self, test_hits):
        """Record one finished test.

        Input:  test_hits, bool[64] the test hit.
        Output: number of bins covered after the test.
        Logic:  refuse once the campaign is over; OR the hits in; count the test.
        """
        if self.done():
            raise ValueError("the campaign is over; call reset() to start a new one")
        test_hits = np.asarray(test_hits, dtype=bool)
        if test_hits.shape != (self.n_bins,):
            raise ValueError(f"test_hits must have shape ({self.n_bins},), got {test_hits.shape}")
        self._covered |= test_hits
        self.tests_done += 1
        return self.n_covered

    def record(self, summary):
        """Input: a TestSummary. Output: nothing. Logic: keep it in the history (logs, plots)."""
        self.history.append(summary)
