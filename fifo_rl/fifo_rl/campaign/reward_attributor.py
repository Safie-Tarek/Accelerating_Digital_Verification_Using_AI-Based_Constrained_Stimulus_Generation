"""Splits the result of one test into one reward per planned action.

Step 1 (this file): Questa gives ONE coverage report per test, so nobody knows which block hit
which bin. The coverage reward of the whole test goes to the plan's last action (END_TEST):
    reward[k]   = -cycle_cost * cycles_k * scale                   every planned action k
    reward[END] += reward.compute_reward(new bins of the test, closed_now)
compute_reward charges step_cost once, so one TEST costs one bin, exactly like one run did in the
22-action design; with cycle_cost = 0 (default) the agent is judged on the number of tests. DQN
passes the END_TEST reward back to the blocks through the next states (raise gamma for that).

Step 2 (later): a per-action attributor reads the act_<idx> covergroup instances and gives each
block the bins it was first to hit. It returns the same AttributionResult, so nothing else changes.
"""

from dataclasses import dataclass

import numpy as np

from fifo_rl.reward import RewardConfig, compute_reward


@dataclass(frozen=True, eq=False)
class AttributionResult:
    rewards: tuple          # one float per planned action, in plan order (END_TEST included)
    test_hits: np.ndarray   # bool[64]: bins the whole test hit
    new_bins: int           # bins hit for the first time in the campaign by this test
    closed_now: bool        # this test brought the campaign to all bins
    mode: str               # "test" (step 1) or "action" (step 2)


class RewardAttributor:
    mode = "test"

    def __init__(self, reward=RewardConfig(), cycle_cost=0.0):
        """Input:  reward, the RewardConfig (step_cost per test, closure_bonus, scale);
                cycle_cost, bins charged per simulated cycle of a block (>= 0).
        Output: nothing.
        """
        if cycle_cost < 0:
            raise ValueError(f"cycle_cost must be >= 0, got {cycle_cost}")
        self.reward = reward
        self.cycle_cost = cycle_cost

    def attribute(self, plan, test_hits, covered_before):
        """Input:  plan, the closed StimulusPlan that was simulated;
                test_hits, bool[64] from the test's coverage report;
                covered_before, bool[64] covered by the campaign before this test.
        Output: an AttributionResult.
        Logic:  new = test_hits AND NOT covered_before; closed_now = everything covered after the
                test and this test added something; every action pays its cycles; END_TEST (the
                last action) also gets the test's coverage reward.
        """
        if not plan.is_closed:
            raise ValueError("only a closed plan (ending with END_TEST) can be rewarded")
        test_hits = np.asarray(test_hits, dtype=bool)
        covered_before = np.asarray(covered_before, dtype=bool)
        if test_hits.shape != covered_before.shape:
            raise ValueError(f"test_hits {test_hits.shape} and covered_before {covered_before.shape} differ")
        new_bins = int((test_hits & ~covered_before).sum())
        closed_now = bool((test_hits | covered_before).all()) and new_bins > 0
        rewards = [-self.cycle_cost * pa.spec.n_cycles * self.reward.scale for pa in plan.actions]
        rewards[-1] += compute_reward(new_bins, closed_now, self.reward)
        return AttributionResult(tuple(float(r) for r in rewards), test_hits, new_bins, closed_now, self.mode)
