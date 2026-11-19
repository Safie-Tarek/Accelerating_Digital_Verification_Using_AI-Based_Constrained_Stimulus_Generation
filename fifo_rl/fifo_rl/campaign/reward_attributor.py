"""Splits the result of one test into one reward per planned action.

Step 1 (this file): Questa gives ONE coverage report per test, so nobody knows which block hit
which bin. The coverage reward of the whole test goes to the plan's last action (END_TEST):
    reward[k]   = -cycle_cost * cycles_k * scale                   every planned action k
    reward[END] += reward.compute_reward(new bins of the test, closed_now)
compute_reward charges step_cost once, so one TEST costs one bin, exactly like one run did in the
22-action design; with cycle_cost = 0 (default) the agent is judged on the number of tests. DQN
passes the END_TEST reward back to the blocks through the next states (raise gamma for that).

Step 2 (ActionRewardAttributor): the report also has one act_<idx> instance per block, so each
block gets the bins it was the FIRST to hit (blocks in idx order; bins covered by earlier tests
count for nobody):
    reward[k]   = scale * new_k - cycle_cost * cycles_k * scale
    reward[END] = scale * (new bins only the test's reset cycle hit - step_cost + closure_bonus * [closed])
The sum over the test is exactly the step-1 sum: only WHERE the reward lands changes.
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
    block_new_bins: tuple = ()   # step 2: new bins credited to each block, in idx order


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

    def attribute(self, plan, test_hits, covered_before, action_hits=None):
        """Input:  plan, the closed StimulusPlan that was simulated;
                test_hits, bool[64] from the test's coverage report;
                covered_before, bool[64] covered by the campaign before this test;
                action_hits, bool[n_blocks, 64] (step 2 only; ignored here).
        Output: an AttributionResult.
        Logic:  new = test_hits AND NOT covered_before; closed_now = everything covered after the
                test and this test added something; every action pays its cycles; END_TEST (the
                last action) also gets the test's coverage reward.
        """
        test_hits, covered_before, new_bins, closed_now = self._check(plan, test_hits, covered_before)
        rewards = self._cycle_costs(plan)
        rewards[-1] += compute_reward(new_bins, closed_now, self.reward)
        return AttributionResult(tuple(float(r) for r in rewards), test_hits, new_bins, closed_now, self.mode)

    def _check(self, plan, test_hits, covered_before):
        """Input: as attribute(). Output: (test_hits, covered_before, new bins, closed_now)."""
        if not plan.is_closed:
            raise ValueError("only a closed plan (ending with END_TEST) can be rewarded")
        test_hits = np.asarray(test_hits, dtype=bool)
        covered_before = np.asarray(covered_before, dtype=bool)
        if test_hits.shape != covered_before.shape:
            raise ValueError(f"test_hits {test_hits.shape} and covered_before {covered_before.shape} differ")
        new_bins = int((test_hits & ~covered_before).sum())
        closed_now = bool((test_hits | covered_before).all()) and new_bins > 0
        return test_hits, covered_before, new_bins, closed_now

    def _cycle_costs(self, plan):
        """Input: the plan. Output: list with every planned action's cycle cost (negative or 0)."""
        return [-self.cycle_cost * pa.spec.n_cycles * self.reward.scale for pa in plan.actions]


class ActionRewardAttributor(RewardAttributor):
    mode = "action"

    def attribute(self, plan, test_hits, covered_before, action_hits=None):
        """Input:  as RewardAttributor.attribute; action_hits, bool[n_blocks, 64] from the act_<idx>
                instances (required).
        Output: an AttributionResult with block_new_bins.
        Logic:  walk the blocks in idx order, seen = covered_before; block k gets the bins it hit
                that are not in seen yet, then seen |= its hits. END_TEST gets the new bins nobody
                hit (only the test's reset cycle did), minus step_cost, plus the closure bonus.
                The test's total reward equals the step-1 total for the same test.
        """
        test_hits, covered_before, new_bins, closed_now = self._check(plan, test_hits, covered_before)
        if action_hits is None:
            raise ValueError("reward_mode=action needs the per-block hits (act_<idx> instances)")
        action_hits = np.asarray(action_hits, dtype=bool)
        if action_hits.shape != (plan.n_blocks, test_hits.size):
            raise ValueError(f"action_hits {action_hits.shape} for {plan.n_blocks} blocks of {test_hits.size} bins")
        if (action_hits & ~test_hits).any():
            raise ValueError("a block hit a bin that the whole test did not: act_<idx> and cg_FIFO disagree")
        seen = covered_before.copy()
        block_new = []
        for hits in action_hits:
            block_new.append(int((hits & ~seen).sum()))
            seen |= hits
        rewards = self._cycle_costs(plan)
        for k, n in enumerate(block_new):
            rewards[k] += self.reward.scale * n
        rest = new_bins - sum(block_new)
        rewards[-1] += compute_reward(rest, closed_now, self.reward)
        return AttributionResult(tuple(float(r) for r in rewards), test_hits, new_bins, closed_now, self.mode,
                                 tuple(block_new))
