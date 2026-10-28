"""Turns one finished test into DQN transitions: one per planned action.

    transitions = TrajectoryBuilder().build(plan, attribution, next_state, done)

For planned action k:
    state      = the state the agent saw when it chose k (kept in the PlannedAction)
    action     = k's action id
    reward     = attribution.rewards[k]          (k's own reward, never the next action's)
    next_state = the state of action k+1; for the last action (END_TEST) the first state of the
                 next test, built after the campaign store took this test's coverage
    done       = only on the last action, and only if the campaign ended with this test
Credit flows back through next_state: Q(s, WRITE) = r_WRITE + gamma * max Q(s_after_WRITE, .).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class TrajectoryBuilder:
    def build(self, plan, attribution, next_state, done):
        """Input:  plan, the closed StimulusPlan; attribution, its AttributionResult;
                next_state, the first state of the next test; done, the campaign ended.
        Output: a list of Transition, one per planned action, in plan order.
        """
        if len(attribution.rewards) != len(plan):
            raise ValueError(f"{len(attribution.rewards)} rewards for {len(plan)} planned actions")
        transitions = []
        last = len(plan) - 1
        for k, planned in enumerate(plan.actions):
            following = plan.actions[k + 1].state if k < last else np.asarray(next_state, dtype=np.float32)
            transitions.append(Transition(planned.state, planned.action_id, attribution.rewards[k], following,
                                          bool(done) and k == last))
        return transitions
