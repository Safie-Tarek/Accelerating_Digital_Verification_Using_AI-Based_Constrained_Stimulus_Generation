"""One test plan: the ordered blocks the agent chose for one simulation.

    plan = StimulusPlan(plan_id=12, seed=..., max_len=8)
    plan.add(action_id, state)        # -> PlannedAction with idx 0, 1, 2, ...
    plan.is_closed                    # True once END_TEST was added

A PlannedAction keeps the state the agent saw when it chose the block: after the simulation the
TrajectoryBuilder pairs that state with the block's reward (DQN learns "in state s, a gave r").
idx is the block's position in the plan; it is also the action_idx the testbench will stamp on
the block's transactions in step 2 (one covergroup instance act_<idx> per block).
"""

from dataclasses import dataclass, field

import numpy as np

from fifo_rl.plan.actions import END_TEST_ID, get_action


@dataclass(frozen=True, eq=False)
class PlannedAction:
    idx: int                                   # position in the plan: 0, 1, 2, ...
    action_id: int                             # which of the 9 actions
    state: np.ndarray = field(repr=False)      # the state vector the agent saw when choosing it
    forced: bool = False                       # True if PlanBuilder added END_TEST (plan was full)

    @property
    def spec(self):
        """Input: nothing. Output: the ActionSpec of this block."""
        return get_action(self.action_id)


class StimulusPlan:
    def __init__(self, plan_id, seed, max_len):
        """Input:  plan_id, the test number; seed, the test's simulation seed;
                max_len, the most blocks a plan may hold before END_TEST (>= 1).
        Output: nothing; an empty, open plan.
        """
        if max_len < 1:
            raise ValueError(f"max_len must be at least 1, got {max_len}")
        self.plan_id = plan_id
        self.seed = seed
        self.max_len = max_len
        self.actions = []

    def __len__(self):
        """Input: nothing. Output: number of planned actions, END_TEST included."""
        return len(self.actions)

    @property
    def blocks(self):
        """Input: nothing. Output: the planned actions that run cycles (everything but END_TEST)."""
        return [pa for pa in self.actions if pa.action_id != END_TEST_ID]

    @property
    def n_blocks(self):
        """Input: nothing. Output: number of blocks (END_TEST not counted)."""
        return len(self.blocks)

    @property
    def is_closed(self):
        """Input: nothing. Output: True once the last planned action is END_TEST."""
        return bool(self.actions) and self.actions[-1].action_id == END_TEST_ID

    @property
    def is_full(self):
        """Input: nothing. Output: True when max_len blocks are planned (only END_TEST may follow)."""
        return self.n_blocks >= self.max_len

    def add(self, action_id, state, forced=False):
        """Append one chosen action.

        Input:  action_id, 0-8; state, the state vector the agent saw; forced, see PlannedAction.
        Output: the new PlannedAction (idx = its position).
        Logic:  refuse to add after END_TEST, refuse a block when the plan is full, check the id,
                keep a copy of the state (the caller may reuse its array).
        """
        if self.is_closed:
            raise ValueError("the plan is closed by END_TEST")
        spec = get_action(action_id)
        if self.is_full and not spec.is_end_test:
            raise ValueError(f"the plan already holds {self.max_len} blocks; only END_TEST may follow")
        planned = PlannedAction(len(self.actions), spec.id, np.array(state, dtype=np.float32), forced)
        self.actions.append(planned)
        return planned

    def action_ids(self):
        """Input: nothing. Output: the action ids in plan order (END_TEST included)."""
        return [pa.action_id for pa in self.actions]

    def last_actions(self, n):
        """Input: n. Output: the ids of the last n planned actions, newest first (fewer if short)."""
        return [pa.action_id for pa in reversed(self.actions[-n:])] if n > 0 else []

    def total_cycles(self):
        """Input: nothing. Output: clock cycles of all blocks (without the test's first reset)."""
        return sum(pa.spec.n_cycles for pa in self.actions)

    def sim_cycles(self):
        """Input: nothing. Output: clock cycles the simulation runs: 1 reset cycle + all blocks."""
        return 1 + self.total_cycles()
