"""Builds one test plan by asking the agent for one block at a time.

    plan = PlanBuilder(encoder, max_len=8).build(agent, store, plan_id=12, seed=...)

Loop: state -> agent.act(state) -> add the block -> repeat, until the agent chooses END_TEST or
the plan holds max_len blocks (then END_TEST is added for it, marked forced=True). Only the agent
chooses blocks; the builder never filters them (no legality mask). An END_TEST chosen first gives
an empty test (just the reset cycle), which still costs a whole simulation run in the reward.
"""

from fifo_rl.plan.actions import END_TEST_ID
from fifo_rl.plan.stimulus_plan import StimulusPlan


class PlanBuilder:
    def __init__(self, encoder, max_len):
        """Input: encoder, a plan StateEncoder; max_len, blocks per plan before END_TEST is forced."""
        self.encoder = encoder
        self.max_len = max_len

    def build(self, agent, store, plan_id, seed):
        """Build one closed plan.

        Input:  agent, anything with act(state) -> action id (DQNAgent, GreedyWrapper, RandomAgent);
                store, the CampaignStore (read through the encoder only, never changed here);
                plan_id, the test number; seed, the test's simulation seed.
        Output: a StimulusPlan ending with END_TEST; every PlannedAction holds the state it was
                chosen in.
        """
        plan = StimulusPlan(plan_id, seed, self.max_len)
        while True:
            state = self.encoder.encode(store, plan)
            if plan.is_full:
                plan.add(END_TEST_ID, state, forced=True)
                return plan
            action_id = int(agent.act(state))
            plan.add(action_id, state)
            if action_id == END_TEST_ID:
                return plan
