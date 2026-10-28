"""Tests for fifo_rl/plan: StimulusPlan, StateEncoder, PlanBuilder and the plan file."""

import numpy as np
import pytest

from fifo_rl.campaign.campaign_store import CampaignStore
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID
from fifo_rl.plan.plan_builder import PlanBuilder
from fifo_rl.plan.plan_file import PlanFileError, TextPlanWriter, parse_plan, read_plan
from fifo_rl.plan.state_encoder import StateEncoder
from fifo_rl.plan.stimulus_plan import StimulusPlan

WRITE, IDLE, READ = ACTION_INDEX["WRITE"], ACTION_INDEX["IDLE"], ACTION_INDEX["READ"]


class ScriptedAgent:
    """Plays a fixed list of actions and remembers the states it was shown."""
    name = "scripted"

    def __init__(self, *actions):
        self.actions = list(actions)
        self.seen = []

    def act(self, state):
        self.seen.append(np.array(state))
        return self.actions.pop(0)


def closed_plan(*actions, max_len=10):
    plan = StimulusPlan(plan_id=3, seed=42, max_len=max_len)
    for a in actions:
        plan.add(a, np.zeros(4))
    return plan


# ---------- StimulusPlan ----------

def test_plan_numbers_actions_and_counts_cycles():
    plan = closed_plan(WRITE, IDLE, END_TEST_ID)
    assert [pa.idx for pa in plan.actions] == [0, 1, 2]
    assert plan.is_closed and plan.n_blocks == 2 and len(plan) == 3
    assert plan.total_cycles() == 14 and plan.sim_cycles() == 15
    assert plan.last_actions(2) == [END_TEST_ID, IDLE]


def test_plan_refuses_after_end_test_and_when_full():
    plan = closed_plan(END_TEST_ID)
    with pytest.raises(ValueError):
        plan.add(WRITE, np.zeros(4))
    full = closed_plan(WRITE, WRITE, max_len=2)
    assert full.is_full
    with pytest.raises(ValueError):
        full.add(IDLE, np.zeros(4))
    full.add(END_TEST_ID, np.zeros(4))
    assert full.is_closed


def test_planned_action_keeps_a_copy_of_the_state():
    plan = StimulusPlan(0, 0, 5)
    state = np.zeros(3)
    plan.add(WRITE, state)
    state[0] = 7
    assert plan.actions[0].state[0] == 0


# ---------- StateEncoder ----------

def test_state_is_93_numbers_with_history_position_and_tests_left():
    store = CampaignStore(budget=10)
    store.commit(np.eye(64, dtype=bool)[5])          # one test done, bin 5 covered
    enc = StateEncoder(plan_max_len=10)
    assert enc.size == 93
    plan = closed_plan(WRITE, IDLE)                  # still open: 2 blocks so far
    s = enc.encode(store, plan)
    assert s.shape == (93,) and s.dtype == np.float32
    assert s[5] == 1 and s[:64].sum() == 1
    assert s[64 + 0 * 9 + IDLE] == 1                 # newest block first
    assert s[64 + 1 * 9 + WRITE] == 1
    assert s[64 + 2 * 9: 91].sum() == 0              # no third block yet
    assert s[91] == pytest.approx(0.2)               # 2 of 10 blocks
    assert s[92] == pytest.approx(0.9)               # 9 of 10 tests left


def test_state_without_plan_is_the_start_of_a_test():
    s = StateEncoder(plan_max_len=10).encode(CampaignStore(budget=4))
    assert s[64:92].sum() == 0 and s[92] == 1.0


# ---------- PlanBuilder ----------

def test_builder_stops_at_end_test_and_stores_the_states_seen():
    enc = StateEncoder(plan_max_len=10)
    agent = ScriptedAgent(WRITE, IDLE, END_TEST_ID)
    plan = PlanBuilder(enc, 10).build(agent, CampaignStore(budget=5), plan_id=0, seed=1)
    assert plan.action_ids() == [WRITE, IDLE, END_TEST_ID]
    assert not plan.actions[-1].forced
    for planned, seen in zip(plan.actions, agent.seen):
        np.testing.assert_array_equal(planned.state, seen)
    assert agent.seen[1][64 + WRITE] == 1            # second choice knew the first block


def test_builder_forces_end_test_when_the_plan_is_full():
    agent = ScriptedAgent(WRITE, WRITE, WRITE)
    plan = PlanBuilder(StateEncoder(plan_max_len=2), 2).build(agent, CampaignStore(budget=5), 0, 1)
    assert plan.action_ids() == [WRITE, WRITE, END_TEST_ID]
    assert plan.actions[-1].forced
    assert len(agent.seen) == 2                      # the agent was not asked for the forced END_TEST


def test_end_test_first_gives_an_empty_test():
    plan = PlanBuilder(StateEncoder(10), 10).build(ScriptedAgent(END_TEST_ID), CampaignStore(budget=5), 0, 1)
    assert plan.n_blocks == 0 and plan.sim_cycles() == 1


# ---------- plan file ----------

def test_writer_format_and_round_trip(tmp_path):
    plan = closed_plan(WRITE, IDLE, END_TEST_ID)
    path = TextPlanWriter().write(plan, tmp_path / "sub" / "plan.txt")
    assert path.read_text().splitlines() == [
        "PLAN id=3 seed=42 version=1",
        "ACT idx=0 name=WRITE wr=100 rd=0 rst=0 cycles=10",
        "ACT idx=1 name=IDLE wr=0 rd=0 rst=0 cycles=4",
        "END_TEST",
    ]
    parsed = read_plan(path)
    assert (parsed.plan_id, parsed.seed, parsed.version) == (3, 42, 1)
    assert [(b.idx, b.name, b.wr_pct, b.rd_pct, b.rst_pct, b.n_cycles) for b in parsed.blocks] == [
        (0, "WRITE", 100, 0, 0, 10), (1, "IDLE", 0, 0, 0, 4)]
    assert parsed.total_cycles() == plan.total_cycles()


def test_writer_refuses_an_open_plan():
    with pytest.raises(ValueError):
        TextPlanWriter().format(closed_plan(WRITE))


@pytest.mark.parametrize("text", [
    "",
    "ACT idx=0 wr=1 rd=0 rst=0 cycles=1\nEND_TEST\n",                                    # no PLAN line
    "PLAN id=1 seed=2 version=1\nACT idx=0 wr=1 rd=0 rst=0 cycles=1\n",                  # no END_TEST
    "PLAN id=1 seed=2 version=1\nACT idx=1 wr=1 rd=0 rst=0 cycles=1\nEND_TEST\n",        # idx must start at 0
    "PLAN id=1 seed=2 version=1\nACT idx=0 wr=101 rd=0 rst=0 cycles=1\nEND_TEST\n",      # percentage > 100
    "PLAN id=1 seed=2 version=1\nACT idx=0 wr=1 rd=0 cycles=1\nEND_TEST\n",              # rst missing
    "PLAN id=1 seed=2 version=1\nHELLO\nEND_TEST\n",                                     # unknown line
])
def test_parser_rejects_broken_files(text):
    with pytest.raises(PlanFileError):
        parse_plan(text)
