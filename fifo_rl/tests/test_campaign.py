"""Tests for fifo_rl/campaign: reward split, campaign store, transitions, the runner, the script."""

import csv
import json

import numpy as np
import pytest

from fifo_rl.agents.stub_agent import StubAgent
from fifo_rl.campaign.campaign_runner import CampaignRunner, RunConfig
from fifo_rl.campaign.campaign_store import CampaignStore
from fifo_rl.campaign.reward_attributor import RewardAttributor
from fifo_rl.campaign.trajectory_builder import TrajectoryBuilder
from fifo_rl.config import load_config
from fifo_rl.launchers.base import RunOutput, SimulationError, SimulatorLauncher
from fifo_rl.launchers.fake_launcher import FakeLauncher, format_report
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID
from fifo_rl.plan.stimulus_plan import StimulusPlan
from fifo_rl.reward import RewardConfig
from scripts import run_campaign

WRITE, IDLE = ACTION_INDEX["WRITE"], ACTION_INDEX["IDLE"]
GOOD_PLAN = [ACTION_INDEX[n] for n in
             ("WRITE", "IDLE", "READ", "WRITE_READ", "IDLE", "WRITE", "WRITE_READ", "IDLE", "RESET_TRAFFIC")]


def bins(*indices):
    hit = np.zeros(64, dtype=bool)
    hit[list(indices)] = True
    return hit


def plan_of(*actions):
    plan = StimulusPlan(0, 0, 10)
    for i, a in enumerate(actions):
        plan.add(a, np.full(3, i, dtype=np.float32))
    return plan


class ScriptedAgent:
    """Replays one action list per test (as many tests as lists), like a learning agent."""
    name = "scripted"

    def __init__(self, *plans):
        self.queue = [a for plan in plans for a in plan]
        self.observed = []
        self.steps = 0

    def act(self, state):
        return self.queue.pop(0)

    def observe(self, s, a, r, s2, done):
        self.observed.append((a, r, done))
        self.steps += 1

    def learn(self):
        return 0.5


# ---------- RewardAttributor ----------

def test_step1_reward_goes_to_end_test():
    plan = plan_of(WRITE, IDLE, END_TEST_ID)
    res = RewardAttributor().attribute(plan, bins(0, 1, 2), covered_before=bins(0))
    assert res.new_bins == 2 and not res.closed_now and res.mode == "test"
    assert res.rewards == (0.0, 0.0, pytest.approx((2 - 1) / 64))


def test_closing_test_gets_the_bonus():
    res = RewardAttributor().attribute(plan_of(END_TEST_ID), np.ones(64, bool), covered_before=bins(*range(60)))
    assert res.closed_now and res.rewards[-1] == pytest.approx((4 - 1 + 10) / 64)


def test_cycle_cost_charges_every_block():
    plan = plan_of(WRITE, IDLE, END_TEST_ID)
    res = RewardAttributor(RewardConfig(step_cost=0, closure_bonus=0, scale=1), cycle_cost=0.5).attribute(
        plan, bins(), bins())
    assert res.rewards == (-5.0, -2.0, 0.0)               # 10 and 4 cycles x 0.5


def test_attributor_needs_a_closed_plan():
    with pytest.raises(ValueError):
        RewardAttributor().attribute(plan_of(WRITE), bins(), bins())


# ---------- CampaignStore ----------

def test_store_commits_and_ends():
    store = CampaignStore(budget=2)
    assert store.commit(bins(1, 2)) == 2 and store.tests_left == 1 and not store.done()
    store.commit(bins(2, 3))
    assert store.n_covered == 3 and store.done()
    with pytest.raises(ValueError):
        store.commit(bins())
    store.reset()
    assert store.n_covered == 0 and store.tests_left == 2


def test_store_ends_when_all_covered():
    store = CampaignStore(budget=5)
    store.commit(np.ones(64, bool))
    assert store.all_covered() and store.done()


# ---------- TrajectoryBuilder ----------

def test_one_transition_per_action_linked_by_next_state():
    plan = plan_of(WRITE, IDLE, END_TEST_ID)
    res = RewardAttributor().attribute(plan, bins(0), bins())
    next_state = np.full(3, 9, dtype=np.float32)
    ts = TrajectoryBuilder().build(plan, res, next_state, done=True)
    assert [t.action for t in ts] == [WRITE, IDLE, END_TEST_ID]
    assert [t.reward for t in ts] == list(res.rewards)    # each keeps its own reward
    np.testing.assert_array_equal(ts[0].next_state, plan.actions[1].state)
    np.testing.assert_array_equal(ts[1].next_state, plan.actions[2].state)
    np.testing.assert_array_equal(ts[2].next_state, next_state)
    assert [t.done for t in ts] == [False, False, True]


# ---------- CampaignRunner ----------

def runner(tmp_path, launcher=None, **kw):
    config = RunConfig(campaign_budget=kw.pop("budget", 5), run_root=tmp_path / "sims", **kw)
    return CampaignRunner(config, launcher or FakeLauncher())


def test_good_plan_closes_in_one_test_and_trains_the_agent(tmp_path):
    r = runner(tmp_path)
    agent = ScriptedAgent(GOOD_PLAN + [END_TEST_ID], GOOD_PLAN + [END_TEST_ID])
    result = r.run_campaign(agent, train=True)
    assert result.closed and result.final_bins == 64
    assert result.tests_used <= 2                          # RESET_TRAFFIC is random: rarely a 2nd test
    test = result.tests[0]
    assert test.plan.n_blocks == 9 and test.plan.sim_cycles() == 67
    assert (tmp_path / "sims" / "current" / "plan.txt").is_file()
    assert len(agent.observed) == sum(len(t.plan) for t in result.tests)
    assert agent.observed[-1][2] is True                   # done only on the last transition
    assert all(done is False for _, _, done in agent.observed[:-1])
    assert result.losses and r.state_size == 93 and r.n_actions == 9


def test_campaign_uses_the_whole_budget_when_it_does_not_close(tmp_path):
    r = runner(tmp_path, budget=3)
    result = r.run_campaign(ScriptedAgent([IDLE, END_TEST_ID], [IDLE, END_TEST_ID], [IDLE, END_TEST_ID]))
    assert result.tests_used == 3 and not result.closed
    assert result.tests[1].attribution.new_bins == 0       # the same test again finds nothing new
    assert result.tests[1].attribution.rewards[-1] == pytest.approx(-1 / 64)


def test_seeds_differ_per_test_and_repeat_per_base_seed(tmp_path):
    config = RunConfig(base_seed=4)
    assert config.seed_for(0, 0) != config.seed_for(0, 1) != config.seed_for(1, 0)
    assert config.seed_for(2, 3) == RunConfig(base_seed=4).seed_for(2, 3) < 2 ** 32


def test_keep_runs_gives_one_folder_per_test(tmp_path):
    r = runner(tmp_path, budget=2, keep_runs=True)
    r.run_campaign(ScriptedAgent([END_TEST_ID], [END_TEST_ID]))
    assert sorted(p.name for p in (tmp_path / "sims").iterdir()) == ["c00000_t000", "c00000_t001"]


class BrokenLauncher(SimulatorLauncher):
    """Writes a report and a result file that says the test failed / ran the wrong number of cycles."""

    def __init__(self, result_text):
        self.result_text = result_text

    def launch(self, plan_file, seed, run_dir):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "coverage.txt").write_text(format_report(np.zeros(64)))
        (run_dir / "result.txt").write_text(self.result_text)
        return RunOutput(run_dir, plan_file, run_dir / "coverage.txt", run_dir / "result.txt", None, 0.0, 0.0)


def test_failed_or_misapplied_tests_stop_the_campaign(tmp_path):
    with pytest.raises(SimulationError, match="failed"):
        runner(tmp_path, launcher=BrokenLauncher("passed=0 errors=1")).run_campaign(ScriptedAgent([END_TEST_ID]))
    with pytest.raises(SimulationError, match="not applied"):
        runner(tmp_path, launcher=BrokenLauncher("passed=1 cycles=99")).run_campaign(ScriptedAgent([END_TEST_ID]))


# ---------- scripts/run_campaign.py ----------

def small_cfg(**kw):
    return load_config(overrides={"n_campaigns": 4, "eval_every": 2, "eval_campaigns": 3, "campaign_budget": 4,
                                  "buffer_capacity": 100, "eps_decay_steps": 50, **kw})


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_script_train_writes_everything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    result = run_campaign.train(small_cfg(), out, StubAgent, verbose=False)
    for name in ("config.yaml", "tests.csv", "eval.csv", "summary.json", "best.pt", "last.pt"):
        assert (out / name).is_file()
    assert len(result["eval"]) == 2
    tests = rows(out / "tests.csv")
    assert {r["phase"] for r in tests} == {"train", "validation"}
    assert all(r["actions"].split()[-1] == "END_TEST" for r in tests)
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["action_counts"]) == set(ACTION_INDEX)


def test_script_train_with_the_real_dqn(tmp_path, monkeypatch):
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    cfg = small_cfg(learning_starts=10, batch_size=8, train_freq=1)
    run_campaign.train(cfg, tmp_path / "out", DQNAgent, verbose=False)
    agent = DQNAgent.load(tmp_path / "out" / "best.pt")
    assert (agent.state_dim, agent.n_actions) == (93, 9)
    assert any(r["loss"] for r in rows(tmp_path / "out" / "tests.csv"))


def test_script_eval_random_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_campaign.main(["eval", "--agent", "random", "--eval-campaigns", "3", "--campaign-budget", "6",
                       "--out", str(tmp_path / "ev")])
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert summary["agent"] == "random" and summary["campaigns"] == 3 and summary["launcher"] == "fake"
    assert summary["cycles_per_campaign"]["mean"] > 0


def test_script_eval_dqn_needs_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_campaign.main(["eval", "--agent", "dqn", "--out", str(tmp_path / "ev")])
