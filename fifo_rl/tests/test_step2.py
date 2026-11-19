"""Step 2 on the fake FIFO: act_<idx> instances in the report, hits_per_action, per-action reward."""

import json

import numpy as np
import pytest

from fifo_rl import fake_fifo
from fifo_rl.campaign.campaign_runner import CampaignRunner, RunConfig
from fifo_rl.campaign.reward_attributor import ActionRewardAttributor, RewardAttributor
from fifo_rl.launchers.fake_launcher import FakeLauncher, format_report
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID, RESET_CYCLE
from fifo_rl.plan.plan_file import TextPlanWriter
from fifo_rl.plan.stimulus_plan import StimulusPlan
from fifo_rl.questa.report_parser import ReportError, hits_per_action, hits_vector, parse_bin_hits
from scripts import run_campaign
from tests.test_campaign import GOOD_PLAN, ScriptedAgent, bins, plan_of, rows, small_cfg

WRITE, IDLE, READ = ACTION_INDEX["WRITE"], ACTION_INDEX["IDLE"], ACTION_INDEX["READ"]


def launch(tmp_path, names, per_action, seed=5):
    plan = StimulusPlan(0, seed, 10)
    for name in names:
        plan.add(ACTION_INDEX[name], np.zeros(1))
    plan.add(END_TEST_ID, np.zeros(1))
    path = TextPlanWriter().write(plan, tmp_path / "plan.txt")
    out = FakeLauncher(per_action=per_action).launch(path, seed, tmp_path / ("a" if per_action else "t"))
    return plan, out.coverage_report.read_text()


# ---------- FakeLauncher + report parser ----------

def test_step1_report_is_unchanged_by_the_option(tmp_path):
    _, text = launch(tmp_path, ("RAND50", "BASELINE"), per_action=False)
    assert "act_" not in text
    assert text == format_report(parse_counts(text))


def parse_counts(text):
    from fifo_rl.coverage_model import BIN_NAMES
    hits = parse_bin_hits(text)
    return [hits[n] for n in BIN_NAMES]


def test_per_action_report_has_one_instance_per_block(tmp_path):
    names = ("WRITE", "IDLE", "READ", "RESET_TRAFFIC")
    plan, text = launch(tmp_path, names, per_action=True)
    _, plain = launch(tmp_path, names, per_action=False)
    np.testing.assert_array_equal(hits_vector(text), hits_vector(plain))      # cg_FIFO is the same test
    per_block = hits_per_action(text, plan.n_blocks)
    assert per_block.shape == (4, 64)
    reset_only = fake_fifo.hits([RESET_CYCLE])
    np.testing.assert_array_equal(per_block.any(axis=0) | reset_only, hits_vector(text))


def test_block_counts_add_up_to_the_test_minus_the_reset_cycle(tmp_path):
    from fifo_rl.questa.report_parser import _parse_lines, _sections
    plan, text = launch(tmp_path, ("WRITE", "RAND50", "READ"), per_action=True)
    sections = _sections(text)
    total = parse_bin_hits(text)
    blocks = [_parse_lines(lines) for (cg, inst), lines in sections.items() if inst]
    reset = fake_fifo.bin_hit_counts(fake_fifo.simulate([RESET_CYCLE]))
    from fifo_rl.coverage_model import BIN_NAMES
    for i, name in enumerate(BIN_NAMES):
        assert sum(b[name] for b in blocks) + reset[i] == total[name]


def test_step1_report_cannot_be_read_per_action(tmp_path):
    plan, text = launch(tmp_path, ("WRITE", "IDLE"), per_action=False)
    with pytest.raises(ReportError, match="no act_0 instance"):
        hits_per_action(text, plan.n_blocks)


def test_missing_or_extra_instances_are_rejected(tmp_path):
    plan, text = launch(tmp_path, ("WRITE", "IDLE"), per_action=True)
    with pytest.raises(ReportError, match="no act_2"):
        hits_per_action(text, 3)
    with pytest.raises(ReportError, match="only 1 blocks"):
        hits_per_action(text, 1)


def test_real_questa_reports_still_parse():
    from pathlib import Path
    text = (Path(__file__).parent / "fixtures" / "cov_seed1_300cyc.txt").read_text()
    assert hits_vector(text).sum() == 63


# ---------- ActionRewardAttributor ----------

def test_first_block_to_hit_a_bin_gets_it():
    plan = plan_of(WRITE, IDLE, READ, END_TEST_ID)
    action_hits = np.array([bins(1, 2), bins(2, 3), bins(4)])
    res = ActionRewardAttributor().attribute(plan, bins(1, 2, 3, 4, 5), bins(1), action_hits)
    assert res.mode == "action" and res.block_new_bins == (1, 1, 1)       # bin 1 was covered before
    assert res.rewards[:3] == pytest.approx((1 / 64, 1 / 64, 1 / 64))
    assert res.rewards[3] == pytest.approx((1 - 1) / 64)                  # bin 5: only the reset cycle hit it


def test_total_reward_equals_step1():
    rng = np.random.default_rng(0)
    plan = plan_of(WRITE, IDLE, READ, END_TEST_ID)
    for _ in range(50):
        action_hits = rng.random((3, 64)) < 0.3
        test_hits = action_hits.any(axis=0) | (rng.random(64) < 0.1)
        before = rng.random(64) < 0.5
        step1 = RewardAttributor().attribute(plan, test_hits, before)
        step2 = ActionRewardAttributor().attribute(plan, test_hits, before, action_hits)
        assert sum(step2.rewards) == pytest.approx(sum(step1.rewards))
        assert step2.new_bins == step1.new_bins and step2.closed_now == step1.closed_now


def test_closing_bonus_and_step_cost_stay_on_end_test():
    plan = plan_of(WRITE, END_TEST_ID)
    res = ActionRewardAttributor().attribute(plan, np.ones(64, bool), bins(*range(60)), np.ones((1, 64), bool))
    assert res.rewards == pytest.approx((4 / 64, (-1 + 10) / 64))


def test_bad_action_hits_are_rejected():
    plan = plan_of(WRITE, END_TEST_ID)
    with pytest.raises(ValueError, match="needs the per-block hits"):
        ActionRewardAttributor().attribute(plan, bins(1), bins())
    with pytest.raises(ValueError, match="disagree"):
        ActionRewardAttributor().attribute(plan, bins(1), bins(), np.array([bins(2)]))


# ---------- runner + script ----------

def test_runner_in_action_mode_closes_with_the_good_plan(tmp_path):
    config = RunConfig(campaign_budget=3, run_root=tmp_path / "sims", reward_mode="action")
    r = CampaignRunner(config, FakeLauncher(per_action=True))
    agent = ScriptedAgent(GOOD_PLAN + [END_TEST_ID], GOOD_PLAN + [END_TEST_ID])
    result = r.run_campaign(agent, train=True)
    first = result.tests[0].attribution
    assert first.mode == "action" and len(first.block_new_bins) == 9
    assert sum(first.block_new_bins) <= first.new_bins
    assert agent.observed[0][1] > 0                       # the first block is rewarded at once


def test_action_mode_with_a_step1_launcher_stops(tmp_path):
    config = RunConfig(campaign_budget=3, run_root=tmp_path / "sims", reward_mode="action")
    with pytest.raises(ReportError):
        CampaignRunner(config, FakeLauncher()).run_campaign(ScriptedAgent([WRITE, END_TEST_ID]))


def test_unknown_reward_mode(tmp_path):
    with pytest.raises(ValueError, match="reward_mode"):
        CampaignRunner(RunConfig(reward_mode="block"), FakeLauncher())


def test_script_train_action_mode_logs_block_rewards(tmp_path, monkeypatch):
    from fifo_rl.agents.stub_agent import StubAgent
    monkeypatch.chdir(tmp_path)
    run_campaign.train(small_cfg(reward_mode="action"), tmp_path / "out", StubAgent, verbose=False)
    tests = rows(tmp_path / "out" / "tests.csv")
    assert {r["reward_mode"] for r in tests} == {"action"}
    assert all(len(r["block_new_bins"].split()) == int(r["n_blocks"]) for r in tests)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["reward_mode"] == "action"


def test_script_never_overwrites_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = ["eval", "--agent", "random", "--eval-campaigns", "2", "--out", str(tmp_path / "ev")]
    run_campaign.main(args)
    with pytest.raises(SystemExit, match="never overwritten"):
        run_campaign.main(args)
