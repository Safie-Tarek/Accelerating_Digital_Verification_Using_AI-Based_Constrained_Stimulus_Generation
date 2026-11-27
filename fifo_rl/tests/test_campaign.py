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


# ---------- train --resume ----------

def pretrained(tmp_path, **kw):
    """Train a small DQN into tmp_path/pre; return the path of its best.pt."""
    from fifo_rl.agents.dqn import DQNAgent
    cfg = small_cfg(learning_starts=10, batch_size=8, **kw)
    run_campaign.train(cfg, tmp_path / "pre", DQNAgent, verbose=False)
    return tmp_path / "pre" / "best.pt"


def resume_cfg(**kw):
    return small_cfg(learning_starts=10, batch_size=8, seed=7, **kw)


def test_resume_goes_on_training_the_saved_agent(tmp_path, monkeypatch):
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    steps_before = DQNAgent.load(ckpt).steps
    run_campaign.train(resume_cfg(), tmp_path / "res", DQNAgent, verbose=False, resume=ckpt)
    evals = rows(tmp_path / "res" / "eval.csv")
    assert [r["campaign"] for r in evals] == ["0", "2", "4"]  # validated before training, then as usual
    assert int(evals[0]["step"]) == steps_before
    assert DQNAgent.load(tmp_path / "res" / "last.pt").steps > steps_before
    summary = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert summary["resumed_from"] == str(ckpt) and summary["resumed_at_step"] == steps_before


def test_resume_keeps_the_saved_settings_unless_given(tmp_path, monkeypatch):
    import yaml
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path, lr=5e-4, gamma=0.8, buffer_capacity=64)
    agent = run_campaign.resume_agent(ckpt, resume_cfg(), 93, 9)  # defaults lr 1e-3, gamma 0.9: not given
    assert agent.cfg["lr"] == 5e-4 and agent.cfg["gamma"] == 0.8 and agent.buffer.capacity == 64
    assert agent.optimizer.param_groups[0]["lr"] == 5e-4
    agent = run_campaign.resume_agent(ckpt, resume_cfg(lr=1e-5), 93, 9, explicit={"lr"})
    assert agent.cfg["lr"] == 1e-5 and agent.optimizer.param_groups[0]["lr"] == 1e-5
    assert agent.cfg["gamma"] == 0.8
    assert len(agent.buffer) == 0 and agent.epsilon_schedule is not None
    from fifo_rl.agents.dqn import DQNAgent
    run_campaign.train(resume_cfg(), tmp_path / "res", DQNAgent, verbose=False, resume=ckpt)
    saved = yaml.safe_load((tmp_path / "res" / "config.yaml").read_text())
    assert saved["lr"] == 5e-4 and saved["gamma"] == 0.8  # config.yaml shows what was used


def test_resume_with_a_fixed_epsilon(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    agent = run_campaign.resume_agent(ckpt, resume_cfg(eps_start=0.1, eps_end=0.1), 93, 9,
                                      explicit={"eps_start", "eps_end"})
    assert agent.epsilon == pytest.approx(0.1)


def test_resume_rejects_a_checkpoint_of_another_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    with pytest.raises(SystemExit, match="87"):
        run_campaign.resume_agent(ckpt, resume_cfg(), 87, 22)  # the old 22-action design


def test_resume_cannot_change_the_network_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    with pytest.raises(SystemExit, match="hidden"):
        run_campaign.resume_agent(ckpt, resume_cfg(hidden=[64]), 93, 9, explicit={"hidden"})
    run_campaign.resume_agent(ckpt, resume_cfg(), 93, 9, explicit={"hidden"})  # same shape given: fine


def test_resume_keeps_the_loaded_agent_when_nothing_beats_it(tmp_path, monkeypatch):
    import torch
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    scores = iter(range(0, -100, -1))  # every validation scores lower than the one before
    monkeypatch.setattr(run_campaign, "score", lambda summary: (next(scores),))
    run_campaign.train(resume_cfg(), tmp_path / "res", DQNAgent, verbose=False, resume=ckpt)
    assert [r["best"] for r in rows(tmp_path / "res" / "eval.csv")] == ["1", "0", "0"]
    loaded, kept = DQNAgent.load(ckpt), DQNAgent.load(tmp_path / "res" / "best.pt")
    assert kept.steps == loaded.steps
    for key, value in loaded.online.state_dict().items():
        assert torch.equal(kept.online.state_dict()[key], value)


def test_resume_refuses_to_write_into_the_checkpoint_folder(tmp_path, monkeypatch):
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path)
    with pytest.raises(SystemExit, match="choose a new folder"):
        run_campaign.train(resume_cfg(), ckpt.parent, DQNAgent, verbose=False, resume=ckpt)


@pytest.mark.parametrize("argv", [["train", "--agent", "stub"], ["train", "--agent", "a2c"],
                                  ["eval", "--agent", "dqn", "--checkpoint", "x.pt"]])
def test_resume_needs_train_with_the_dqn(tmp_path, monkeypatch, argv):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="--resume"):
        run_campaign.main(argv + ["--resume", "x.pt", "--out", str(tmp_path / "out")])


def test_script_resume_from_the_command_line(tmp_path, monkeypatch):
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    ckpt = pretrained(tmp_path, lr=5e-4)
    run_campaign.main(["train", "--resume", str(ckpt), "--seed", "7", "--lr", "2e-4", "--n-campaigns", "2",
                       "--eval-every", "2", "--eval-campaigns", "2", "--campaign-budget", "4",
                       "--out", str(tmp_path / "res")])
    agent = DQNAgent.load(tmp_path / "res" / "last.pt")
    assert agent.cfg["lr"] == 2e-4  # given on the command line
    assert agent.cfg["learning_starts"] == 10 and agent.cfg["batch_size"] == 8  # kept from the checkpoint


# ---------- A2C wiring (Contract F) ----------

def a2c_small_cfg(**kw):
    return small_cfg(reward_mode="action", gamma=0.99, a2c_hidden=[16, 16], a2c_rollout_tests=1, **kw)


def test_default_config_has_the_a2c_keys():
    cfg = load_config()
    assert cfg["a2c_hidden"] == [128, 128] and cfg["a2c_lr"] == pytest.approx(7e-4)
    assert (cfg["a2c_grad_clip"], cfg["vf_coef"], cfg["ent_coef"], cfg["gae_lambda"]) == (0.5, 0.5, 0.01, 0.95)
    assert cfg["a2c_rollout_tests"] == 4 and cfg["a2c_normalize_adv"] is True
    assert (cfg["hidden"], cfg["lr"], cfg["grad_clip"]) == ([128, 128], 1e-3, 10.0)  # DQN defaults unchanged


def test_script_train_with_the_a2c_logs_its_losses(tmp_path, monkeypatch):
    from fifo_rl.agents.a2c import A2CAgent
    monkeypatch.chdir(tmp_path)
    run_campaign.train(a2c_small_cfg(), tmp_path / "out", A2CAgent, verbose=False)
    agent = A2CAgent.load(tmp_path / "out" / "best.pt")
    assert (agent.state_dim, agent.n_actions) == (93, 9) and agent.updates > 0
    train_rows = [r for r in rows(tmp_path / "out" / "tests.csv") if r["phase"] == "train"]
    assert all(r["epsilon"] == "0.0" for r in train_rows)
    updated = [r for r in train_rows if r["loss"]]
    assert updated and all(r["policy_loss"] and r["value_loss"] and r["entropy"] for r in updated)
    assert all(not r["entropy"] for r in train_rows if not r["loss"])  # stats only on tests with an update
    assert all(r["tests_charged_mean"] for r in rows(tmp_path / "out" / "eval.csv"))


def test_script_train_and_eval_a2c_from_the_command_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    common = ["--reward-mode", "action", "--gamma", "0.99", "--a2c-hidden", "[16,16]", "--campaign-budget", "4"]
    run_campaign.main(["train", "--agent", "a2c", "--n-campaigns", "2", "--eval-every", "2",
                       "--eval-campaigns", "2", "--out", str(tmp_path / "tr")] + common)
    assert json.loads((tmp_path / "tr" / "summary.json").read_text())["agent"] == "a2c"
    run_campaign.main(["eval", "--agent", "a2c", "--checkpoint", str(tmp_path / "tr" / "best.pt"),
                       "--eval-campaigns", "3", "--out", str(tmp_path / "ev")] + common)
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert summary["agent"] == "a2c" and summary["campaigns"] == 3 and summary["tests_charged"]["mean"] > 0
    for out in ("s1", "s2"):  # sampled play: its own name, and the same seed replays the same blocks
        run_campaign.main(["eval", "--agent", "a2c", "--sample", "--checkpoint", str(tmp_path / "tr" / "best.pt"),
                           "--eval-campaigns", "3", "--out", str(tmp_path / out)] + common)
    assert json.loads((tmp_path / "s1" / "summary.json").read_text())["agent"] == "a2c_sampled"
    assert [r["actions"] for r in rows(tmp_path / "s1" / "tests.csv")] ==         [r["actions"] for r in rows(tmp_path / "s2" / "tests.csv")]
    with pytest.raises(SystemExit, match="--sample"):
        run_campaign.main(["eval", "--agent", "random", "--sample", "--out", str(tmp_path / "x")])


def test_eval_a2c_needs_an_a2c_checkpoint(tmp_path, monkeypatch):
    from fifo_rl.agents.dqn import DQNAgent
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="a2c"):
        run_campaign.main(["eval", "--agent", "a2c", "--out", str(tmp_path / "ev")])
    run_campaign.train(small_cfg(learning_starts=10, batch_size=8), tmp_path / "dqn", DQNAgent, verbose=False)
    with pytest.raises(ValueError, match="not an A2C"):
        run_campaign.eval_agent("a2c", tmp_path / "dqn" / "best.pt", small_cfg())


# ---------- the shared checkpoint-selection rule ----------

class _Result:
    def __init__(self, tests_used, closed, cycles=100):
        self.tests_used, self.closed, self.total_cycles = tests_used, closed, cycles
        self.tests, self.final_bins, self.episode_return, self.wall_time = [], 64, 0.0, 0.0


def test_a_failed_campaign_is_charged_the_whole_budget():
    s = run_campaign.summarize([_Result(2, True), _Result(5, False)], budget=30)  # stopped early, still 30
    assert s["tests_charged"]["mean"] == pytest.approx(16.0)
    assert s["tests_used"]["mean"] == pytest.approx(3.5)


def test_score_prefers_fewer_expected_tests_over_closure_rate():
    """The case of docs/fifo_rl_finetune_results.md round 2: 99.5% closure in 1.77 tests must beat
    100% closure in 3.86 tests (the old closure-first rule picked the second)."""
    fast = run_campaign.summarize([_Result(1, True)] * 199 + [_Result(30, False)], budget=30)  # 1.145 expected
    slow = run_campaign.summarize([_Result(4, True)] * 200, budget=30)
    assert fast["closure_rate"] < slow["closure_rate"]
    assert run_campaign.score(fast) > run_campaign.score(slow)


def test_score_breaks_ties_on_cycles():
    short = run_campaign.summarize([_Result(2, True, cycles=200)], budget=30)
    long = run_campaign.summarize([_Result(2, True, cycles=300)], budget=30)
    assert run_campaign.score(short) > run_campaign.score(long)


def test_plot_campaign_draws_the_a2c_run(tmp_path, monkeypatch):
    from fifo_rl.agents.a2c import A2CAgent
    from scripts import plot_campaign
    monkeypatch.chdir(tmp_path)
    for seed in (0, 1):
        run_campaign.train(a2c_small_cfg(seed=seed), tmp_path / f"a2c{seed}", A2CAgent, verbose=False)
    plot_campaign.main([str(tmp_path / "a2c0"), str(tmp_path / "a2c1"), "--out", str(tmp_path / "plots")])
    assert (tmp_path / "plots" / "validation.png").is_file() and (tmp_path / "plots" / "a2c_losses.png").is_file()


# ---------- random block cost (random_block_cost / random_blocks) ----------

BASELINE_ID, RAND50_ID = ACTION_INDEX["BASELINE"], ACTION_INDEX["RAND50"]
NO_REWARD = RewardConfig(step_cost=0, closure_bonus=0, scale=1)


def test_block_cost_charges_only_the_named_blocks():
    plan = plan_of(BASELINE_ID, WRITE, RAND50_ID, END_TEST_ID)
    res = RewardAttributor(NO_REWARD, block_costs={BASELINE_ID: 0.5, RAND50_ID: 0.5}).attribute(plan, bins(), bins())
    assert res.rewards == (-0.5, 0.0, -0.5, 0.0)


def test_block_cost_adds_to_the_cycle_cost_in_step_2():
    from fifo_rl.campaign.reward_attributor import ActionRewardAttributor
    plan = plan_of(BASELINE_ID, WRITE, END_TEST_ID)
    hits = np.array([bins(1, 2), bins(3)])
    res = ActionRewardAttributor(NO_REWARD, cycle_cost=0.1, block_costs={BASELINE_ID: 1.0}).attribute(
        plan, bins(1, 2, 3), bins(), hits)
    assert res.rewards == pytest.approx((2 - 0.1 * 18 - 1.0, 1 - 0.1 * 10, 0.0))
    assert res.block_new_bins == (2, 1)  # the coverage numbers are not changed by the cost


def test_zero_cost_gives_exactly_todays_rewards():
    plan = plan_of(BASELINE_ID, RAND50_ID, END_TEST_ID)
    before = RewardAttributor().attribute(plan, bins(0, 1, 2), bins(0))
    after = RewardAttributor(block_costs={BASELINE_ID: 0.0, RAND50_ID: 0.0}).attribute(plan, bins(0, 1, 2), bins(0))
    assert before.rewards == after.rewards


def test_negative_block_cost_is_refused():
    with pytest.raises(ValueError, match="block costs"):
        RewardAttributor(block_costs={BASELINE_ID: -1.0})


def test_config_keys_become_block_costs():
    from fifo_rl.campaign.campaign_runner import random_block_costs
    cfg = load_config()
    assert cfg["random_block_cost"] == 0.0 and cfg["random_blocks"] == ["BASELINE", "RAND50"]
    assert random_block_costs(cfg) == {BASELINE_ID: 0.0, RAND50_ID: 0.0}
    assert random_block_costs({**cfg, "random_block_cost": 0.5}) == {BASELINE_ID: 0.5, RAND50_ID: 0.5}
    with pytest.raises(ValueError, match="unknown block"):
        random_block_costs({**cfg, "random_blocks": ["BASELIN"]})
    assert RunConfig.from_cfg({**cfg, "random_block_cost": 0.25}).block_costs[RAND50_ID] == 0.25


def test_runner_charges_the_cost_and_logs_the_random_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = [BASELINE_ID, BASELINE_ID, END_TEST_ID]
    rewards = {}
    for cost in (0.0, 1.0):
        cfg = load_config(overrides={"random_block_cost": cost, "campaign_budget": 1})
        r = CampaignRunner(RunConfig.from_cfg(cfg, run_root=tmp_path / f"c{cost}"), FakeLauncher())
        agent = ScriptedAgent(plan)
        r.run_campaign(agent, train=True)
        rewards[cost] = [rw for _, rw, _ in agent.observed]
    assert rewards[1.0][0] == pytest.approx(rewards[0.0][0] - 1 / 64)  # one block = cost x scale (1/64)
    assert rewards[1.0][2] == pytest.approx(rewards[0.0][2])           # END_TEST is not a random block


def test_script_logs_random_block_share(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_campaign.main(["eval", "--agent", "baseline", "--eval-campaigns", "2", "--campaign-budget", "4",
                       "--out", str(tmp_path / "ev")])
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert summary["random_block_share"] == 1.0  # always BASELINE
    assert all(r["n_random_blocks"] == r["n_blocks"] for r in rows(tmp_path / "ev" / "tests.csv"))
    run_campaign.train(small_cfg(), tmp_path / "tr", StubAgent, verbose=False)
    assert all(r["random_block_share"] for r in rows(tmp_path / "tr" / "eval.csv"))
