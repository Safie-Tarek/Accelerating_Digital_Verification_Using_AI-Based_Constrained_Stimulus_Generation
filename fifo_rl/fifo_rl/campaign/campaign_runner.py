"""The conductor of the plan-based design: runs tests until the campaign ends.

    runner = CampaignRunner(RunConfig.from_cfg(cfg), FakeLauncher())
    agent  = DQNAgent(runner.state_size, runner.n_actions, cfg, buffer, schedule)   # unchanged agent
    result = runner.run_campaign(agent, train=True, train_freq=cfg["train_freq"])

One test (run_one_test):
    1. PlanBuilder asks the agent for blocks until END_TEST          -> StimulusPlan
    2. TextPlanWriter writes it                                       -> run_dir/plan.txt
    3. the launcher simulates it (fake FIFO or Questa)                -> RunOutput
    4. ResultParser: did it pass?   report_parser: which bins?        -> RunResult, bool[64]
    5. RewardAttributor: one reward per planned action                -> AttributionResult
    6. CampaignStore.commit: the campaign's coverage grows
    7. TrajectoryBuilder: one transition per planned action (next state of END_TEST = the first
       state of the next test, encoded after step 6)
When training, every transition goes to agent.observe() and agent.learn() runs every train_freq
transitions - the same calls scripts/train.py makes, so DQNAgent needs no change.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fifo_rl.campaign.campaign_store import CampaignStore, TestSummary
from fifo_rl.campaign.reward_attributor import RewardAttributor
from fifo_rl.campaign.trajectory_builder import TrajectoryBuilder
from fifo_rl.launchers.base import SimulationError
from fifo_rl.plan.actions import N_ACTIONS
from fifo_rl.plan.plan_builder import PlanBuilder
from fifo_rl.plan.plan_file import TextPlanWriter
from fifo_rl.plan.state_encoder import StateEncoder
from fifo_rl.questa.report_parser import hits_vector
from fifo_rl.questa.result_parser import ResultParser
from fifo_rl.reward import RewardConfig


@dataclass(frozen=True)
class RunConfig:
    campaign_budget: int = 30          # tests per campaign (= one DQN episode)
    plan_max_len: int = 10             # blocks per plan before END_TEST is added
    n_history: int = 3                 # last blocks shown in the state
    cycle_cost: float = 0.0            # bins charged per simulated cycle
    base_seed: int = 0                 # every test's seed comes from (base_seed, campaign, test)
    run_root: Path = field(default=Path("runs/campaign/sims"))
    keep_runs: bool = False            # True: one folder per test; False: one folder reused
    stop_on_failure: bool = True       # a failed test (UVM error, assertion) stops the campaign

    @classmethod
    def from_cfg(cls, cfg, **overrides):
        """Input: the config dict (configs/default.yaml keys) and explicit overrides.
        Output: a RunConfig.
        """
        values = dict(campaign_budget=cfg["campaign_budget"], plan_max_len=cfg["plan_max_len"],
                      n_history=cfg["plan_history"], cycle_cost=cfg["cycle_cost"], base_seed=cfg["seed"])
        values.update(overrides)
        return cls(**values)

    def seed_for(self, campaign, test_no):
        """Input: campaign and test numbers. Output: a 32-bit seed usable as -sv_seed.
        Logic:  numpy's SeedSequence mixes (base seed, campaign, test): every test differs and the
                same base seed replays the same seeds.
        """
        return int(np.random.SeedSequence([self.base_seed, campaign, test_no]).generate_state(1)[0])


@dataclass(eq=False)
class CampaignResult:
    campaign: int
    tests: list            # TestSummary per test
    closed: bool           # 64/64 reached within the budget
    losses: list           # learn() losses (training only; None entries dropped)

    @property
    def tests_used(self):
        return len(self.tests)

    @property
    def final_bins(self):
        return self.tests[-1].total_bins if self.tests else 0

    @property
    def total_cycles(self):
        return sum(t.plan.sim_cycles() for t in self.tests)

    @property
    def wall_time(self):
        return sum(t.output.wall_time for t in self.tests)

    @property
    def episode_return(self):
        return float(sum(sum(t.attribution.rewards) for t in self.tests))


class CampaignRunner:
    def __init__(self, config, launcher, reward=RewardConfig(), result_parser=None):
        """Input:  config, a RunConfig; launcher, a FakeLauncher or QuestaLauncher;
                reward, the RewardConfig; result_parser, a ResultParser (default one).
        Output: nothing; call reset() (run_campaign does) before the first test.
        """
        self.config = config
        self.launcher = launcher
        self.encoder = StateEncoder(config.plan_max_len, config.n_history)
        self.planner = PlanBuilder(self.encoder, config.plan_max_len)
        self.writer = TextPlanWriter()
        self.result_parser = result_parser or ResultParser()
        self.attributor = RewardAttributor(reward, config.cycle_cost)
        self.traj_builder = TrajectoryBuilder()
        self.store = CampaignStore(config.campaign_budget)
        self.campaign = -1

    @property
    def state_size(self):
        """Input: nothing. Output: length of the state vector (93 by default)."""
        return self.encoder.size

    @property
    def n_actions(self):
        """Input: nothing. Output: number of actions (9)."""
        return N_ACTIONS

    def reset(self):
        """Start the next campaign. Input: nothing. Output: its first state."""
        self.campaign += 1
        self.store.reset()
        return self.encoder.encode(self.store)

    def run_dir(self, test_no):
        """Input: test number. Output: the folder for this test's files."""
        name = f"c{self.campaign:05d}_t{test_no:03d}" if self.config.keep_runs else "current"
        return Path(self.config.run_root) / name

    def run_one_test(self, agent):
        """Plan, simulate and score one test.

        Input:  agent, anything with act(state) -> action id.
        Output: (TestSummary, list of Transition).
        """
        if self.store.done():
            raise RuntimeError("the campaign is over: call reset() first")
        test_no = self.store.tests_done
        seed = self.config.seed_for(self.campaign, test_no)
        plan = self.planner.build(agent, self.store, test_no, seed)
        run_dir = self.run_dir(test_no)
        plan_file = self.writer.write(plan, run_dir / "plan.txt")
        output = self.launcher.launch(plan_file, seed, run_dir)

        result = self.result_parser.parse(output.result_file, output.log_file)
        if not result.passed and self.config.stop_on_failure:
            raise SimulationError(f"test {test_no} of campaign {self.campaign} failed ({result}); see {run_dir}")
        if result.cycles is not None and result.cycles != plan.sim_cycles():
            raise SimulationError(f"the simulator ran {result.cycles} cycles, the plan has {plan.sim_cycles()}; "
                                  f"the plan file was not applied as written ({plan_file})")
        test_hits = hits_vector(Path(output.coverage_report).read_text())

        attribution = self.attributor.attribute(plan, test_hits, self.store.covered)
        total_bins = self.store.commit(test_hits)
        done = self.store.done()
        next_state = self.encoder.encode(self.store)
        transitions = self.traj_builder.build(plan, attribution, next_state, done)
        summary = TestSummary(self.campaign, test_no, seed, plan, result, attribution, output, total_bins)
        self.store.record(summary)
        return summary, transitions

    def run_campaign(self, agent, train=False, train_freq=1, on_test=None):
        """Play one whole campaign.

        Input:  agent; train, call agent.observe() for every transition and agent.learn() every
                train_freq transitions (needs a learning agent); on_test, optional
                callback(summary, transitions, losses) after each test (logging).
        Output: a CampaignResult.
        """
        self.reset()
        losses = []
        while not self.store.done():
            summary, transitions = self.run_one_test(agent)
            test_losses = []
            if train:
                for t in transitions:
                    agent.observe(t.state, t.action, t.reward, t.next_state, t.done)
                    if agent.steps % train_freq == 0:
                        loss = agent.learn()
                        if loss is not None:
                            test_losses.append(loss)
            losses += test_losses
            if on_test is not None:
                on_test(summary, transitions, test_losses)
        return CampaignResult(self.campaign, list(self.store.history), self.store.all_covered(), losses)
