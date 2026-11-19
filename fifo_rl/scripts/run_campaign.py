"""Plan-based campaigns: train DQN / DDQN on the fake FIFO, evaluate on the fake FIFO or real Questa.

Run from the fifo_rl/ folder:
    python -m scripts.run_campaign train --gamma 0.99 --eps-decay-steps 60000      # DDQN, fake FIFO
    python -m scripts.run_campaign eval --agent dqn --checkpoint runs/campaign/train_ddqn_seed0/best.pt
    python -m scripts.run_campaign eval --agent random                              # baseline
    python -m scripts.run_campaign eval --agent dqn --checkpoint <best.pt> --launcher questa --eval-campaigns 3
    python -m scripts.run_campaign train --resume runs/campaign/train_ddqn_seed0/best.pt --seed 10   # go on training

One test = one simulation of a plan of blocks; one campaign = up to campaign_budget tests = one
DQN episode. Seeds: training campaigns use base seed `seed`, the validation inside `train` uses
seed + 1000 (the same campaigns at every evaluation), `eval` uses seed + 2000 (unseen campaigns).

train --resume <checkpoint> goes on training a saved DQN agent instead of a new one:
    - the agent's learning settings (RESUMABLE_KEYS) stay as saved, unless given on the command
      line or in a --config file; the network shape (hidden, double) cannot change
    - the replay buffer starts empty (checkpoints hold no buffer): no learning until it holds
      learning_starts transitions
    - epsilon follows the agent's own step count, so a finished schedule stays at eps_end; for a
      fixed amount of exploration give the same --eps-start and --eps-end (e.g. 0.1)
    - the loaded agent is validated before the first training campaign (eval.csv campaign 0), so
      best.pt is only replaced by an agent that scores higher
    - use a new --seed: the campaign numbers start at 0 again, so the old seed replays old campaigns

Writes into --out (default runs/campaign/<mode>_<agent>_seed<seed>/, git-ignored):
    config.yaml       the config actually used
    tests.csv         one row per test: blocks, actions, cycles, new/total bins, reward, passed,
                      simulation / report / wall time (+ epsilon and loss while training)
    summary.json      closure rate, tests to close, cycles and wall time per campaign, action counts
    eval.csv          (train) one row per evaluation
    last.pt, best.pt  (train) checkpoints; best = highest closure rate, then fewest tests, then cycles
With --launcher questa every test keeps its folder (plan.txt, sim.log, coverage.txt) under
<questa_work_dir>/tests/<out name>/.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from fifo_rl.agents.random_agent import RandomAgent
from fifo_rl.agents.replay_buffer import ReplayBuffer
from fifo_rl.agents.schedules import make_schedule
from fifo_rl.agents.stub_agent import StubAgent
from fifo_rl.campaign.campaign_runner import CampaignRunner, RunConfig
from fifo_rl.config import add_config_args, config_from_args, explicit_keys, save_config
from fifo_rl.evaluation import GreedyWrapper
from fifo_rl.launchers.fake_launcher import FakeLauncher
from fifo_rl.launchers.questa_launcher import QuestaLauncher
from fifo_rl.plan.actions import ACTION_INDEX, ACTIONS, N_ACTIONS
from scripts.train import make_agent
from fifo_rl.agents.dqn import DQNAgent

VALIDATION_SEED_OFFSET = 1000
EVAL_SEED_OFFSET = 2000
TEST_FIELDS = ("phase", "agent", "reward_mode", "campaign", "test", "seed", "n_blocks", "actions", "forced_end",
               "cycles", "new_bins", "block_new_bins", "total_bins", "reward", "passed", "sim_time", "report_time",
               "wall_time", "epsilon", "loss")
EVAL_FIELDS = ("campaign", "step", "epsilon", "closure_rate", "tests_used_mean", "tests_to_close_mean",
               "cycles_per_campaign_mean", "return_mean", "best")
# settings a resumed agent keeps from its checkpoint unless the user gives them (see the module doc)
RESUMABLE_KEYS = ("lr", "gamma", "batch_size", "learning_starts", "target_sync", "grad_clip", "buffer_capacity",
                  "eps_schedule", "eps_start", "eps_end", "eps_decay_steps")
FIXED_KEYS = ("hidden", "double")  # the network shape: a resumed agent cannot change them


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--agent", choices=("dqn", "random", "baseline", "stub"), default="dqn",
                        help="dqn = DQNAgent (cfg 'double' picks DQN or DDQN); random = random block each step; "
                             "baseline = always BASELINE (constrained random 70/30/10, full plans); stub = no learning")
    parser.add_argument("--checkpoint", type=Path, default=None, help="eval --agent dqn: a best.pt / last.pt")
    parser.add_argument("--resume", type=Path, default=None,
                        help="train --agent dqn: go on training this best.pt / last.pt instead of a new agent")
    parser.add_argument("--launcher", choices=("fake", "questa"), default="fake")
    parser.add_argument("--out", type=Path, default=None, help="default runs/campaign/<mode>_<agent>_seed<seed>")
    add_config_args(parser)
    return parser.parse_args(argv)


def make_launcher(kind, cfg):
    """Input: "fake" or "questa", config. Output: the launcher (Questa: compiled once here)."""
    if kind == "fake":
        return FakeLauncher(per_action=cfg["reward_mode"] == "action")
    launcher = QuestaLauncher(cfg["questa_work_dir"], cfg["questa_src_list"], test_name=cfg["questa_test"],
                              top=cfg["questa_top"], questa_bin=cfg["questa_bin"] or "",
                              tool_suffix=cfg["questa_tool_suffix"] or "", timeout_s=cfg["questa_timeout_s"],
                              per_action=cfg["reward_mode"] == "action")
    launcher.compile()
    return launcher


def make_runner(cfg, launcher, name, seed_offset):
    """Input: config, launcher, a folder name, the seed offset of this phase. Output: a CampaignRunner.

    Logic: the fake FIFO reuses one folder per phase; Questa keeps one folder per test, inside its
           work dir (vsim runs there and gets relative paths).
    """
    if isinstance(launcher, QuestaLauncher):
        run_root, keep = launcher.work_dir / "tests" / name, True
    else:
        run_root, keep = Path("runs/campaign/sims") / name, False
    config = RunConfig.from_cfg(cfg, base_seed=cfg["seed"] + seed_offset, run_root=run_root, keep_runs=keep)
    return CampaignRunner(config, launcher)


def row_for_test(phase, agent_name, summary, epsilon=None, losses=()):
    """Input: phase name, agent name, a TestSummary, epsilon, the test's learn() losses. Output: a CSV row."""
    plan, out = summary.plan, summary.output
    return {"phase": phase, "agent": agent_name, "reward_mode": summary.attribution.mode,
            "campaign": summary.campaign, "test": summary.test_no,
            "seed": summary.seed, "n_blocks": plan.n_blocks,
            "actions": " ".join(pa.spec.name for pa in plan.actions),
            "forced_end": int(plan.actions[-1].forced), "cycles": plan.sim_cycles(),
            "new_bins": summary.attribution.new_bins,
            "block_new_bins": " ".join(str(n) for n in summary.attribution.block_new_bins),
            "total_bins": summary.total_bins,
            "reward": round(sum(summary.attribution.rewards), 6), "passed": int(summary.result.passed),
            "sim_time": round(out.sim_time, 4), "report_time": round(out.report_time, 4),
            "wall_time": round(out.wall_time, 4), "epsilon": "" if epsilon is None else round(epsilon, 4),
            "loss": round(statistics.fmean(losses), 6) if losses else ""}


def _stats(values):
    return {"mean": statistics.fmean(values), "median": statistics.median(values)} if values else None


def summarize(results):
    """Input: a list of CampaignResult. Output: dict of campaign-level numbers (see the module doc)."""
    closed = [r for r in results if r.closed]
    tests = [t for r in results for t in r.tests]
    counts = Counter(pa.spec.name for t in tests for pa in t.plan.actions)
    return {
        "campaigns": len(results),
        "closure_rate": len(closed) / len(results) if results else 0.0,
        "tests_used": _stats([r.tests_used for r in results]),
        "tests_to_close": _stats([r.tests_used for r in closed]),
        "cycles_per_campaign": _stats([r.total_cycles for r in results]),
        "cycles_to_close": _stats([r.total_cycles for r in closed]),
        "final_bins": _stats([r.final_bins for r in results]),
        "return": _stats([r.episode_return for r in results]),
        "blocks_per_test": _stats([t.plan.n_blocks for t in tests]),
        "wall_time_per_test_s": _stats([t.output.wall_time for t in tests]),
        "sim_time_per_test_s": _stats([t.output.sim_time for t in tests]),
        "wall_time_per_campaign_s": _stats([r.wall_time for r in results]),
        "action_counts": {spec.name: counts.get(spec.name, 0) for spec in ACTIONS},
    }


def play(runner, agent, n_campaigns, writer, phase, agent_name):
    """Input: runner, agent, number of campaigns, a csv writer, phase and agent names.
    Output: list of CampaignResult (no learning; the agent plays as given).
    """
    results = []
    for _ in range(n_campaigns):
        results.append(runner.run_campaign(
            agent, on_test=lambda s, tr, ls: writer.writerow(row_for_test(phase, agent_name, s))))
    return results


def score(summary):
    """Input: a summary. Output: sort key, larger = better (closure, then fewer tests, then fewer cycles)."""
    tests = summary["tests_used"]["mean"] if summary["tests_used"] else float("inf")
    cycles = summary["cycles_per_campaign"]["mean"] if summary["cycles_per_campaign"] else float("inf")
    return (summary["closure_rate"], -tests, -cycles)


def resume_agent(path, cfg, state_size, n_actions, explicit=()):
    """Load a saved DQN agent so that training can go on.

    Input:  path, a best.pt / last.pt; cfg, the config of this run; state_size, n_actions, the
            env sizes; explicit, the config keys the user gave (config.explicit_keys).
    Output: the DQNAgent with the saved networks, optimizer and step counters, an empty replay
            buffer and its exploration schedule.
    Logic:  1. the checkpoint must fit this env (no 22-action checkpoints) and keep its network shape
            2. the RESUMABLE_KEYS the user gave replace the saved ones (lr also inside the optimizer)
            3. new buffer, schedule and exploration generator from the merged settings and this seed
    """
    agent = DQNAgent.load(path)
    if (agent.state_dim, agent.n_actions) != (state_size, n_actions):
        raise SystemExit(f"{path} has a {agent.state_dim}-number state and {agent.n_actions} actions, "
                         f"this run needs {state_size} and {n_actions}")
    changed = [k for k in FIXED_KEYS if k in explicit and cfg[k] != agent.cfg[k]]
    if changed:
        raise SystemExit(f"--resume cannot change the network shape ({', '.join(changed)}): "
                         f"the checkpoint has {', '.join(f'{k}={agent.cfg[k]}' for k in changed)}")
    agent.cfg.update({k: cfg[k] for k in RESUMABLE_KEYS if k in explicit})
    for group in agent.optimizer.param_groups:
        group["lr"] = agent.cfg["lr"]
    agent.buffer = ReplayBuffer(agent.cfg["buffer_capacity"], state_size, seed=cfg["seed"])
    agent.epsilon_schedule = make_schedule(agent.cfg)
    agent.rng = np.random.default_rng(cfg["seed"])
    return agent


def train(cfg, out, agent_cls, launcher_kind="fake", verbose=True, resume=None, explicit=()):
    """Train on campaigns; greedy validation every eval_every campaigns.

    Input:  cfg, output folder, agent class (DQNAgent or StubAgent), launcher kind, verbose;
            resume, a checkpoint to go on training (then agent_cls is not used); explicit, the
            config keys the user gave (see resume_agent).
    Output: dict with the best summary, the number of agent steps and the eval rows.
    Logic:  a resumed agent is validated once before the first training campaign (campaign 0), so
            best.pt never gets an agent that scores lower than the one loaded.
    """
    out = Path(out)
    if resume is not None and Path(resume).resolve().parent == out.resolve():
        raise SystemExit(f"--out {out} holds the checkpoint being resumed; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)
    launcher = make_launcher(launcher_kind, cfg)
    runner = make_runner(cfg, launcher, f"{out.name}_train", 0)
    if resume is None:
        agent = make_agent(agent_cls, cfg, runner.state_size, runner.n_actions)
        resumed = {}
    else:
        agent = resume_agent(resume, cfg, runner.state_size, runner.n_actions, explicit)
        cfg = {**cfg, **{k: agent.cfg[k] for k in RESUMABLE_KEYS + FIXED_KEYS}}  # config.yaml = what is used
        resumed = {"resumed_from": str(resume), "resumed_at_step": agent.steps}
    save_config(cfg, out / "config.yaml")
    best, best_summary, eval_rows = None, None, []
    with open(out / "tests.csv", "w", newline="") as f_tests, open(out / "eval.csv", "w", newline="") as f_eval:
        tests_writer = csv.DictWriter(f_tests, fieldnames=TEST_FIELDS)
        eval_writer = csv.DictWriter(f_eval, fieldnames=EVAL_FIELDS)
        tests_writer.writeheader()
        eval_writer.writeheader()

        def validate(campaign):
            """Input: campaigns trained so far. Output: nothing; validates, saves, logs one eval row."""
            nonlocal best, best_summary
            val_runner = make_runner(cfg, launcher, f"{out.name}_val", VALIDATION_SEED_OFFSET)
            summary = summarize(play(val_runner, GreedyWrapper(agent), cfg["eval_campaigns"],
                                     tests_writer, "validation", agent.name))
            is_best = best is None or score(summary) > best
            if is_best:
                best, best_summary = score(summary), summary
                agent.save(out / "best.pt")
                (out / "summary.json").write_text(json.dumps(
                    {"agent": agent.name, "reward_mode": cfg["reward_mode"], "seed": cfg["seed"],
                         "campaign": campaign, **resumed, **summary}, indent=1))
            agent.save(out / "last.pt")
            row = {"campaign": campaign, "step": agent.steps, "epsilon": round(agent.epsilon, 4),
                   "closure_rate": summary["closure_rate"],
                   "tests_used_mean": summary["tests_used"]["mean"],
                   "tests_to_close_mean": summary["tests_to_close"]["mean"] if summary["tests_to_close"] else "",
                   "cycles_per_campaign_mean": summary["cycles_per_campaign"]["mean"],
                   "return_mean": summary["return"]["mean"], "best": int(is_best)}
            eval_writer.writerow(row)
            f_eval.flush()
            f_tests.flush()
            eval_rows.append(row)
            if verbose:
                print(f"campaign {campaign:5d}  step {agent.steps:7d}  eps {agent.epsilon:.3f}  "
                      f"closes {100 * row['closure_rate']:3.0f}%  tests {row['tests_used_mean']:5.2f}  "
                      f"cycles {row['cycles_per_campaign_mean']:6.1f}" + ("  *best" if is_best else ""))

        if resume is not None:
            validate(0)  # the score a resumed run has to beat
        for campaign in range(cfg["n_campaigns"]):
            runner.run_campaign(agent, train=True, train_freq=cfg["train_freq"], on_test=lambda s, tr, ls: (
                tests_writer.writerow(row_for_test("train", agent.name, s, agent.epsilon, ls))))
            if (campaign + 1) % cfg["eval_every"] == 0 or campaign + 1 == cfg["n_campaigns"]:
                validate(campaign + 1)
    return {"best": best_summary, "steps": agent.steps, "eval": eval_rows}


def evaluate(cfg, out, agent, launcher_kind="fake"):
    """Play eval_campaigns unseen campaigns without learning.

    Input:  cfg, output folder, the agent (already greedy-wrapped if it learns), launcher kind.
    Output: the summary dict (also written to summary.json).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    runner = make_runner(cfg, make_launcher(launcher_kind, cfg), out.name, EVAL_SEED_OFFSET)
    with open(out / "tests.csv", "w", newline="") as f_tests:
        writer = csv.DictWriter(f_tests, fieldnames=TEST_FIELDS)
        writer.writeheader()
        summary = summarize(play(runner, agent, cfg["eval_campaigns"], writer, "eval", agent.name))
    summary = {"agent": agent.name, "launcher": launcher_kind, "reward_mode": cfg["reward_mode"], "seed": cfg["seed"],
               **summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


class BaselineAgent:
    """Constrained-random baseline: always BASELINE (70/30/10, the FIFO_test knobs); never ends a plan
    itself, so every test is a full plan (plan_max_len blocks, END_TEST forced)."""
    name = "baseline"

    def act(self, state):
        return ACTION_INDEX["BASELINE"]


def eval_agent(kind, checkpoint, cfg):
    """Input: agent kind, checkpoint path, config. Output: an agent that plays without learning."""
    if kind == "random":
        return RandomAgent(n_actions=N_ACTIONS, seed=cfg["seed"])
    if kind == "baseline":
        return BaselineAgent()
    if kind == "stub":
        return GreedyWrapper(StubAgent(n_actions=N_ACTIONS, seed=cfg["seed"]))
    if checkpoint is None:
        raise SystemExit("eval --agent dqn needs --checkpoint <best.pt>")
    return GreedyWrapper(DQNAgent.load(checkpoint))


def mode_tag(cfg):
    """Input: config. Output: "" for step 1, "_action" for step 2 (kept in folder names)."""
    return "_action" if cfg["reward_mode"] == "action" else ""


def refuse_overwrite(out):
    """Input: an output folder. Output: nothing; stops if it already holds results (never overwrite a run)."""
    if (Path(out) / "summary.json").exists():
        raise SystemExit(f"{out} already holds results: pick another --out (results are never overwritten)")


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from_args(args)
    if args.resume is not None and (args.mode != "train" or args.agent != "dqn"):
        raise SystemExit("--resume goes on training a DQN checkpoint: use it with 'train --agent dqn'")
    if args.mode == "train":
        if args.agent == "random":
            raise SystemExit("a random agent does not learn: use 'eval --agent random'")
        if args.agent == "stub":
            cls, name = StubAgent, "stub"
        else:
            cls, name = DQNAgent, ("ddqn" if cfg["double"] else "dqn")
        default_out = (f"resume{mode_tag(cfg)}_seed{cfg['seed']}" if args.resume
                       else f"train_{name}{mode_tag(cfg)}_seed{cfg['seed']}")
        out = args.out or Path("runs/campaign") / default_out
        refuse_overwrite(out)
        result = train(cfg, out, cls, args.launcher, resume=args.resume, explicit=explicit_keys(args))
        best = result["best"]
        print(f"\nbest: closes {100 * best['closure_rate']:.0f}%, tests used {best['tests_used']['mean']:.2f}"
              f" (budget {cfg['campaign_budget']}), {result['steps']} agent steps; written to {out}/")
    else:
        agent = eval_agent(args.agent, args.checkpoint, cfg)
        out = args.out or Path("runs/campaign") / f"eval_{agent.name}{mode_tag(cfg)}_{args.launcher}_seed{cfg['seed']}"
        refuse_overwrite(out)
        s = evaluate(cfg, out, agent, args.launcher)
        to_close = f"{s['tests_to_close']['mean']:.2f}" if s["tests_to_close"] else "-"
        print(f"{agent.name} on {args.launcher}: closes {100 * s['closure_rate']:.0f}% of {s['campaigns']} campaigns, "
              f"tests to close {to_close}, cycles per campaign {s['cycles_per_campaign']['mean']:.1f}, "
              f"wall time per test {s['wall_time_per_test_s']['mean']:.3f} s; written to {out}/")


if __name__ == "__main__":
    main()
