"""Plan-based campaigns: train DQN / DDQN on the fake FIFO, evaluate on the fake FIFO or real Questa.

Run from the fifo_rl/ folder:
    python -m scripts.run_campaign train --gamma 0.99 --eps-decay-steps 60000      # DDQN, fake FIFO
    python -m scripts.run_campaign eval --agent dqn --checkpoint runs/campaign/train_ddqn_seed0/best.pt
    python -m scripts.run_campaign eval --agent random                              # baseline
    python -m scripts.run_campaign eval --agent dqn --checkpoint <best.pt> --launcher questa --eval-campaigns 3

One test = one simulation of a plan of blocks; one campaign = up to campaign_budget tests = one
DQN episode. Seeds: training campaigns use base seed `seed`, the validation inside `train` uses
seed + 1000 (the same campaigns at every evaluation), `eval` uses seed + 2000 (unseen campaigns).

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

from fifo_rl.agents.random_agent import RandomAgent
from fifo_rl.agents.stub_agent import StubAgent
from fifo_rl.campaign.campaign_runner import CampaignRunner, RunConfig
from fifo_rl.config import add_config_args, config_from_args, save_config
from fifo_rl.evaluation import GreedyWrapper
from fifo_rl.launchers.fake_launcher import FakeLauncher
from fifo_rl.launchers.questa_launcher import QuestaLauncher
from fifo_rl.plan.actions import ACTIONS, N_ACTIONS
from scripts.train import make_agent

VALIDATION_SEED_OFFSET = 1000
EVAL_SEED_OFFSET = 2000
TEST_FIELDS = ("phase", "agent", "campaign", "test", "seed", "n_blocks", "actions", "forced_end", "cycles",
               "new_bins", "total_bins", "reward", "passed", "sim_time", "report_time", "wall_time",
               "epsilon", "loss")
EVAL_FIELDS = ("campaign", "step", "epsilon", "closure_rate", "tests_used_mean", "tests_to_close_mean",
               "cycles_per_campaign_mean", "return_mean", "best")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--agent", choices=("dqn", "random", "stub"), default="dqn",
                        help="dqn = DQNAgent (cfg 'double' picks DQN or DDQN); random = baseline; stub = no learning")
    parser.add_argument("--checkpoint", type=Path, default=None, help="eval --agent dqn: a best.pt / last.pt")
    parser.add_argument("--launcher", choices=("fake", "questa"), default="fake")
    parser.add_argument("--out", type=Path, default=None, help="default runs/campaign/<mode>_<agent>_seed<seed>")
    add_config_args(parser)
    return parser.parse_args(argv)


def make_launcher(kind, cfg):
    """Input: "fake" or "questa", config. Output: the launcher (Questa: compiled once here)."""
    if kind == "fake":
        return FakeLauncher()
    launcher = QuestaLauncher(cfg["questa_work_dir"], cfg["questa_src_list"], test_name=cfg["questa_test"],
                              top=cfg["questa_top"], questa_bin=cfg["questa_bin"] or "",
                              tool_suffix=cfg["questa_tool_suffix"] or "", timeout_s=cfg["questa_timeout_s"])
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
    return {"phase": phase, "agent": agent_name, "campaign": summary.campaign, "test": summary.test_no,
            "seed": summary.seed, "n_blocks": plan.n_blocks,
            "actions": " ".join(pa.spec.name for pa in plan.actions),
            "forced_end": int(plan.actions[-1].forced), "cycles": plan.sim_cycles(),
            "new_bins": summary.attribution.new_bins, "total_bins": summary.total_bins,
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


def train(cfg, out, agent_cls, launcher_kind="fake", verbose=True):
    """Train on campaigns; greedy validation every eval_every campaigns.

    Input:  cfg, output folder, agent class (DQNAgent or StubAgent), launcher kind, verbose.
    Output: dict with the best summary, the number of agent steps and the eval rows.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    launcher = make_launcher(launcher_kind, cfg)
    runner = make_runner(cfg, launcher, f"{out.name}_train", 0)
    agent = make_agent(agent_cls, cfg, runner.state_size, runner.n_actions)
    best, best_summary, eval_rows = None, None, []
    with open(out / "tests.csv", "w", newline="") as f_tests, open(out / "eval.csv", "w", newline="") as f_eval:
        tests_writer = csv.DictWriter(f_tests, fieldnames=TEST_FIELDS)
        eval_writer = csv.DictWriter(f_eval, fieldnames=EVAL_FIELDS)
        tests_writer.writeheader()
        eval_writer.writeheader()
        for campaign in range(cfg["n_campaigns"]):
            runner.run_campaign(agent, train=True, train_freq=cfg["train_freq"], on_test=lambda s, tr, ls: (
                tests_writer.writerow(row_for_test("train", agent.name, s, agent.epsilon, ls))))
            if (campaign + 1) % cfg["eval_every"] == 0 or campaign + 1 == cfg["n_campaigns"]:
                val_runner = make_runner(cfg, launcher, f"{out.name}_val", VALIDATION_SEED_OFFSET)
                summary = summarize(play(val_runner, GreedyWrapper(agent), cfg["eval_campaigns"],
                                         tests_writer, "validation", agent.name))
                is_best = best is None or score(summary) > best
                if is_best:
                    best, best_summary = score(summary), summary
                    agent.save(out / "best.pt")
                    (out / "summary.json").write_text(json.dumps(
                        {"agent": agent.name, "seed": cfg["seed"], "campaign": campaign + 1, **summary}, indent=1))
                agent.save(out / "last.pt")
                row = {"campaign": campaign + 1, "step": agent.steps, "epsilon": round(agent.epsilon, 4),
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
                    print(f"campaign {campaign + 1:5d}  step {agent.steps:7d}  eps {agent.epsilon:.3f}  "
                          f"closes {100 * row['closure_rate']:3.0f}%  tests {row['tests_used_mean']:5.2f}  "
                          f"cycles {row['cycles_per_campaign_mean']:6.1f}" + ("  *best" if is_best else ""))
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
    summary = {"agent": agent.name, "launcher": launcher_kind, "seed": cfg["seed"], **summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def eval_agent(kind, checkpoint, cfg):
    """Input: agent kind, checkpoint path, config. Output: an agent that plays without learning."""
    if kind == "random":
        return RandomAgent(n_actions=N_ACTIONS, seed=cfg["seed"])
    if kind == "stub":
        return GreedyWrapper(StubAgent(n_actions=N_ACTIONS, seed=cfg["seed"]))
    if checkpoint is None:
        raise SystemExit("eval --agent dqn needs --checkpoint <best.pt>")
    from fifo_rl.agents.dqn import DQNAgent
    return GreedyWrapper(DQNAgent.load(checkpoint))


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from_args(args)
    if args.mode == "train":
        if args.agent == "random":
            raise SystemExit("a random agent does not learn: use 'eval --agent random'")
        if args.agent == "stub":
            cls, name = StubAgent, "stub"
        else:
            from fifo_rl.agents.dqn import DQNAgent
            cls, name = DQNAgent, ("ddqn" if cfg["double"] else "dqn")
        out = args.out or Path("runs/campaign") / f"train_{name}_seed{cfg['seed']}"
        result = train(cfg, out, cls, args.launcher)
        best = result["best"]
        print(f"\nbest: closes {100 * best['closure_rate']:.0f}%, tests used {best['tests_used']['mean']:.2f}"
              f" (budget {cfg['campaign_budget']}), {result['steps']} agent steps; written to {out}/")
    else:
        agent = eval_agent(args.agent, args.checkpoint, cfg)
        out = args.out or Path("runs/campaign") / f"eval_{agent.name}_{args.launcher}_seed{cfg['seed']}"
        s = evaluate(cfg, out, agent, args.launcher)
        to_close = f"{s['tests_to_close']['mean']:.2f}" if s["tests_to_close"] else "-"
        print(f"{agent.name} on {args.launcher}: closes {100 * s['closure_rate']:.0f}% of {s['campaigns']} campaigns, "
              f"tests to close {to_close}, cycles per campaign {s['cycles_per_campaign']['mean']:.1f}, "
              f"wall time per test {s['wall_time_per_test_s']['mean']:.3f} s; written to {out}/")


if __name__ == "__main__":
    main()
