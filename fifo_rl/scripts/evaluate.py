"""Measure the random baselines on the fake FIFO.

Run from the fifo_rl/ folder:
    python -m scripts.evaluate                                   # everything, default settings
    python -m scripts.evaluate --agents random --episodes 200 --budget 40

Baselines:
    random        (i)    a random one of the 22 actions every run
    fixed         (ii-a) always baseline_random (id 21): 70/30/10 random, 18 cycles per run
    long random   (ii-b) one continuous 70/30/10 run like FIFO_test: cycles until 64/64

Writes into --out (default runs/baselines/, git-ignored):
    <agent>_steps.csv      one row per run (fifo_rl/utils/logger.py format)
    <agent>_summary.json   metrics + coverage curve (fifo_rl/evaluation.py summarize)
    long_random.json       baseline (ii-b)
"""

import argparse
import json
from pathlib import Path

from fifo_rl.agents.random_agent import FixedActionAgent, RandomAgent
from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.envs.coverage_env import CoverageEnv
from fifo_rl.evaluation import long_random_cycles_to_close, run_episodes, summarize
from fifo_rl.utils.logger import CsvLogger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", choices=("random", "fixed"), default=["random", "fixed"])
    parser.add_argument("--action", type=int, default=21, help="action of the fixed agent (default 21)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--budget", type=int, default=40, help="runs per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--long-seeds", type=int, default=100, help="seeds for baseline (ii-b); 0 = skip")
    parser.add_argument("--out", type=Path, default=Path("runs/baselines"))
    return parser.parse_args(argv)


def make_agent(kind, args):
    """Input: "random" or "fixed", the parsed args. Output: the agent."""
    if kind == "random":
        return RandomAgent(seed=args.seed)
    return FixedActionAgent(args.action)


def evaluate_agent(kind, args):
    """Play the episodes of one agent, write its CSV and summary, return the summary.

    Logic:  every agent gets a fresh env with the SAME base seed, so episode e / run s uses the
            same simulation seed for every agent ("common random numbers"): differences between
            agents then come from their choices, not from luck of the seeds.
    """
    agent = make_agent(kind, args)
    env = CoverageEnv(FakeBackend(), budget=args.budget, seed=args.seed)
    with CsvLogger(args.out / f"{agent.name}_steps.csv") as logger:
        results = run_episodes(env, agent, args.episodes, logger)
    summary = {"agent": agent.name, "seed": args.seed, **summarize(results, args.budget, env.n_actions)}
    (args.out / f"{agent.name}_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _fmt(stats, digits=1):
    """Input: a _stats dict. Output: "mean ± std (median m)" or "-" when empty."""
    if stats["mean"] is None:
        return "-"
    return f"{stats['mean']:.{digits}f} +/- {stats['std']:.{digits}f} (median {stats['median']:.{digits}f})"


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"fake FIFO, {args.episodes} episodes, budget {args.budget} runs, seed {args.seed}\n")

    for kind in args.agents:
        s = evaluate_agent(kind, args)
        print(f"[{s['agent']}]")
        print(f"  closes 64/64 within budget : {100 * s['closure_rate']:.0f}% of episodes")
        print(f"  runs to close (closed only): {_fmt(s['runs_to_close'])}")
        print(f"  runs used (unclosed=budget): {_fmt(s['runs_used'])}")
        print(f"  bins at the end            : {_fmt(s['final_bins'])}")
        print(f"  return                     : {_fmt(s['return'], 3)}")
        print(f"  bins after run 1 / 5 / 10  : "
              + " / ".join(f"{s['curve_mean'][k]:.1f}" for k in (0, 4, 9) if k < args.budget) + "\n")

    if args.long_seeds > 0:
        long = {"pure_random": long_random_cycles_to_close(args.long_seeds, args.seed, directed_prefix=False),
                "fifo_test": long_random_cycles_to_close(args.long_seeds, args.seed, directed_prefix=True)}
        (args.out / "long_random.json").write_text(json.dumps(long, indent=1))
        print(f"[long random run, {args.long_seeds} seeds] cycles in ONE simulation until 64/64")
        for name, res in long.items():
            print(f"  {name:12s}: {_fmt(res['cycles_to_close'], 0)}, 95% of seeds by {res['p95']:.0f},"
                  f" closes in {100 * res['closure_rate']:.0f}%")
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
