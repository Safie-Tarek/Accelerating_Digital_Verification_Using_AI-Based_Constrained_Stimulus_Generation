"""Train the DQN / DDQN agent on the fake FIFO.

Run from the fifo_rl/ folder:
    python -m scripts.train                                   # DDQN, configs/default.yaml
    python -m scripts.train --double false --seed 3           # vanilla DQN, another seed
    python -m scripts.train --agent stub --n-episodes 50      # plumbing check, no learning

Loop: act -> env.step -> observe -> learn (every train_freq steps) on
CoverageEnv(FakeBackend(), budget=budget_train). Every eval_every episodes the agent plays
eval_episodes greedy episodes on a separate budget_eval env. That validation env always replays
the SAME episodes (fresh env, base seed + VALIDATION_SEED_OFFSET), so two evaluations differ only
because the agent changed, and its seeds differ from the test seeds scripts/evaluate.py uses.

Writes into --out (default runs/train/<agent>_seed<seed>/, git-ignored):
    config.yaml            the config actually used
    steps.csv              one row per training step (fifo_rl/utils/logger.py format, with epsilon, loss)
    eval.csv               one row per evaluation (closure rate, runs used, coverage AUC, ...)
    last.pt, best.pt       checkpoints; best = highest closure rate, then fewest runs used
    <agent>_summary.json   summarize() of the best evaluation, the format scripts/plot.py reads
"""

import argparse
import csv
import json
from pathlib import Path

from fifo_rl.agents.replay_buffer import ReplayBuffer
from fifo_rl.agents.schedules import make_schedule
from fifo_rl.agents.stub_agent import StubAgent
from fifo_rl.backends.fake_backend import FakeBackend
from fifo_rl.config import add_config_args, config_from_args, save_config
from fifo_rl.envs.coverage_env import CoverageEnv
from fifo_rl.evaluation import GreedyWrapper, run_episodes, summarize
from fifo_rl.utils.logger import CsvLogger

VALIDATION_SEED_OFFSET = 1000
EVAL_FIELDS = ("episode", "step", "epsilon", "closure_rate", "runs_to_close_mean", "runs_used_mean",
               "final_bins_mean", "return_mean", "coverage_auc", "best")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=("dqn", "stub"), default="dqn",
                        help="dqn = DQNAgent (cfg 'double' picks DQN or DDQN); stub = no learning")
    parser.add_argument("--out", type=Path, default=None, help="default runs/train/<agent>_seed<seed>")
    add_config_args(parser)
    return parser.parse_args(argv)


def agent_class(kind):
    """Input: "dqn" or "stub". Output: the agent class (dqn imported only when asked for)."""
    if kind == "stub":
        return StubAgent
    from fifo_rl.agents.dqn import DQNAgent
    return DQNAgent


def make_agent(cls, cfg, state_dim, n_actions):
    """Input: agent class, config, env sizes. Output: the agent with its own buffer and schedule."""
    buffer = ReplayBuffer(cfg["buffer_capacity"], state_dim, seed=cfg["seed"])
    return cls(state_dim=state_dim, n_actions=n_actions, cfg=cfg, buffer=buffer,
               epsilon_schedule=make_schedule(cfg), seed=cfg["seed"])


def evaluate(agent, cfg):
    """Input: agent, config. Output: summarize() dict of eval_episodes greedy episodes.

    Logic: a fresh validation env every time, so every evaluation plays the same episodes.
    """
    env = CoverageEnv(FakeBackend(), budget=cfg["budget_eval"], seed=cfg["seed"] + VALIDATION_SEED_OFFSET)
    results = run_episodes(env, GreedyWrapper(agent), cfg["eval_episodes"])
    return summarize(results, cfg["budget_eval"], env.n_actions)


def score(summary):
    """Input: a summary. Output: a sort key, larger = better (closure rate, then fewer runs used)."""
    return (summary["closure_rate"], -summary["runs_used"]["mean"])


def train(cfg, out, agent_cls=None, verbose=True):
    """Run one training.

    Input:  cfg, config dict; out, output folder; agent_cls, the agent class (default DQNAgent);
            verbose, print one line per evaluation.
    Output: dict with the best summary, the number of steps and the eval rows.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    env = CoverageEnv(FakeBackend(), budget=cfg["budget_train"], seed=cfg["seed"])
    agent = make_agent(agent_cls or agent_class("dqn"), cfg, env.state_size, env.n_actions)

    best, best_summary, eval_rows, step = None, None, [], 0
    with CsvLogger(out / "steps.csv") as logger, open(out / "eval.csv", "w", newline="") as eval_file:
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_FIELDS)
        eval_writer.writeheader()
        for episode in range(cfg["n_episodes"]):
            state, done, t = env.reset(), False, 0
            while not done:
                action_id = agent.act(state)
                next_state, reward, done, info = env.step(action_id)
                agent.observe(state, action_id, reward, next_state, done)
                loss = agent.learn() if step % cfg["train_freq"] == 0 else None
                logger.log(agent=agent.name, episode=env.episode, step=t, action_id=action_id,
                           reward=reward, epsilon=agent.epsilon, loss=loss, **info)
                state, t, step = next_state, t + 1, step + 1

            if (episode + 1) % cfg["eval_every"] == 0 or episode + 1 == cfg["n_episodes"]:
                summary = evaluate(agent, cfg)
                is_best = best is None or score(summary) > best
                if is_best:
                    best, best_summary = score(summary), summary
                    agent.save(out / "best.pt")
                    (out / f"{agent.name}_summary.json").write_text(json.dumps(
                        {"agent": agent.name, "seed": cfg["seed"], "episode": episode + 1, **summary}, indent=1))
                row = {"episode": episode + 1, "step": step, "epsilon": agent.epsilon,
                       "closure_rate": summary["closure_rate"],
                       "runs_to_close_mean": summary["runs_to_close"]["mean"],
                       "runs_used_mean": summary["runs_used"]["mean"],
                       "final_bins_mean": summary["final_bins"]["mean"],
                       "return_mean": summary["return"]["mean"],
                       "coverage_auc": summary["coverage_auc"], "best": int(is_best)}
                eval_writer.writerow(row)
                eval_file.flush()
                eval_rows.append(row)
                agent.save(out / "last.pt")
                if verbose:
                    print(f"episode {episode + 1:5d}  step {step:7d}  eps {agent.epsilon:.3f}  "
                          f"closes {100 * row['closure_rate']:3.0f}%  runs used {row['runs_used_mean']:5.2f}  "
                          f"AUC {row['coverage_auc']:.3f}" + ("  *best" if is_best else ""))
    return {"best": best_summary, "steps": step, "eval": eval_rows}


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from_args(args)
    cls = agent_class(args.agent)
    name = cls.name if args.agent == "stub" else ("ddqn" if cfg["double"] else "dqn")
    out = args.out or Path("runs/train") / f"{name}_seed{cfg['seed']}"
    result = train(cfg, out, cls)
    best = result["best"]
    print(f"\nbest: closes {100 * best['closure_rate']:.0f}%, runs used {best['runs_used']['mean']:.2f}"
          f" (budget {cfg['budget_eval']}), {result['steps']} training steps; written to {out}/")


if __name__ == "__main__":
    main()
