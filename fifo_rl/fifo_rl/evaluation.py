"""Play episodes with any agent and measure how fast it closes coverage.

Used by scripts/evaluate.py for the baselines and, later, by train.py to evaluate the DQN the
same way, so every agent is measured with exactly the same code.

Metrics (per agent, over many episodes):
    closure_rate        share of episodes that reach all 64 bins within the budget
    runs_to_close       runs needed, over the episodes that closed (mean / std / median)
    runs_used           runs needed, counting an unclosed episode as the full budget; a lower
                        bound on the true mean that, unlike runs_to_close, does not flatter an
                        agent by dropping its failures
    final_bins          bins covered when the episode ended
    return              sum of rewards (the number the DQN is trained to maximise)
    curve               bins covered after run 1, 2, ..., budget (an episode that closed early
                        stays at 64), averaged over episodes -> the "coverage vs runs" plot

Also here: the "one long constrained-random run" baseline (long_random_cycles_to_close), which
answers how many cycles plain random needs in ONE simulation - the fairness check for the
whole FIFO pilot, since a Questa run costs ~3 s whatever its length.
"""

from dataclasses import dataclass, field

import numpy as np

from fifo_rl import fake_fifo
from fifo_rl.actions import RESET_CYCLE, get_action
from fifo_rl.coverage_model import N_BINS


@dataclass
class EpisodeResult:
    """Everything one episode produced, one list entry per run."""
    actions: list = field(default_factory=list)      # action id of every run
    new_bins: list = field(default_factory=list)     # bins that run added
    total_bins: list = field(default_factory=list)   # bins covered after that run
    rewards: list = field(default_factory=list)
    closed: bool = False                             # reached all bins within the budget

    @property
    def runs(self):
        return len(self.actions)

    @property
    def episode_return(self):
        return float(sum(self.rewards))

    @property
    def final_bins(self):
        return self.total_bins[-1] if self.total_bins else 0


def run_episode(env, agent, logger=None):
    """Play one episode.

    Input:  env, a CoverageEnv; agent, anything with act(state) -> action id and a name;
            logger, an optional CsvLogger (one row per run).
    Output: an EpisodeResult.
    Logic:  reset the env, then ask the agent for an action and step until the env says done
            (all bins covered or budget used). The episode closed if the last run's info says
            all bins are covered.
    """
    result = EpisodeResult()
    state = env.reset()
    done = False
    info = {}
    while not done:
        action_id = agent.act(state)
        state, reward, done, info = env.step(action_id)
        result.actions.append(action_id)
        result.new_bins.append(info["new_bins"])
        result.total_bins.append(info["total_bins"])
        result.rewards.append(reward)
        if logger is not None:
            logger.log(agent=agent.name, episode=env.episode, step=result.runs - 1,
                       action_id=action_id, reward=reward, **info)
    result.closed = bool(info.get("all_covered", False))
    return result


def run_episodes(env, agent, n_episodes, logger=None):
    """Input: env, agent, number of episodes, optional logger. Output: list of EpisodeResult."""
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
    return [run_episode(env, agent, logger) for _ in range(n_episodes)]


def coverage_curve(results, budget):
    """Bins covered after each run, one row per episode.

    Input:  results, a list of EpisodeResult; budget, runs per episode.
    Output: int array (n_episodes, budget); column k = bins covered after run k+1.
    Logic:  copy each episode's total_bins and repeat its last value up to the budget, so an
            episode that closed after 5 runs reads 64 from run 5 on (coverage cannot go down).
    """
    curve = np.zeros((len(results), budget), dtype=np.int64)
    for i, res in enumerate(results):
        if res.runs > budget:
            raise ValueError(f"episode {i} has {res.runs} runs, more than the budget {budget}")
        curve[i, : res.runs] = res.total_bins
        curve[i, res.runs:] = res.final_bins
    return curve


def _stats(values):
    """Input: list of numbers. Output: dict mean/std/median/min/max, or Nones if the list is empty."""
    if len(values) == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize(results, budget, n_actions):
    """All metrics of one agent, as a JSON-ready dict.

    Input:  results, a list of EpisodeResult; budget, runs per episode; n_actions (22).
    Output: dict with n_episodes, budget, closure_rate, runs_to_close, runs_to_close_all,
            runs_used, final_bins, return, curve_mean, curve_std, action_counts.
    Logic:  see the module docstring for what each metric means and why it is there.
    """
    closed_runs = [res.runs for res in results if res.closed]
    curve = coverage_curve(results, budget)
    counts = np.bincount([a for res in results for a in res.actions], minlength=n_actions)
    return {
        "n_episodes": len(results),
        "budget": budget,
        "closure_rate": len(closed_runs) / len(results),
        "runs_to_close": _stats(closed_runs),
        "runs_to_close_all": closed_runs,
        "runs_used": _stats([res.runs for res in results]),
        "final_bins": _stats([res.final_bins for res in results]),
        "return": _stats([res.episode_return for res in results]),
        "curve_mean": curve.mean(axis=0).tolist(),
        "curve_std": curve.std(axis=0).tolist(),
        "action_counts": counts.tolist(),
    }


# ---------------------------------------------------------------------------------------------
# Baseline (ii-b): one long constrained-random run, like the original FIFO_test
# ---------------------------------------------------------------------------------------------

RECIPE = get_action(21)  # baseline_random: wr 70 / rd 30 / rst 10, the plan's FIFO_4 recipe
DIRECTED_CYCLES = 100     # FIFO_write_only_sequence and FIFO_read_only_sequence: 100 cycles each


def constrained_random_stream(rng, directed_prefix=True, wr_pct=RECIPE.wr_pct, rd_pct=RECIPE.rd_pct,
                              rst_pct=RECIPE.rst_pct):
    """Endless stimulus of the original test, cycle by cycle.

    Input:  rng, a numpy Generator; directed_prefix, True to copy FIFO_test (reset, 100
            write-only, 100 read-only, then random) or False for pure random after the reset;
            the three percentages (default 70 / 30 / 10).
    Output: a generator of (rst_n, wr_en, rd_en) tuples, never ending.
    Logic:  like "src codes/FIFO_sequence.sv": FIFO_reset_sequence is one reset cycle; the
            write-only and read-only sequences still randomize rst_n, so they reset 10% of the
            time too; FIFO_write_read_sequence draws all three bits with their percentages.
    """
    def bit(pct):
        return int(rng.random() * 100 < pct)

    yield RESET_CYCLE
    if directed_prefix:
        for _ in range(DIRECTED_CYCLES):
            yield (1 - bit(rst_pct), 1, 0)
        for _ in range(DIRECTED_CYCLES):
            yield (1 - bit(rst_pct), 0, 1)
    while True:
        yield (1 - bit(rst_pct), bit(wr_pct), bit(rd_pct))


def cycles_to_close(stream, max_cycles=100_000):
    """How many cycles a stimulus stream needs to hit all 64 bins.

    Input:  stream, an iterable of (rst_n, wr_en, rd_en) starting with a reset;
            max_cycles, give up after this many.
    Output: the cycle number (1-based) at which the last bin was hit, or None if never.
    Logic:  run fake_fifo.step cycle by cycle (one continuous simulation, count carries over)
            and mark each sample's bins until all are covered.
    """
    covered = np.zeros(N_BINS, dtype=bool)
    count = 0
    for n, (rst_n, wr_en, rd_en) in enumerate(stream, start=1):
        if n == 1 and rst_n != 0:
            raise ValueError("the stream must start with a reset cycle (rst_n=0)")
        count, sample = fake_fifo.step(count, rst_n, wr_en, rd_en)
        covered[fake_fifo.sample_bins(sample)] = True
        if covered.all():
            return n
        if n >= max_cycles:
            return None
    return None


def long_random_cycles_to_close(n_seeds, base_seed=0, directed_prefix=True, max_cycles=100_000):
    """Baseline (ii-b) over many seeds.

    Input:  n_seeds, how many independent long runs; base_seed; directed_prefix (see
            constrained_random_stream); max_cycles per run.
    Output: dict with the list of cycles-to-close per seed ("cycles", None = never closed) and
            their statistics.
    Logic:  for each seed, one endless constrained-random stream -> cycles_to_close. Every seed
            uses its own generator from SeedSequence([base_seed, seed]).
    """
    cycles = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, seed]))
        cycles.append(cycles_to_close(constrained_random_stream(rng, directed_prefix), max_cycles))
    closed = [c for c in cycles if c is not None]
    return {
        "n_seeds": n_seeds,
        "directed_prefix": directed_prefix,
        "closure_rate": len(closed) / n_seeds,
        "cycles": cycles,
        "cycles_to_close": _stats(closed),
        "p95": float(np.percentile(closed, 95)) if closed else None,
    }
