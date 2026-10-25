"""Plots of the evaluation results (baselines and trained agents) and of training itself.

Run from the fifo_rl/ folder after scripts.evaluate / scripts.train:
    python -m scripts.plot                       # reads runs/baselines/*_summary.json
    python -m scripts.plot --dir runs/baselines
    python -m scripts.plot --train runs/train/dqn_seed* runs/train/ddqn_seed*

--dir writes into that folder:
    coverage_vs_runs.png    mean bins covered after each run, ± 1 std band, one line per agent
    runs_to_close.png       histogram of runs needed to reach 64/64 (closed episodes only)

It reads the *_summary.json files, not the CSVs: they already hold the averaged curve, so a DQN
evaluation that writes the same summary appears in the same plot with no extra code.

--train reads each folder's eval.csv and steps.csv, groups the folders by agent (the seeds of
one agent -> mean ± 1 std across seeds) and writes into --train-out (default runs/train/):
    learning_curve.png      greedy-evaluation closure rate and runs used vs training episodes
    loss_epsilon.png        training loss (mean per window of steps) and epsilon vs steps
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window (works in WSL without a display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fifo_rl.coverage_model import N_BINS  # noqa: E402

# First slots of a colour-blind-checked categorical palette, always in this order. Each agent
# also gets its own line style and a legend entry, so identity never depends on colour alone. No
# direct labels on the curves: all agents converge near 64, so labels there collide.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
STYLES = ("-", "--", "-.", ":")
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_summaries(folder):
    """Input: folder. Output: list of summary dicts, sorted by agent name (stable colours)."""
    summaries = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*_summary.json"))]
    if not summaries:
        raise SystemExit(f"no *_summary.json in {folder}: run 'python -m scripts.evaluate' first")
    return sorted(summaries, key=lambda s: s["agent"])


def _style(ax, title, xlabel, ylabel):
    """Recessive axes: light grid behind the data, no top/right frame, text in ink colours."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_coverage(summaries, path):
    """Mean coverage vs runs with a ± 1 std band; a dashed line marks full coverage (64)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    means = [np.array(s["curve_mean"]) for s in summaries]
    n_runs = min(len(m) for m in means)
    for i, s in enumerate(summaries):
        mean, std = means[i], np.array(s["curve_std"])
        runs = np.arange(1, len(mean) + 1)
        color = COLORS[i % len(COLORS)]
        ax.fill_between(runs, mean - std, np.minimum(mean + std, N_BINS), color=color, alpha=0.15, linewidth=0)
        ax.plot(runs, mean, color=color, linestyle=STYLES[i % len(STYLES)], linewidth=2,
                label=f"{s['agent']}  (closes {100 * s['closure_rate']:.0f}%)")
    ax.axhline(N_BINS, color=MUTED, linestyle=":", linewidth=1)
    ax.text(n_runs, N_BINS + 0.3, f"all {N_BINS} bins", color=MUTED, fontsize=9, ha="right", va="bottom")
    _style(ax, f"Coverage vs simulation runs (fake FIFO, {summaries[0]['n_episodes']} episodes, mean ± 1 std)",
           "simulation runs", "bins covered")
    ax.set_xlim(1, n_runs)
    ax.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_runs_to_close(summaries, path):
    """Histogram of runs-to-close, one outlined step histogram per agent on shared bins."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    budget = max(s["budget"] for s in summaries)
    edges = np.arange(0.5, budget + 1.5)
    for i, s in enumerate(summaries):
        ax.hist(s["runs_to_close_all"], bins=edges, histtype="step", linewidth=2,
                color=COLORS[i % len(COLORS)], linestyle=STYLES[i % len(STYLES)],
                label=f"{s['agent']}  ({len(s['runs_to_close_all'])}/{s['n_episodes']} closed)")
    _style(ax, "Runs needed to reach 64/64 (closed episodes only)", "simulation runs", "episodes")
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Training plots (scripts/train.py output folders)
# ---------------------------------------------------------------------------------------------

LOSS_WINDOW = 500  # training steps per point of the loss curve


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(text):
    """Input: a CSV cell. Output: float, or NaN for an empty cell (no loss yet, no closed episode)."""
    return float(text) if text else float("nan")


def load_training_runs(folders):
    """Input: train.py output folders. Output: {agent name: [run, ...]}, sorted by name.

    Logic:  a run is a dict of numpy arrays: per evaluation (episode, closure_rate, runs_used)
            from eval.csv, per training step (epsilon, loss) from steps.csv. The agent name is
            the steps.csv "agent" column, so the seeds of one agent land in one group.
    """
    groups = {}
    for folder in map(Path, folders):
        if not (folder / "eval.csv").exists():
            raise SystemExit(f"no eval.csv in {folder}: run 'python -m scripts.train' first")
        evals, steps = _read_csv(folder / "eval.csv"), _read_csv(folder / "steps.csv")
        run = {
            "episode": np.array([_num(r["episode"]) for r in evals]),
            "closure_rate": np.array([_num(r["closure_rate"]) for r in evals]),
            "runs_used": np.array([_num(r["runs_used_mean"]) for r in evals]),
            "epsilon": np.array([_num(r["epsilon"]) for r in steps]),
            "loss": np.array([_num(r["loss"]) for r in steps]),
        }
        groups.setdefault(steps[0]["agent"] if steps else folder.name, []).append(run)
    return dict(sorted(groups.items()))


def _window_mean(values, window):
    """Input: per-step values (NaN = none), window size. Output: (window centres, mean per window)."""
    n = len(values) // window
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows before learning starts
        means = np.nanmean(values[: n * window].reshape(n, window), axis=1)
    return (np.arange(n) + 0.5) * window, means


def _stack(arrays):
    """Input: 1-D arrays of possibly different lengths. Output: (2-D array cut to the shortest, length)."""
    n = min(len(a) for a in arrays)
    return np.stack([a[:n] for a in arrays]), n


def _band(ax, x, rows, color, style, label):
    """Mean line over rows (one per seed, NaNs ignored), with a ± 1 std band when there are 2+ seeds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, std = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    if len(rows) > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, mean, color=color, linestyle=style, linewidth=2, label=label)


def plot_learning_curve(groups, path):
    """Two panels on one episode axis: closure rate (%) and runs used, both from greedy evaluation."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    for i, (name, runs) in enumerate(groups.items()):
        color, style = COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]
        closure, n = _stack([r["closure_rate"] for r in runs])
        used, _ = _stack([r["runs_used"] for r in runs])
        episodes = runs[0]["episode"][:n]
        label = f"{name}  ({len(runs)} seed{'s' if len(runs) > 1 else ''})"
        _band(top, episodes, 100 * closure, color, style, label)
        _band(bottom, episodes, used, color, style, label)
    spread = ", mean ± 1 std across seeds" if any(len(r) > 1 for r in groups.values()) else ""
    _style(top, f"Greedy evaluation during training{spread}", "", "episodes closed (%)")
    _style(bottom, "Runs used per episode (unclosed = full budget; lower is better)", "training episodes",
           "simulation runs")
    top.set_ylim(0, 105)
    top.legend(frameon=False, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_epsilon(groups, path, window=LOSS_WINDOW):
    """Two panels on one step axis (never one dual-axis chart): loss per window and epsilon."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    any_loss = False
    for i, (name, runs) in enumerate(groups.items()):
        color, style = COLORS[i % len(COLORS)], STYLES[i % len(STYLES)]
        windows = [_window_mean(r["loss"], window) for r in runs]
        losses, n = _stack([means for _, means in windows])
        if n > 0 and not np.isnan(losses).all():
            _band(top, windows[0][0][:n], losses, color, style, name)
            any_loss = True
        eps, n = _stack([r["epsilon"] for r in runs])
        _band(bottom, np.arange(1, n + 1), eps, color, style, name)
    if not any_loss:
        top.text(0.5, 0.5, "no loss logged (the agent never learned)", transform=top.transAxes,
                 ha="center", va="center", color=MUTED)
    _style(top, f"Training loss (mean per {window} steps)", "", "loss")
    _style(bottom, "Exploration rate", "training steps", "epsilon")
    bottom.set_ylim(0, 1.05)
    bottom.legend(frameon=False, labelcolor=TEXT, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=None, help="evaluation folder (default runs/baselines)")
    parser.add_argument("--train", type=Path, nargs="+", help="train.py output folders")
    parser.add_argument("--train-out", type=Path, default=Path("runs/train"))
    args = parser.parse_args(argv)
    if args.dir is not None or not args.train:
        folder = args.dir or Path("runs/baselines")
        summaries = load_summaries(folder)
        plot_coverage(summaries, folder / "coverage_vs_runs.png")
        plot_runs_to_close(summaries, folder / "runs_to_close.png")
        print(f"written {folder / 'coverage_vs_runs.png'} and {folder / 'runs_to_close.png'}")
    if args.train:
        groups = load_training_runs(args.train)
        args.train_out.mkdir(parents=True, exist_ok=True)
        plot_learning_curve(groups, args.train_out / "learning_curve.png")
        plot_loss_epsilon(groups, args.train_out / "loss_epsilon.png")
        print(f"written {args.train_out / 'learning_curve.png'} and {args.train_out / 'loss_epsilon.png'}")


if __name__ == "__main__":
    main()
