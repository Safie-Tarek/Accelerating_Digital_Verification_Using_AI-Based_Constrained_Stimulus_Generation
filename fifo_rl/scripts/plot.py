"""Plots of the evaluation results (baselines now, the DQN later).

Run from the fifo_rl/ folder after scripts.evaluate:
    python -m scripts.plot                       # reads runs/baselines/*_summary.json
    python -m scripts.plot --dir runs/baselines

Writes into the same folder:
    coverage_vs_runs.png    mean bins covered after each run, ± 1 std band, one line per agent
    runs_to_close.png       histogram of runs needed to reach 64/64 (closed episodes only)

It reads the *_summary.json files, not the CSVs: they already hold the averaged curve, so a DQN
evaluation that writes the same summary appears in the same plot with no extra code.
"""

import argparse
import json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("runs/baselines"))
    args = parser.parse_args(argv)
    summaries = load_summaries(args.dir)
    plot_coverage(summaries, args.dir / "coverage_vs_runs.png")
    plot_runs_to_close(summaries, args.dir / "runs_to_close.png")
    print(f"written {args.dir / 'coverage_vs_runs.png'} and {args.dir / 'runs_to_close.png'}")


if __name__ == "__main__":
    main()
