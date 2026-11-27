"""Plots of plan-based training runs (scripts/run_campaign.py train), DQN / DDQN and A2C alike.

Run from the fifo_rl/ folder:
    python -m scripts.plot_campaign runs/campaign/a2c/train_s2_seed* --out runs/campaign/a2c/plots
    python -m scripts.plot_campaign runs/a/seed* runs/b/seed* --out plots   # folders grouped by agent

Reads each folder's eval.csv (validation every eval_every campaigns) and tests.csv (training rows),
groups the folders by the agent name in their summary.json (seeds of one agent -> mean, min and
max across seeds) and writes into --out:
    validation.png    expected tests (failed campaign = full budget) and closure rate vs campaigns
    a2c_losses.png    A2C only: policy loss, value loss and entropy of the updates vs campaigns
                      (mean per window of campaigns)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = 50  # campaigns per point of the loss curves


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_run(folder):
    """Input: a training folder. Output: dict name, eval rows, training rows."""
    folder = Path(folder)
    name = json.loads((folder / "summary.json").read_text())["agent"]
    train = [r for r in _read_csv(folder / "tests.csv") if r["phase"] == "train"]
    return {"name": name, "folder": folder.name, "eval": _read_csv(folder / "eval.csv"), "train": train}


def group(runs):
    """Input: runs from load_run. Output: {agent name: [runs]} in first-seen order."""
    groups = defaultdict(list)
    for run in runs:
        groups[run["name"]].append(run)
    return dict(groups)


def _band(ax, x, rows, label, color):
    rows = np.array(rows, dtype=float)
    ax.plot(x, rows.mean(axis=0), color=color, label=f"{label} (mean of {len(rows)})")
    if len(rows) > 1:
        ax.fill_between(x, rows.min(axis=0), rows.max(axis=0), color=color, alpha=0.2, linewidth=0)


def plot_validation(groups, path):
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, runs) in enumerate(groups.items()):
        n = min(len(r["eval"]) for r in runs)
        x = [int(r["campaign"]) for r in runs[0]["eval"][:n]]
        color = f"C{i}"
        _band(ax_t, x, [[float(e["tests_charged_mean"]) for e in r["eval"][:n]] for r in runs], name, color)
        _band(ax_c, x, [[100 * float(e["closure_rate"]) for e in r["eval"][:n]] for r in runs], name, color)
    ax_t.set(title="Expected tests per campaign (lower is better)", xlabel="training campaigns", ylabel="tests")
    ax_t.set_yscale("log")
    ax_c.set(title="Closure rate on validation campaigns", xlabel="training campaigns", ylabel="% closed")
    for ax in (ax_t, ax_c):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _windowed(train_rows, key, n_campaigns):
    """Input: training rows, a column, campaigns trained. Output: mean of the column per WINDOW campaigns."""
    sums, counts = np.zeros(n_campaigns // WINDOW + 1), np.zeros(n_campaigns // WINDOW + 1)
    for r in train_rows:
        if r.get(key):
            w = int(r["campaign"]) // WINDOW
            sums[w] += float(r[key])
            counts[w] += 1
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def plot_a2c_losses(groups, path):
    a2c = {name: runs for name, runs in groups.items() if runs[0]["train"] and runs[0]["train"][0].get("entropy") is not None
           and any(r["entropy"] for r in runs[0]["train"])}
    if not a2c:
        return False
    keys = ("policy_loss", "value_loss", "entropy")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, runs in a2c.items():
        for run in runs:
            n = max(int(r["campaign"]) for r in run["train"]) + 1
            x = np.arange(n // WINDOW + 1) * WINDOW
            for ax, key in zip(axes, keys):
                ax.plot(x, _windowed(run["train"], key, n), label=f"{name} {run['folder']}", linewidth=1)
    for ax, key in zip(axes, keys):
        ax.set(title=key.replace("_", " "), xlabel="training campaigns")
        ax.grid(alpha=0.3)
    axes[2].axhline(np.log(9), color="grey", linestyle=":", label="uniform over 9 blocks")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/campaign/plots"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    groups = group([load_run(f) for f in args.folders])
    plot_validation(groups, args.out / "validation.png")
    wrote = ["validation.png"] + (["a2c_losses.png"] if plot_a2c_losses(groups, args.out / "a2c_losses.png") else [])
    print(f"wrote {', '.join(wrote)} to {args.out}/")


if __name__ == "__main__":
    main()
