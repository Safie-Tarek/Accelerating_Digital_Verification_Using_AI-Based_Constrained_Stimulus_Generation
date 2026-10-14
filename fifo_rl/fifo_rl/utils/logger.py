"""One CSV row per simulation run (env step), the same format for every agent.

Columns:
    agent, episode, step, action_id, action, seed      who did what, with which run seed
    new_bins, total_bins, run_bins, reward             what the run achieved
    runs_left, closed_now                              where the episode stands
    epsilon, loss                                      DQN only; left empty for the baselines

Having the DQN write exactly these columns too means one plot script reads both.
"""

import csv
from pathlib import Path

FIELDS = (
    "agent",
    "episode",
    "step",
    "action_id",
    "action",
    "seed",
    "new_bins",
    "total_bins",
    "run_bins",
    "reward",
    "runs_left",
    "closed_now",
    "epsilon",
    "loss",
)


class CsvLogger:
    def __init__(self, path):
        """Input:  path of the CSV file to create (parent folders are created; an old file is replaced).
        Output: nothing; the header row is written immediately.
        Logic:  csv.DictWriter with restval="" (missing columns stay empty, e.g. epsilon for a
                random agent) and extrasaction="ignore" (the env's info dict may carry extra keys
                such as all_covered; only FIELDS are written).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS, restval="", extrasaction="ignore")
        self._writer.writeheader()

    def log(self, **row):
        """Input: column values as keyword arguments. Output: nothing. Logic: write one row."""
        self._writer.writerow(row)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
