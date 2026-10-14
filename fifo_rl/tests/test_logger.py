"""Tests for fifo_rl/utils/logger.py: the per-run CSV format."""

import csv

from fifo_rl.utils.logger import FIELDS, CsvLogger


def test_header_rows_and_empty_columns(tmp_path):
    """Header = FIELDS; missing columns are empty; extra keys are ignored."""
    path = tmp_path / "sub" / "log.csv"
    with CsvLogger(path) as logger:
        logger.log(agent="random", episode=0, step=0, new_bins=37, all_covered=False)
        logger.log(agent="random", episode=0, step=1, new_bins=5, epsilon=0.5)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0].keys()) == FIELDS
    assert len(rows) == 2
    assert rows[0]["new_bins"] == "37" and rows[0]["epsilon"] == "" and rows[0]["loss"] == ""
    assert rows[1]["epsilon"] == "0.5"
