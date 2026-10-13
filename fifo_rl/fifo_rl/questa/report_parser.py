"""Reads a Questa covergroup report and says which of the 64 bins were hit.

The report must come from:
    vcover report -cvg -details -nocompactcrossbins run.ucdb > report.txt

Lines the parser looks at:
    Coverpoint cp_full                                100.00%        100          -    Covered
        bin auto[0]                                       263          1          -    Covered
    Cross cx_almostfull                                87.50%        100          -    Uncovered
            bin <auto[0],auto[0],auto[1]>                   0          1          1    ZERO
            ignore_bin auto_full_1                          0                     -    ZERO   (skipped)

Bins are matched by name, never by line position: Questa lists covered bins first, so the
order inside a cross changes from run to run. The report prints the covergroup twice; the
second copy is only checked against the first.
"""

import re

import numpy as np

from fifo_rl.coverage_model import BIN_INDEX, BIN_NAMES, N_BINS, bin_name

_ITEM_LINE = re.compile(r"^\s*(?:Coverpoint|Cross)\s+(\S+)")
_BIN_LINE = re.compile(r"^\s*bin\s+(\S+)\s+(\d+)")


class ReportError(ValueError):
    """The report is not a usable cg_FIFO report."""


def parse_bin_hits(report_text):
    """Return {bin name: hit count} for every bin line in the report."""
    hits = {}
    item = None
    for line in report_text.splitlines():
        match = _ITEM_LINE.match(line)
        if match:
            item = match.group(1)
            continue
        match = _BIN_LINE.match(line)
        if not match:
            continue
        label, count = match.group(1), int(match.group(2))
        if item is None:
            raise ReportError(f"bin line before any Coverpoint/Cross line: {line.strip()!r}")
        if "*" in label:
            raise ReportError(
                f"compacted cross bin {label} in {item}: "
                "create the report with 'vcover report -cvg -details -nocompactcrossbins'"
            )
        name = bin_name(item, label)
        if name in hits and hits[name] != count:
            raise ReportError(f"{name} is listed twice with different counts: {hits[name]} and {count}")
        hits[name] = count
    return hits


def hits_vector(report_text):
    """Return a bool array of N_BINS, True where the bin was hit, in BIN_NAMES order."""
    hits = parse_bin_hits(report_text)
    unknown = sorted(name for name in hits if name not in BIN_INDEX)
    if unknown:
        raise ReportError(f"{len(unknown)} bin(s) not in the coverage model, e.g. {unknown[0]}")
    missing = [name for name in BIN_NAMES if name not in hits]
    if missing:
        raise ReportError(f"{len(missing)} of {N_BINS} bins missing from the report, e.g. {missing[0]}")
    return np.array([hits[name] > 0 for name in BIN_NAMES], dtype=bool)
