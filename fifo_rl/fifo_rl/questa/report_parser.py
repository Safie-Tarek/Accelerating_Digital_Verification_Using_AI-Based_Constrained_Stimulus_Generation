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

Step 2: the report may also hold the per-block covergroup (cg_action, option.per_instance = 1):
    TYPE /FIFO_coverage_pkg/FIFO_coverage/cg_action               (type-level totals: skipped)
    INSTANCE /FIFO_coverage_pkg/FIFO_coverage/cg_action/act_3     (bins of block idx 3)
hits_vector reads only cg_FIFO; hits_per_action reads the act_<idx> instances. The exact
instance header of the step-2 testbench is still to be checked on a real Questa report.
"""

import re

import numpy as np

from fifo_rl.coverage_model import BIN_INDEX, BIN_NAMES, N_BINS, bin_name

_ITEM_LINE = re.compile(r"^\s*(?:Coverpoint|Cross)\s+(\S+)")
_BIN_LINE = re.compile(r"^\s*bin\s+(\S+)\s+(\d+)")
_TYPE_LINE = re.compile(r"^\s*TYPE\s+(\S+)")
_INSTANCE_LINE = re.compile(r"^\s*(?:INSTANCE|Covergroup instance)\s+(\S+)")
_ACT_NAME = re.compile(r"act_(\d+)\b")
MAIN_COVERGROUP = "cg_FIFO"


class ReportError(ValueError):
    """The report is not a usable cg_FIFO report."""


def _sections(report_text):
    """Split the report into covergroup sections.

    Input:  report_text.
    Output: {(covergroup, instance): [lines]}; instance is None for the type-level section.
    Logic:  a TYPE line starts a covergroup (its last path part, e.g. "cg_FIFO"), an INSTANCE line
            starts one instance of it; lines before any TYPE line count as cg_FIFO (short reports).
    """
    sections = {}
    key = (MAIN_COVERGROUP, None)
    for line in report_text.splitlines():
        match = _TYPE_LINE.match(line)
        if match:
            key = (match.group(1).rsplit("/", 1)[-1], None)
            continue
        match = _INSTANCE_LINE.match(line)
        if match:
            key = (key[0], match.group(1))
            continue
        sections.setdefault(key, []).append(line)
    return sections


def parse_bin_hits(report_text):
    """Return {bin name: hit count} for every bin line of cg_FIFO in the report.

    Input:  report_text, the text of `vcover report -cvg -details -nocompactcrossbins`.
    Output: a dict {full bin name: hit count}, e.g. {"cp_full/auto[1]": 163, ...}.
    Logic:  only the cg_FIFO lines (the step-2 cg_action sections are skipped), then _parse_lines.
    """
    return _parse_lines(_sections(report_text).get((MAIN_COVERGROUP, None), []))


def _parse_lines(lines):
    """Input: the lines of one covergroup section. Output: {full bin name: hit count}.

    Logic:  read line by line; a "Coverpoint"/"Cross" line sets the current item, a "bin" line
            gives (label, count) for that item ("ignore_bin" lines do not match and are skipped).
            Raise ReportError for a compacted report ("*" in a label), a bin line before any item,
            or the same bin listed twice with different counts (the report prints it twice).
    """
    hits = {}
    item = None
    for line in lines:
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
    """Return a bool array of N_BINS, True where the bin was hit, in BIN_NAMES order.

    Input:  report_text, the text of a Questa coverage report.
    Output: a bool numpy array of 64 (same order as the state vector): True = hit at least once.
    Logic:  parse the counts with parse_bin_hits, check the report has exactly our 64 bins
            (no unknown bin, none missing - else ReportError), then map each name to count > 0.
    """
    return _to_vector(parse_bin_hits(report_text))


def _to_vector(hits, where="the report"):
    """Input: {bin name: count}, a name for errors. Output: bool[64] in BIN_NAMES order.
    Logic:  exactly our 64 bins (no unknown bin, none missing - else ReportError), then count > 0.
    """
    unknown = sorted(name for name in hits if name not in BIN_INDEX)
    if unknown:
        raise ReportError(f"{len(unknown)} bin(s) of {where} not in the coverage model, e.g. {unknown[0]}")
    missing = [name for name in BIN_NAMES if name not in hits]
    if missing:
        raise ReportError(f"{len(missing)} of {N_BINS} bins missing from {where}, e.g. {missing[0]}")
    return np.array([hits[name] > 0 for name in BIN_NAMES], dtype=bool)


def hits_per_action(report_text, n_blocks):
    """Return which bins each block hit (step 2).

    Input:  report_text, a report with the cg_action instances act_0 .. act_<n_blocks-1>;
            n_blocks, the number of ACT lines of the plan.
    Output: bool array [n_blocks, 64], row k = bins hit during block idx k.
    Logic:  each act_<k> instance must be there with all 64 bins; a missing one (a step-1 report,
            or a block the testbench did not run) or an extra one is a ReportError - it never
            silently turns into "this block hit nothing".
    """
    found = {}
    for (covergroup, instance), lines in _sections(report_text).items():
        if instance is None:
            continue
        match = _ACT_NAME.search(instance)
        if match:
            found[int(match.group(1))] = lines
    missing = [k for k in range(n_blocks) if k not in found]
    if missing:
        raise ReportError(f"report has no act_{missing[0]} instance ({len(missing)} of {n_blocks} missing): "
                          "reward_mode=action needs per-action coverage (cg_action, one act_<idx> per block)")
    extra = sorted(k for k in found if k >= n_blocks)
    if extra:
        raise ReportError(f"report has act_{extra[0]} but the plan has only {n_blocks} blocks")
    return np.array([_to_vector(_parse_lines(found[k]), f"act_{k}") for k in range(n_blocks)],
                    dtype=bool).reshape(n_blocks, N_BINS)
