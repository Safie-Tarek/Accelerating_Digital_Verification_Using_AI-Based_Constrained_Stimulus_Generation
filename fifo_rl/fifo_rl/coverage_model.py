"""The 64 coverage bins of covergroup cg_FIFO, in one fixed order.

Mirrors "src codes/FIFO_coverage.sv". If that covergroup changes, change this file.

Bin names use Questa's own labels, so a coverage report can be looked up by name:
    "cp_full/auto[1]"                      coverpoint bin: full was seen as 1
    "cx_full/<auto[0],auto[0],auto[1]>"    cross bin: wr_en=0, rd_en=0, full=1
"""

from itertools import product

# The 9 coverpoints, in the order they are declared. Each has two bins: auto[0], auto[1].
COVERPOINTS = (
    "cp_wr_en",
    "cp_rd_en",
    "cp_wr_ack",
    "cp_overflow",
    "cp_full",
    "cp_empty",
    "cp_almostfull",
    "cp_almostempty",
    "cp_underflow",
)

# The 7 crosses (wr_en x rd_en x flag), in the order they are declared.
# Value = the (wr_en, rd_en, flag) combinations removed by ignore_bins.
CROSSES = {
    "cx_wr_ack": {(0, 1, 1), (0, 0, 1), (1, 1, 0)},
    "cx_overflow": {(0, 1, 1), (0, 0, 1), (1, 1, 1)},
    "cx_full": {(0, 1, 1), (1, 1, 1)},
    "cx_empty": set(),
    "cx_almostfull": set(),
    "cx_almostempty": set(),
    "cx_underflow": {(1, 0, 1), (0, 0, 1)},
}


def bin_name(item, label):
    """Full bin name from a coverpoint/cross name and Questa's bin label.

    Input:  item, a coverpoint/cross name (e.g. "cx_full"); label, Questa's bin label
            (e.g. "<auto[0],auto[0],auto[1]>").
    Output: the bin's full name, e.g. "cx_full/<auto[0],auto[0],auto[1]>".
    Logic:  join the two with "/" so every bin has one unique name across the covergroup.
    """
    return f"{item}/{label}"


def coverpoint_label(value):
    """Questa's label of a coverpoint bin.

    Input:  value, 0 or 1 (the value of the sampled signal).
    Output: the label "auto[0]" or "auto[1]".
    Logic:  the coverpoints have no explicit bins, so Questa names them auto[<value>].
    """
    return f"auto[{value}]"


def cross_label(wr_en, rd_en, flag):
    """Questa's label of a cross bin.

    Input:  wr_en, rd_en, flag - each 0 or 1.
    Output: the label, e.g. "<auto[1],auto[0],auto[1]>".
    Logic:  a cross bin is named by the auto bin of each crossed coverpoint, in declaration order.
    """
    return f"<auto[{wr_en}],auto[{rd_en}],auto[{flag}]>"


def _build_bin_names():
    """Build the fixed list of the 64 bin names.

    Input:  nothing (uses COVERPOINTS and CROSSES above).
    Output: a tuple of 64 names; the position of a name is its index in every state vector.
    Logic:  first the 9 coverpoints x (auto[0], auto[1]) = 18 names; then for each cross every
            (wr_en, rd_en, flag) combo in sorted order, skipping the ignore_bins = 46 names.
    """
    names = []
    for cp in COVERPOINTS:
        for value in (0, 1):
            names.append(bin_name(cp, coverpoint_label(value)))
    for cx, ignored in CROSSES.items():
        # product() yields (0,0,0), (0,0,1), ... (1,1,1): sorted by (wr_en, rd_en, flag)
        for combo in product((0, 1), repeat=3):
            if combo not in ignored:
                names.append(bin_name(cx, cross_label(*combo)))
    return tuple(names)


# Index 0-17 coverpoints, 18-22 cx_wr_ack, 23-27 cx_overflow, 28-33 cx_full, 34-41 cx_empty,
# 42-49 cx_almostfull, 50-57 cx_almostempty, 58-63 cx_underflow.
BIN_NAMES = _build_bin_names()
BIN_INDEX = {name: i for i, name in enumerate(BIN_NAMES)}
N_BINS = len(BIN_NAMES)
