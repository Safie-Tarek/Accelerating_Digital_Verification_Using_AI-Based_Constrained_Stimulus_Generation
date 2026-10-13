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
    """Full bin name from a coverpoint/cross name and Questa's bin label."""
    return f"{item}/{label}"


def coverpoint_label(value):
    return f"auto[{value}]"


def cross_label(wr_en, rd_en, flag):
    return f"<auto[{wr_en}],auto[{rd_en}],auto[{flag}]>"


def _build_bin_names():
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
