from fifo_rl.coverage_model import BIN_INDEX, BIN_NAMES, COVERPOINTS, CROSSES, N_BINS


def bins_of(item):
    return [name for name in BIN_NAMES if name.startswith(item + "/")]


def test_64_unique_bins():
    assert N_BINS == 64
    assert len(BIN_NAMES) == 64
    assert len(set(BIN_NAMES)) == 64


def test_index_matches_order():
    for i, name in enumerate(BIN_NAMES):
        assert BIN_INDEX[name] == i


def test_coverpoints_come_first_with_two_bins_each():
    assert len(COVERPOINTS) == 9
    for i, cp in enumerate(COVERPOINTS):
        assert BIN_NAMES[2 * i] == f"{cp}/auto[0]"
        assert BIN_NAMES[2 * i + 1] == f"{cp}/auto[1]"


def test_cross_sizes():
    sizes = {cx: len(bins_of(cx)) for cx in CROSSES}
    assert sizes == {
        "cx_wr_ack": 5,
        "cx_overflow": 5,
        "cx_full": 6,
        "cx_empty": 8,
        "cx_almostfull": 8,
        "cx_almostempty": 8,
        "cx_underflow": 6,
    }


def test_cross_index_ranges():
    ranges = {
        "cx_wr_ack": (18, 22),
        "cx_overflow": (23, 27),
        "cx_full": (28, 33),
        "cx_empty": (34, 41),
        "cx_almostfull": (42, 49),
        "cx_almostempty": (50, 57),
        "cx_underflow": (58, 63),
    }
    for cx, (first, last) in ranges.items():
        assert [BIN_INDEX[name] for name in bins_of(cx)] == list(range(first, last + 1))


def test_cross_bins_sorted_by_wr_rd_flag():
    assert bins_of("cx_full") == [
        "cx_full/<auto[0],auto[0],auto[0]>",
        "cx_full/<auto[0],auto[0],auto[1]>",
        "cx_full/<auto[0],auto[1],auto[0]>",
        "cx_full/<auto[1],auto[0],auto[0]>",
        "cx_full/<auto[1],auto[0],auto[1]>",
        "cx_full/<auto[1],auto[1],auto[0]>",
    ]


def test_ignored_bins_are_absent():
    for cx, ignored in CROSSES.items():
        for wr_en, rd_en, flag in ignored:
            assert f"{cx}/<auto[{wr_en}],auto[{rd_en}],auto[{flag}]>" not in BIN_INDEX
