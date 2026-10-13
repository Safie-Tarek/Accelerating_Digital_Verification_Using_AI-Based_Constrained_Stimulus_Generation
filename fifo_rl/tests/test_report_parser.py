from pathlib import Path

import pytest

from fifo_rl.coverage_model import BIN_INDEX, BIN_NAMES
from fifo_rl.questa.report_parser import ReportError, hits_vector, parse_bin_hits
from fifo_rl.state import StateEncoder

FIXTURES = Path(__file__).parent / "fixtures"

# Real Questa reports of the default test, seed 1, stopped after 100 / 300 / all 10,200 cycles.
REPORT_100 = (FIXTURES / "cov_seed1_100cyc.txt").read_text()
REPORT_300 = (FIXTURES / "cov_seed1_300cyc.txt").read_text()
REPORT_FULL = (FIXTURES / "cov_seed1_full.txt").read_text()
# Same run as REPORT_100, but created without -nocompactcrossbins.
REPORT_COMPACT = (FIXTURES / "cov_seed1_100cyc_compact.txt").read_text()

IDLE_AT_ALMOSTFULL = "cx_almostfull/<auto[0],auto[0],auto[1]>"


def test_report_has_exactly_the_64_bins():
    for report in (REPORT_100, REPORT_300, REPORT_FULL):
        assert set(parse_bin_hits(report)) == set(BIN_NAMES)


def test_number_of_bins_hit():
    assert hits_vector(REPORT_100).sum() == 36
    assert hits_vector(REPORT_300).sum() == 63
    assert hits_vector(REPORT_FULL).sum() == 64


def test_vector_shape_and_type():
    vector = hits_vector(REPORT_100)
    assert vector.shape == (64,)
    assert vector.dtype == bool


def test_only_missing_bin_after_300_cycles():
    vector = hits_vector(REPORT_300)
    assert [BIN_NAMES[i] for i in range(64) if not vector[i]] == [IDLE_AT_ALMOSTFULL]


def test_hit_counts():
    hits = parse_bin_hits(REPORT_300)
    assert hits["cp_wr_en/auto[0]"] == 135
    assert hits["cp_wr_en/auto[1]"] == 165
    assert hits[IDLE_AT_ALMOSTFULL] == 0

    hits = parse_bin_hits(REPORT_FULL)
    assert hits["cp_wr_en/auto[0]"] == 3171
    assert hits["cp_wr_en/auto[1]"] == 7029
    assert hits["cx_full/<auto[0],auto[0],auto[1]>"] == 163


def test_bins_are_placed_by_name_not_by_line_position():
    # In the 300-cycle report the ZERO bin is printed last inside cx_almostfull,
    # but in our order it is the second bin of that cross.
    index = BIN_INDEX[IDLE_AT_ALMOSTFULL]
    assert index == 43
    assert not hits_vector(REPORT_300)[index]
    assert hits_vector(REPORT_FULL)[index]


def test_line_endings_do_not_matter():
    unix = REPORT_300.replace("\r\n", "\n")
    windows = unix.replace("\n", "\r\n")
    assert (hits_vector(unix) == hits_vector(windows)).all()


def test_one_copy_of_the_covergroup_is_enough():
    first_copy = REPORT_300.split("COVERGROUP COVERAGE:")[0]
    assert len(first_copy) < len(REPORT_300)
    assert (hits_vector(first_copy) == hits_vector(REPORT_300)).all()


def test_compacted_report_is_rejected():
    with pytest.raises(ReportError, match="nocompactcrossbins"):
        hits_vector(REPORT_COMPACT)


def test_report_without_the_covergroup_is_rejected():
    with pytest.raises(ReportError, match="64 of 64 bins missing"):
        hits_vector("TOTAL COVERGROUP COVERAGE: 0.00%  COVERGROUP TYPES: 0\n")


def test_cut_off_report_is_rejected():
    cut = REPORT_300.split("Cross cx_underflow")[0]
    with pytest.raises(ReportError, match="6 of 64 bins missing"):
        hits_vector(cut)


def test_unknown_bin_is_rejected():
    extra = REPORT_300 + "    Coverpoint cp_rst_n\n        bin auto[0]    5    1    -    Covered\n"
    with pytest.raises(ReportError, match="cp_rst_n/auto\\[0\\]"):
        hits_vector(extra)


def test_two_copies_that_disagree_are_rejected():
    changed = REPORT_300 + "    Coverpoint cp_wr_en\n        bin auto[0]    999    1    -    Covered\n"
    with pytest.raises(ReportError, match="different counts"):
        hits_vector(changed)


def test_bin_line_without_an_item_is_rejected():
    with pytest.raises(ReportError, match="before any Coverpoint/Cross"):
        parse_bin_hits("        bin auto[0]    5    1    -    Covered\n")


def test_reports_feed_the_state_encoder():
    enc = StateEncoder(budget=20)

    state, n_new = enc.update(hits_vector(REPORT_100), action_id=12)
    assert n_new == 36
    assert state[:64].sum() == 36
    assert state[64] == pytest.approx(0.95)
    assert state[77] == 1.0

    # Same seed, longer run: it hits everything the short run hit, plus 27 more.
    state, n_new = enc.update(hits_vector(REPORT_300), action_id=3)
    assert n_new == 27
    assert state[:64].sum() == 63

    state, n_new = enc.update(hits_vector(REPORT_FULL), action_id=3)
    assert n_new == 1
    assert enc.all_covered
