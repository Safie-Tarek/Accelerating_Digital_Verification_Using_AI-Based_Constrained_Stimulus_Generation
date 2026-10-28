"""Tests for fifo_rl/launchers (fake + Questa) and fifo_rl/questa/result_parser.py."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from fifo_rl import fake_fifo
from fifo_rl.launchers.base import SimulationError
from fifo_rl.launchers.fake_launcher import FakeLauncher, format_report
from fifo_rl.launchers.questa_launcher import QuestaLauncher
from fifo_rl.plan.actions import ACTION_INDEX, END_TEST_ID, plan_cycles
from fifo_rl.plan.plan_file import TextPlanWriter
from fifo_rl.plan.stimulus_plan import StimulusPlan
from fifo_rl.questa.report_parser import hits_vector
from fifo_rl.questa.result_parser import ResultError, ResultParser

FIXTURES = Path(__file__).parent / "fixtures"


def write_plan(path, *names, seed=7):
    plan = StimulusPlan(plan_id=0, seed=seed, max_len=10)
    for name in names:
        plan.add(ACTION_INDEX[name], np.zeros(1))
    plan.add(END_TEST_ID, np.zeros(1))
    return plan, TextPlanWriter().write(plan, path)


# ---------- FakeLauncher ----------

def test_fake_report_is_read_like_a_real_one():
    counts = np.arange(64)
    np.testing.assert_array_equal(hits_vector(format_report(counts)), counts > 0)


def test_fake_launcher_simulates_the_plan_file(tmp_path):
    plan, path = write_plan(tmp_path / "plan.txt", "WRITE", "IDLE")
    out = FakeLauncher().launch(path, seed=1, run_dir=tmp_path)
    expected = fake_fifo.hits(plan_cycles(plan.action_ids()))
    np.testing.assert_array_equal(hits_vector(out.coverage_report.read_text()), expected)
    result = ResultParser().parse(out.result_file, out.log_file)
    assert result.passed and result.cycles == plan.sim_cycles() == 15 and result.n_actions == 2
    assert out.log_file is None and out.report_time == 0 and out.wall_time >= 0


def test_fake_launcher_same_seed_same_test(tmp_path):
    _, path = write_plan(tmp_path / "plan.txt", "RAND50", "BASELINE", "RESET_TRAFFIC")
    a = FakeLauncher().launch(path, 3, tmp_path / "a").coverage_report.read_text()
    b = FakeLauncher().launch(path, 3, tmp_path / "b").coverage_report.read_text()
    assert a == b


# ---------- ResultParser ----------

def test_result_file_wins_over_log(tmp_path):
    (tmp_path / "result.txt").write_text("passed=1 errors=0 assert_fail=0 cycles=23 actions=2\n")
    (tmp_path / "sim.log").write_text("# UVM_ERROR :    3\n")
    r = ResultParser().parse(tmp_path / "result.txt", tmp_path / "sim.log")
    assert r.passed and r.source == "result" and r.cycles == 23 and r.n_actions == 2


def test_assertion_errors_in_the_log_fail_a_passed_result(tmp_path):
    """SVA $error never reaches the UVM counters in result.txt: the log's ** Error lines must count."""
    (tmp_path / "result.txt").write_text("passed=1 errors=0 fatals=0 cycles=15 actions=2\n")
    (tmp_path / "sim.log").write_text("# ** Error: Assertion failed: dut.count did not increment\n# UVM_ERROR :    0\n")
    r = ResultParser().parse(tmp_path / "result.txt", tmp_path / "sim.log")
    assert not r.passed and r.assert_fails == 1 and r.cycles == 15


def test_result_file_with_errors_fails():
    assert not ResultParser().parse_result_text("passed=1 errors=2").passed
    with pytest.raises(ResultError):
        ResultParser().parse_result_text("errors=0")


def test_log_summary_clean_and_dirty():
    clean = "# --- UVM Report Summary ---\n# UVM_INFO :   12\n# UVM_WARNING :    0\n# UVM_ERROR :    0\n# UVM_FATAL :    0\n"
    r = ResultParser().parse_log_text(clean)
    assert r.passed and r.source == "log" and r.cycles is None
    dirty = "# ** Error: Assertion failed: dut.count did not increment\n" + clean
    assert ResultParser().parse_log_text(dirty).assert_fails == 1
    assert not ResultParser().parse_log_text(dirty).passed
    assert not ResultParser().parse_log_text(clean.replace("UVM_ERROR :    0", "UVM_ERROR :    4")).passed


def test_log_without_summary_did_not_finish():
    r = ResultParser().parse_log_text("# Loading work.FIFO_top\n")
    assert not r.passed and "did not finish" in r.message


def test_no_result_and_no_log(tmp_path):
    with pytest.raises(ResultError):
        ResultParser().parse(tmp_path / "result.txt", tmp_path / "sim.log")


# ---------- QuestaLauncher (with a stand-in for subprocess.run) ----------

class FakeQuesta:
    """Records every command; vsim writes the log + UCDB, vcover prints a real report."""

    def __init__(self, vsim_exit=0, save_ucdb=True):
        self.calls = []
        self.vsim_exit = vsim_exit
        self.save_ucdb = save_ucdb

    def __call__(self, args, cwd, capture_output, text, timeout):
        self.calls.append((list(args), Path(cwd)))
        tool = Path(args[0]).name
        stdout = ""
        if tool.startswith("vsim"):
            do = args[args.index("-do") + 1]
            ucdb = do.split("-onexit ")[1].split(";")[0]
            log = args[args.index("-l") + 1]
            (Path(cwd) / log).write_text("# UVM_ERROR :    0\n# UVM_FATAL :    0\n")
            if self.save_ucdb:
                (Path(cwd) / ucdb).write_text("ucdb")
            return subprocess.CompletedProcess(args, self.vsim_exit, "", "vsim error text")
        if tool.startswith("vcover"):
            stdout = (FIXTURES / "cov_seed1_300cyc.txt").read_text()
        return subprocess.CompletedProcess(args, 0, stdout, "")


def make_launcher(tmp_path, runner, **kw):
    return QuestaLauncher(tmp_path / "work", tmp_path / "src codes" / "src_files.list", convert_paths=False,
                          run_command=runner, **kw)


def test_questa_commands_and_outputs(tmp_path):
    runner = FakeQuesta()
    launcher = make_launcher(tmp_path, runner)
    launcher.compile()
    run_dir = launcher.work_dir / "tests" / "t0"
    _, plan_file = write_plan(run_dir / "plan.txt", "WRITE")
    out = launcher.launch(plan_file, seed=99, run_dir=run_dir)

    (vlib, _), (vlog, _), (vsim, cwd), (vcover, _) = runner.calls
    assert vlib == ["vlib.exe", "work"]
    assert vlog[:2] == ["vlog.exe", "-F"] and vlog[2].endswith("src_files.list") and "+cover" in vlog
    assert cwd == launcher.work_dir
    assert vsim[:4] == ["vsim.exe", "-c", "-sv_seed", "99"]
    assert "+UVM_TESTNAME=FIFO_rl_test" in vsim and "+PLAN_FILE=tests/t0/plan.txt" in vsim
    assert "+RESULT_FILE=tests/t0/result.txt" in vsim
    assert vsim[vsim.index("-do") + 1] == "coverage save -onexit tests/t0/run.ucdb; run -all; quit -f"
    assert "-du" not in " ".join(vsim)                       # -du FIFO would drop cg_FIFO
    assert vcover == ["vcover.exe", "report", "-cvg", "-details", "-nocompactcrossbins", "tests/t0/run.ucdb"]

    assert hits_vector(out.coverage_report.read_text()).sum() == 63   # the 300-cycle fixture
    assert out.result_file is None and out.log_file.is_file()
    assert ResultParser().parse(out.result_file, out.log_file).passed


def test_questa_tool_folder_and_linux_suffix(tmp_path):
    launcher = make_launcher(tmp_path, FakeQuesta(), questa_bin="/opt/questa/bin", tool_suffix="")
    assert Path(launcher.tool("vsim")) == Path("/opt/questa/bin/vsim")


def test_questa_failures_raise(tmp_path):
    launcher = make_launcher(tmp_path, FakeQuesta(vsim_exit=3))
    run_dir = launcher.work_dir / "tests" / "t0"
    _, plan_file = write_plan(run_dir / "plan.txt", "WRITE")
    with pytest.raises(SimulationError, match="exit code 3"):
        launcher.launch(plan_file, 1, run_dir)
    launcher = make_launcher(tmp_path, FakeQuesta(save_ucdb=False))
    with pytest.raises(SimulationError, match="no coverage database"):
        launcher.launch(plan_file, 1, run_dir)


def test_questa_run_dir_must_be_inside_work_dir(tmp_path):
    launcher = make_launcher(tmp_path, FakeQuesta())
    _, plan_file = write_plan(tmp_path / "elsewhere" / "plan.txt", "WRITE")
    with pytest.raises(ValueError, match="inside the Questa work dir"):
        launcher.launch(plan_file, 1, tmp_path / "elsewhere")


def test_questa_missing_tool(tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError
    with pytest.raises(SimulationError, match="tool not found"):
        make_launcher(tmp_path, missing).compile()
