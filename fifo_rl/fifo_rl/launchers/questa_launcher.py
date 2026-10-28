"""Launcher on real Questa (step 1: one coverage report per test).

    launcher = QuestaLauncher(work_dir="runs/questa", src_list="../src codes/src_files.list")
    launcher.compile()                                   # once: vlib work + vlog
    output = launcher.launch(plan_file, seed, run_dir)   # run_dir must be inside work_dir

Per test, from work_dir (where the `work` library is), with paths relative to it:
    vsim -c -sv_seed <seed> <top> -coverage -l <run>/sim.log +UVM_TESTNAME=<test> +PLAN_FILE=<run>/plan.txt
         +RESULT_FILE=<run>/result.txt -do "coverage save -onexit <run>/run.ucdb; run -all; quit -f"
    vcover report -cvg -details -nocompactcrossbins <run>/run.ucdb      -> <run>/coverage.txt
Coverage is saved WITHOUT -du FIFO (that option drops cg_FIFO). Relative paths are used because
Windows tools called from WSL translate the working directory but not path arguments; the one
absolute path (the source list for vlog) goes through `wslpath -w` when needed.

The testbench side: fifo_rl/tb_rl/ holds the AI team's stand-in FIFO_rl_test (top FIFO_rl_top), compiled
from tb_rl/rl_files.list on top of the unchanged "src codes/" files, until the verification team
delivers its own test; then only the questa_* config keys change.
"""

import subprocess
import sys
import time
from pathlib import Path

from fifo_rl.launchers.base import RunOutput, SimulationError, SimulatorLauncher


def running_in_wsl():
    """Input: nothing. Output: True when this Python runs inside WSL (Linux kernel built by Microsoft)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


class QuestaLauncher(SimulatorLauncher):
    name = "questa"

    def __init__(self, work_dir, src_list, test_name="FIFO_rl_test", top="work.FIFO_top", questa_bin="",
                 tool_suffix=".exe", timeout_s=600, convert_paths=None, run_command=subprocess.run):
        """Input:  work_dir, folder holding the compiled `work` library (test folders go below it);
                src_list, path of "src codes/src_files.list";
                test_name, the UVM test that reads +PLAN_FILE; top, the top module;
                questa_bin, folder of the tools ("" = found on PATH);
                tool_suffix, ".exe" for Windows Questa (also from WSL), "" for Linux Questa;
                timeout_s, seconds before one simulation is killed;
                convert_paths, pass absolute paths through `wslpath -w` (None = only in WSL with .exe tools);
                run_command, subprocess.run or a stand-in with the same interface (tests).
        Output: nothing.
        """
        self.work_dir = Path(work_dir).resolve()
        self.src_list = Path(src_list).resolve()
        self.test_name = test_name
        self.top = top
        self.questa_bin = questa_bin
        self.tool_suffix = tool_suffix
        self.timeout_s = timeout_s
        self.convert_paths = (running_in_wsl() and tool_suffix == ".exe") if convert_paths is None else convert_paths
        self.run_command = run_command

    def tool(self, name):
        """Input: "vsim", "vcover", "vlib" or "vlog". Output: the command to call."""
        exe = name + self.tool_suffix
        return str(Path(self.questa_bin) / exe) if self.questa_bin else exe

    def host_path(self, path):
        """Input: an absolute path. Output: the path the Questa tools understand (Windows form from WSL)."""
        if not self.convert_paths:
            return str(path)
        done = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True)
        return done.stdout.strip()

    def relative(self, path):
        """Input: a path inside work_dir. Output: it relative to work_dir, with forward slashes."""
        try:
            return Path(path).resolve().relative_to(self.work_dir).as_posix()
        except ValueError:
            raise ValueError(f"{path} must be inside the Questa work dir {self.work_dir}") from None

    def _run(self, args, what, log_file=None):
        """Input: command list, a short name for errors, an optional log to point at.
        Output: the finished process (stdout captured as text).
        Logic:  run from work_dir; a missing tool, a timeout or a non-zero exit -> SimulationError.
        """
        try:
            done = self.run_command(args, cwd=self.work_dir, capture_output=True, text=True, timeout=self.timeout_s)
        except FileNotFoundError as exc:
            raise SimulationError(f"{what}: tool not found ({args[0]}); set questa_bin / questa_tool_suffix") from exc
        except subprocess.TimeoutExpired as exc:
            raise SimulationError(f"{what}: no result after {self.timeout_s} s") from exc
        if done.returncode != 0:
            where = f"; see {log_file}" if log_file else ""
            raise SimulationError(f"{what} failed with exit code {done.returncode}{where}\n{(done.stderr or '')[-2000:]}")
        return done

    def compile(self):
        """Create the work library and compile the testbench (once per session).

        Input:  nothing. Output: nothing.
        Logic:  vlib work; vlog -F <src_list> +cover -covercells (-F: file paths relative to the list).
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._run([self.tool("vlib"), "work"], "vlib")
        self._run([self.tool("vlog"), "-F", self.host_path(self.src_list), "+cover", "-covercells"], "vlog")

    def launch(self, plan_file, seed, run_dir):
        """Input:  plan_file, seed, run_dir (inside work_dir; see SimulatorLauncher).
        Output: RunOutput with sim.log, coverage.txt and result.txt (None if the test wrote none).
        Logic:  1. vsim the plan in batch mode, timed (Questa start-up included)
                2. vcover the saved UCDB into the text report, timed
                3. a missing UCDB means the simulation did not save coverage -> SimulationError
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log, ucdb, report, result = (run_dir / n for n in ("sim.log", "run.ucdb", "coverage.txt", "result.txt"))
        for stale in (log, ucdb, report, result):  # a reused folder must not show an older test's files
            stale.unlink(missing_ok=True)

        start = time.perf_counter()
        self._run([self.tool("vsim"), "-c", "-sv_seed", str(seed), self.top, "-coverage", "-l", self.relative(log),
                   f"+UVM_TESTNAME={self.test_name}", f"+PLAN_FILE={self.relative(plan_file)}",
                   f"+RESULT_FILE={self.relative(result)}", "-do", f"coverage save -onexit {self.relative(ucdb)}; run -all; quit -f"], "vsim", log)
        sim_time = time.perf_counter() - start
        if not ucdb.is_file():
            raise SimulationError(f"vsim saved no coverage database {ucdb}; see {log}")

        start = time.perf_counter()
        done = self._run([self.tool("vcover"), "report", "-cvg", "-details", "-nocompactcrossbins",
                          self.relative(ucdb)], "vcover")
        report.write_text(done.stdout)
        report_time = time.perf_counter() - start
        return RunOutput(run_dir, Path(plan_file), report, result if result.is_file() else None, log,
                         sim_time=sim_time, report_time=report_time)
