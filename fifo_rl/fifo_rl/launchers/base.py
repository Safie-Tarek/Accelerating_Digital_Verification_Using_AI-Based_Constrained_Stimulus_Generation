"""A launcher runs one test plan and says where the outputs are.

    output = launcher.launch(plan_file, seed, run_dir)    -> RunOutput

FakeLauncher (Python FIFO) and QuestaLauncher (real simulator) write the SAME files, so the rest of
the loop (result parser, report parser, reward) cannot tell them apart:
    run_dir/plan.txt     the plan (written before launch by TextPlanWriter)
    run_dir/sim.log      simulator log (Questa only)
    run_dir/result.txt   pass / errors / cycles (FakeLauncher always; Questa if the test writes it)
    run_dir/coverage.txt coverage report in the vcover -cvg -details -nocompactcrossbins format
"""

from dataclasses import dataclass
from pathlib import Path


class SimulationError(RuntimeError):
    """The simulator could not run the plan (crash, non-zero exit, missing output)."""


@dataclass(frozen=True)
class RunOutput:
    run_dir: Path
    plan_file: Path
    coverage_report: Path
    result_file: Path | None
    log_file: Path | None
    sim_time: float      # seconds of wall-clock spent simulating (Questa: start-up included)
    report_time: float   # seconds spent producing the coverage report (vcover); 0 for the fake

    @property
    def wall_time(self):
        """Input: nothing. Output: seconds the launch took in total."""
        return self.sim_time + self.report_time


class SimulatorLauncher:
    name = "base"

    def launch(self, plan_file, seed, run_dir):
        """Run one plan.

        Input:  plan_file, path of the plan text file; seed, the simulation seed (-sv_seed);
                run_dir, folder for this test's outputs.
        Output: a RunOutput.
        Logic:  none here - FakeLauncher and QuestaLauncher override it.
        """
        raise NotImplementedError
