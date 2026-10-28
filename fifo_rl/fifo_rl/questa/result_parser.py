"""Did the test pass? Read from result.txt, or from the simulator log when there is no result.txt.

result.txt (written by the UVM test with +RESULT_FILE=<path>; one line of key=value):
    passed=1 errors=0 fatals=0 cycles=23 actions=2

Without result.txt the UVM report summary at the end of the Questa log is used:
    # UVM_ERROR :    0
    # UVM_FATAL :    0
and every "** Error" / "** Fatal" line (assertion failures from FIFO_SVA print like that) counts as
a failed assertion. A log without the UVM summary means the simulation did not finish.

The result only decides whether a test is usable; the coverage comes from the coverage report.
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

_FIELD = re.compile(r"(\w+)=(\S+)")
_UVM_COUNT = re.compile(r"^\s*(?:#\s*)?(UVM_ERROR|UVM_FATAL)\s*:\s*(\d+)\s*$", re.MULTILINE)
_TOOL_ERROR = re.compile(r"^\s*(?:#\s*)?\*\* (?:Error|Fatal)", re.MULTILINE)


class ResultError(ValueError):
    """Neither a result file nor a log was found, or the result file is malformed."""


@dataclass(frozen=True)
class RunResult:
    passed: bool
    errors: int                # UVM_ERROR count (scoreboard mismatches, ...)
    fatals: int = 0            # UVM_FATAL count
    assert_fails: int = 0      # failed SVA assertions / other tool errors
    cycles: int | None = None  # simulated cycles, if the test reports them
    n_actions: int | None = None
    source: str = "result"     # "result" (result.txt) or "log"
    message: str = ""


class ResultParser:
    def parse(self, result_file=None, log_file=None):
        """Input:  result_file, path of result.txt (may be None or missing);
                log_file, path of the simulator log (may be None or missing).
        Output: a RunResult.
        Logic:  result.txt wins when it exists; otherwise the log; neither -> ResultError.
                SVA assertion failures never reach the UVM counters in result.txt, so when the log
                exists too its "** Error" lines are added to the result.
        """
        if result_file is not None and Path(result_file).is_file():
            result = self.parse_result_text(Path(result_file).read_text())
            if log_file is not None and Path(log_file).is_file():
                tool_errors = len(_TOOL_ERROR.findall(Path(log_file).read_text(errors="replace")))
                if tool_errors > result.assert_fails:
                    result = replace(result, passed=False, assert_fails=tool_errors)
            return result
        if log_file is not None and Path(log_file).is_file():
            return self.parse_log_text(Path(log_file).read_text(errors="replace"))
        raise ResultError(f"no result file ({result_file}) and no log ({log_file})")

    def parse_result_text(self, text):
        """Input: result.txt text. Output: RunResult (passed must be given; counts default to 0)."""
        values = dict(_FIELD.findall(text))
        if "passed" not in values:
            raise ResultError(f"result file has no 'passed=' field: {text.strip()!r}")
        try:
            numbers = {k: int(v) for k, v in values.items()}
        except ValueError as exc:
            raise ResultError(f"result file values must be whole numbers: {text.strip()!r}") from exc
        errors = numbers.get("errors", 0)
        fatals = numbers.get("fatals", 0)
        assert_fails = numbers.get("assert_fail", 0)
        passed = bool(numbers["passed"]) and errors == 0 and fatals == 0 and assert_fails == 0
        return RunResult(passed, errors, fatals, assert_fails, numbers.get("cycles"), numbers.get("actions"),
                         source="result")

    def parse_log_text(self, text):
        """Input: simulator log text. Output: RunResult from the UVM summary and the ** Error lines."""
        counts = {}
        for name, value in _UVM_COUNT.findall(text):
            counts[name] = int(value)  # the last summary wins if the log has more than one
        assert_fails = len(_TOOL_ERROR.findall(text))
        if "UVM_ERROR" not in counts:
            return RunResult(False, 0, 0, assert_fails, source="log",
                             message="no UVM report summary in the log: the simulation did not finish")
        errors, fatals = counts["UVM_ERROR"], counts.get("UVM_FATAL", 0)
        passed = errors == 0 and fatals == 0 and assert_fails == 0
        return RunResult(passed, errors, fatals, assert_fails, source="log")
