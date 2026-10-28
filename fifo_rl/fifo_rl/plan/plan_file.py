"""The plan file: the contract between the Python agent and the UVM test (+PLAN_FILE=<path>).

    PLAN id=12 seed=3141592653 version=1
    ACT idx=0 name=WRITE wr=100 rd=0 rst=0 cycles=10
    ACT idx=1 name=IDLE wr=0 rd=0 rst=0 cycles=4
    END_TEST

One ACT line per block, idx = 0, 1, 2, ... in order; END_TEST closes the plan and has no line of
its own besides the marker. The test applies one reset cycle before the first block, then runs
`cycles` transactions per line with those percentages. `name` is for people (logs, debugging);
the UVM test needs only wr, rd, rst and cycles.

    TextPlanWriter().write(plan, path)   StimulusPlan -> file
    read_plan(path) -> PlanFile          file -> blocks (used by FakeLauncher, and to check files)
"""

import re
from dataclasses import dataclass
from pathlib import Path

FORMAT_VERSION = 1

_FIELD = re.compile(r"(\w+)=(\S+)")


class PlanFileError(ValueError):
    """The text is not a valid plan file."""


@dataclass(frozen=True)
class PlanBlock:
    idx: int
    name: str
    wr_pct: int
    rd_pct: int
    rst_pct: int
    n_cycles: int


@dataclass(frozen=True)
class PlanFile:
    plan_id: int
    seed: int
    version: int
    blocks: tuple

    def total_cycles(self):
        """Input: nothing. Output: cycles of all blocks (without the test's first reset)."""
        return sum(block.n_cycles for block in self.blocks)


class TextPlanWriter:
    def format(self, plan):
        """Input: a closed StimulusPlan. Output: the plan file text (ends with a newline)."""
        if not plan.is_closed:
            raise ValueError("only a closed plan (ending with END_TEST) can be written")
        lines = [f"PLAN id={plan.plan_id} seed={plan.seed} version={FORMAT_VERSION}"]
        for pa in plan.blocks:
            spec = pa.spec
            lines.append(f"ACT idx={pa.idx} name={spec.name} wr={spec.wr_pct} rd={spec.rd_pct} "
                         f"rst={spec.rst_pct} cycles={spec.n_cycles}")
        lines.append("END_TEST")
        return "\n".join(lines) + "\n"

    def write(self, plan, path):
        """Input: a closed StimulusPlan, the file path. Output: the path (parent folders are created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format(plan))
        return path


def _fields(line, required, line_no):
    """Input: one line, the keys it must have, its line number. Output: {key: int or str}."""
    values = dict(_FIELD.findall(line))
    missing = [key for key in required if key not in values]
    if missing:
        raise PlanFileError(f"line {line_no}: missing {missing}: {line!r}")
    out = {}
    for key, value in values.items():
        try:
            out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def parse_plan(text):
    """Read a plan file.

    Input:  the file text.
    Output: a PlanFile.
    Logic:  first non-empty line PLAN (id, seed, version), then ACT lines with idx 0, 1, 2, ...,
            percentages 0-100 and cycles >= 0, then END_TEST as the last line; anything else is
            a PlanFileError, so a broken file never reaches the simulator unnoticed.
    """
    lines = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not lines or not lines[0][1].startswith("PLAN "):
        raise PlanFileError("the first line must start with 'PLAN '")
    head = _fields(lines[0][1], ("id", "seed", "version"), lines[0][0])
    if lines[-1][1] != "END_TEST":
        raise PlanFileError("the last line must be END_TEST")
    blocks = []
    for line_no, line in lines[1:-1]:
        if not line.startswith("ACT "):
            raise PlanFileError(f"line {line_no}: expected an ACT line, got {line!r}")
        f = _fields(line, ("idx", "wr", "rd", "rst", "cycles"), line_no)
        if f["idx"] != len(blocks):
            raise PlanFileError(f"line {line_no}: idx {f['idx']}, expected {len(blocks)}")
        if not all(isinstance(f[k], int) and 0 <= f[k] <= 100 for k in ("wr", "rd", "rst")):
            raise PlanFileError(f"line {line_no}: wr, rd and rst must be whole numbers 0-100")
        if not isinstance(f["cycles"], int) or f["cycles"] < 0:
            raise PlanFileError(f"line {line_no}: cycles must be a whole number >= 0")
        blocks.append(PlanBlock(f["idx"], str(f.get("name", "")), f["wr"], f["rd"], f["rst"], f["cycles"]))
    return PlanFile(head["id"], head["seed"], head["version"], tuple(blocks))


def read_plan(path):
    """Input: path of a plan file. Output: its PlanFile."""
    return parse_plan(Path(path).read_text())
