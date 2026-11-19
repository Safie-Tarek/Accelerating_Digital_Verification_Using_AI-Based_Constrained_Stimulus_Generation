"""Reference: run the ORIGINAL FIFO_test on Questa with the eval campaigns' seeds and find the cycle where it closes 64/64.

Run from fifo_rl/ (Questa tools in C:/questasim64_2024.1/win64, MGLS_LICENSE_FILE set, runs/questa compiled):
    python -m scripts.run_fifo_test <n_campaigns> <out name>       -> runs/questa/tests/<out name>/summary.json

For each campaign c: seed = seed of test 0 of eval campaign c (base seed 2000, the same as the agents).
  1. vsim +UVM_TESTNAME=FIFO_test +UVM_VERBOSITY=UVM_HIGH: the monitor prints every sampled cycle
  2. the printed signals (Questa's own values) -> 64 bins cycle by cycle -> first cycle with 64/64,
     checked against Questa's coverage report
  3. vsim again without UVM_HIGH for a clean wall time; vcover report must say 64/64
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fifo_rl.fake_fifo import Sample, sample_bins
from fifo_rl.questa.report_parser import hits_vector

BIN = "C:/questasim64_2024.1/win64/"
WORK = Path("runs/questa").resolve()
LINE = re.compile(r"env\.agt\.mon.*rst_n = 0b(\d), wr_en = 0b(\d), rd_en = 0b(\d), data_out = 0b\d+, wr_ack = 0b(\d), "
                  r"overflow = 0b(\d), full = 0b(\d), empty = 0b(\d), almostfull = 0b(\d), almostempty = 0b(\d), "
                  r"underflow = 0b(\d)")


def vsim(out, seed, tag, verbose):
    """Input: output folder, seed, campaign number, UVM_HIGH or not. Output: (log, report text, sim s, report s)."""
    d = out / f"c{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rel = d.relative_to(WORK).as_posix()
    name = "high" if verbose else "plain"
    args = [BIN + "vsim.exe", "-c", "-sv_seed", str(seed), "work.FIFO_rl_top", "-coverage", "-l", f"{rel}/{name}.log",
            "+UVM_TESTNAME=FIFO_test", "-do", f"coverage save -onexit {rel}/{name}.ucdb; run -all; quit -f"]
    if verbose:
        args.insert(-2, "+UVM_VERBOSITY=UVM_HIGH")
    t = time.perf_counter()
    subprocess.run(args, cwd=WORK, capture_output=True, text=True, check=True)
    sim = time.perf_counter() - t
    t = time.perf_counter()
    rep = subprocess.run([BIN + "vcover.exe", "report", "-cvg", "-details", "-nocompactcrossbins", f"{rel}/{name}.ucdb"],
                         cwd=WORK, capture_output=True, text=True, check=True).stdout
    (d / f"{name}_coverage.txt").write_text(rep)
    return d / f"{name}.log", rep, sim, time.perf_counter() - t


def main(n, out_name):
    out = WORK / "tests" / out_name
    if (out / "summary.json").exists():
        raise SystemExit(f"{out} already holds results (never overwritten)")
    rows = []
    for c in range(n):
        seed = int(np.random.SeedSequence([2000, c, 0]).generate_state(1)[0])
        log, rep_high, _, _ = vsim(out, seed, c, True)
        seen, close_at, n_cycles, curve = np.zeros(64, bool), None, 0, {}
        for line in log.read_text().splitlines():
            m = LINE.search(line)
            if not m:
                continue
            v = [int(x) for x in m.groups()]
            n_cycles += 1
            seen[sample_bins(Sample(*v[1:]))] = True
            if n_cycles in (100, 200, 300, 400, 500, 1000):
                curve[n_cycles] = int(seen.sum())
            if close_at is None and seen.all():
                close_at = n_cycles
        assert (seen == hits_vector(rep_high)).all(), "log replay != Questa report"
        _, rep, sim, report = vsim(out, seed, c, False)
        rows.append({"campaign": c, "seed": seed, "cycles_run": n_cycles, "cycles_to_64": close_at,
                     "bins_final": int(hits_vector(rep).sum()), "bins_at": curve,
                     "sim_time": round(sim, 2), "report_time": round(report, 2), "wall_time": round(sim + report, 2)})
        print(rows[-1], flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
