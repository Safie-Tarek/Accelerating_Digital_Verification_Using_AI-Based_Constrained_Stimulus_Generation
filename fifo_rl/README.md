# fifo_rl

RL agent for the FIFO testbench in `../src codes/`. Work in progress: only the state part exists so far.

| File | What it is |
|---|---|
| `fifo_rl/coverage_model.py` | The 64 coverage bins of `cg_FIFO`, in one fixed order (`BIN_NAMES`, `BIN_INDEX`, `N_BINS`) |
| `fifo_rl/state.py` | `StateEncoder`: 64 bin flags + budget left + one-hot last action = 87 numbers |
| `tests/` | pytest tests for both |

## Setup (WSL Ubuntu)

```bash
python3 -m venv ~/venvs/fifo_rl
source ~/venvs/fifo_rl/bin/activate
cd fifo_rl
pip install -r requirements.txt
pytest -q
```
