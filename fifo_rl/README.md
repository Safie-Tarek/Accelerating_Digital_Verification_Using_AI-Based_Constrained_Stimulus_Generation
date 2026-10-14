# fifo_rl

RL agent for the FIFO testbench in `../src codes/`. Work in progress: state, actions, fake FIFO and the env exist; the DQN agent and the Questa backend do not yet.

| File | What it is |
|---|---|
| `fifo_rl/coverage_model.py` | The 64 coverage bins of `cg_FIFO`, in one fixed order (`BIN_NAMES`, `BIN_INDEX`, `N_BINS`) |
| `fifo_rl/state.py` | `StateEncoder`: 64 bin flags + budget left + one-hot last action = 87 numbers |
| `fifo_rl/actions.py` | The 22 actions (reset → fill to level L → pattern for K cycles): `ACTIONS`, `to_cycles`, `to_plusargs` |
| `fifo_rl/fake_fifo.py` | Python copy of the FIFO + monitor sampling: cycle list → 64-bin hits |
| `fifo_rl/backends/` | `run(action_id, seed) -> bool[64]`: `FakeBackend` (Python FIFO); Questa backend to come |
| `fifo_rl/reward.py` | `compute_reward`: (new bins − 1 + 10 on the run that closes 64/64) / 64; settings in `RewardConfig` |
| `fifo_rl/envs/coverage_env.py` | `CoverageEnv`: `reset()` / `step(action)` → state, reward, done, info |
| `fifo_rl/questa/report_parser.py` | `hits_vector(report_text)`: Questa coverage report → 64 true/false flags, one per bin |
| `tests/` | pytest tests for all of the above |
| `tests/fixtures/` | 4 real Questa reports used by the tests |

The report must be created with `vcover report -cvg -details -nocompactcrossbins run.ucdb > report.txt`.

## Setup (WSL Ubuntu)

```bash
python3 -m venv ~/venvs/fifo_rl
source ~/venvs/fifo_rl/bin/activate
cd fifo_rl
pip install -r requirements.txt
pytest -q
```
