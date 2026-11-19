# FIFO RL agent — fine-tuning results

> AI team notes. This file records the rounds of fine-tuning the plan-based DDQN agent (`train --resume`) on the fake FIFO, and the final check on real Questa. Background: [`fifo_agent_context.md`](fifo_agent_context.md) (STATUS 2026-11-01 and the resume section). Started 2026-11-17.

## Why fine-tuning
Plain resume (2026-11-16, `resume1k_seed10/11/12`) kept the original learning settings (lr 1e-3, buffer 10k, target sync 200) with ε 0.1. No validation beat the loaded agent, and all three runs collapsed (0–16% closure at campaign 1000). This file tries **gentle** settings instead: small lr, low ε, slow target sync, big buffer and frequent validation, and runs it in rounds:

1. Each round resumes the current best agent in 3 runs side by side with lr 1e-5 / 3e-5 / 1e-4, `--eps-start 0.05 --eps-end 0.05 --target-sync 1000 --buffer-capacity 100000 --n-campaigns 1000 --eval-every 50 --eval-campaigns 200`, `OMP_NUM_THREADS=1`.
2. Each run's `best.pt` is evaluated on two fixed fake sets of 300 campaigns: `--seed 0` (same set as earlier results) and `--seed 5` (holdout).
3. A candidate wins only if it is at least as good as the parent on **both** sets (closure first, then tests used). The winner is the parent of the next round.
4. Stop after 2 rounds without a winner, or after 5 rounds. Then evaluate the final agent on real Questa (30 campaigns, `--seed 1`).

Score = closure rate, then mean tests used (budget 30 per campaign), then cycles, as in `run_campaign.score`.

## Starting point: `long_same_seed1/best.pt`
| Set | Closed | Tests used (mean) | Tests to close (mean) | Cycles per campaign |
|---|---|---|---|---|
| fake `--seed 0`, 300 campaigns | 300/300 | 1.74 | 1.74 | 308 |
| fake `--seed 5`, 300 campaigns (holdout) | 298/300 | 1.94 | 1.76 | 318 |
| Questa `--seed 1`, 30 campaigns | 30/30 | 1.53 | 1.53 | 270 |

## Rounds

### Round 1 (2026-11-17): low lr, ε 0.05, parent `long_same_seed1/best.pt`. No winner.
Folders `runs/campaign/ft_r1_lr<lr>_s<seed>`. Stopped at campaign ~300 of 1000, because every run was getting worse (validation tests used kept rising).

Validation during training (200 campaigns, closure / tests used):

| Run | campaign 0 (parent) | 50 | 100 | 150 | 200 | 250 |
|---|---|---|---|---|---|---|
| lr 1e-5, seed 20 | 100% / 1.77 | 84% / 9.84 | 100% / 2.71 | 100% / 2.60 | 100% / 3.40 | 100% / 3.18 |
| lr 3e-5, seed 21 | 99.5% / 1.87 | **100% / 2.23** | 91.5% / 6.00 | 100% / 3.12 | 99% / 4.15 | 99% / 4.26 |
| lr 1e-4, seed 22 | 99.5% / 1.86 | 96% / 4.21 | 100% / 3.38 | 100% / 3.74 | 99.5% / 3.46 | 100% / 2.85 (best at 350) |

Each `best.pt` on the two fixed sets (closed / tests used / cycles per campaign):

| Candidate | fake `--seed 0` | fake `--seed 5` (holdout) | Verdict |
|---|---|---|---|
| parent | 300 / 1.74 / 308 | 298 / 1.94 / 318 | – |
| lr 1e-5 (= parent, campaign 0) | 300 / 1.74 / 308 | 298 / 1.94 / 318 | same agent |
| lr 3e-5 (campaign 50) | 300 / 2.27 / 247 | **300** / 2.25 / 243 | closes all holdout campaigns, but +0.5 tests |
| lr 1e-4 (campaign 350) | 300 / 2.72 / 321 | **300** / 2.75 / 324 | +1 test |

**Finding.** Even lr 1e-5 (about 500 gradient steps) shifted the greedy policy. The Q-values of the competing blocks are probably very close, so tiny weight changes flip the argmax. With ε 0.05 the new replay buffer holds almost only the agent's own choices, so the other blocks' Q-values get no corrective data. Fine-tuning bought closure on hard campaigns (300/300 on the holdout) at the cost of ~0.5 extra tests, which is not a win, since tests are what costs time on Questa.

**Selection fix (from round 2 on).** `run_campaign.score` ranks closure rate first. In round 2 that replaced a 99.5% / 1.77-test agent with a 100% / 3.86-test one. From round 2 on, runs go through a small driver outside the repo (no repo code changed) that:
- picks `best.pt` on **mean tests used** (a failed campaign counts as the full budget of 30, so this is the expected cost of a campaign), then cycles;
- keeps every validation's agent as `ckpt_vNNN.pt` (NNN = validation number, in `eval.csv` row order).

### Round 2 (2026-11-17): more exploration, even smaller lr. No winner.
Folders `runs/campaign/ft_r2b_*`, 600 campaigns, `--target-sync 1000 --buffer-capacity 100000 --eval-every 50 --eval-campaigns 200`.

Validation (closure / tests used):

| Run | 0 (parent) | 50 | 100 | 200 | 300 | 400 | 500 | last |
|---|---|---|---|---|---|---|---|---|
| ε 0.3, lr 1e-5, seed 31 | 100% / 1.60 | 100% / 4.25 | 100% / 3.90 | 100% / 3.56 | 98.5% / 3.26 | 100% / 3.08 | 100% / 3.18 | – |
| ε 0.3, lr 3e-6, seed 32 | 99.5% / 1.775 | **99.5% / 1.765** | 100% / 3.86 | 100% / 3.06 | 100% / 2.33 | 100% / 3.09 | 99.5% / 3.72 | 97% / 3.92 (550) |
| ε 0.15, lr 3e-6, seed 33 | 100% / 1.625 | 100% / 1.69 | 100% / 3.24 | 100% / 3.40 | 100% / 2.32 | 100% / 3.03 | 99.5% / 3.39 | 99% / 3.36 (600) |

Only candidate: seed 32 at campaign 50. On the fixed sets it is the parent within noise: fake `--seed 0` 300/300, 1.757 tests (parent 1.743); fake `--seed 5` 298/300, 1.957 tests (parent 1.943). **Not a winner.**

**Finding.** Every fine-tune setting, even lr 3e-6 with ε 0.3, moves from ~1.7 tests to ~3 tests within 100 campaigns and stays there. The parent is a sharp peak that the learning dynamics do not hold: with the reward only at the end of each test, the stable point of training is a ~3-test policy. Two rounds without a winner → fine-tuning stops here.

### Round 3 (2026-11-17): fresh training, best-of-seeds. No winner.
The parent was itself the best checkpoint of 8 fresh runs, so this round tried that again, with checkpoints kept at every validation: `fresh_r3_same_s40` and `fresh_r3_same_s41` (the parent's settings: `--gamma 0.99 --eps-decay-steps 80000`, lr 1e-3, 3000 campaigns, validation on 100 campaigns every 100), and `fresh_r2b_steady_s30` (`--lr 1e-4 --buffer-capacity 100000 --target-sync 1000`). Wall time ~2.5 h for the three side by side, longer than the ~1 h expected, because stuck policies use the whole 30-test budget in every validation campaign.

Validation (closure / tests used), every 100 campaigns:
- seed 30 (steady): best 100% / 2.53 at campaign 100. Later between 0% and 100%, with good points at 1300 (100% / 3.08) and 2300 (100% / 2.57). Ended at 100% / 4.95.
- seed 40: best 100% / 2.50 at campaign 1200. Mostly 0–37% closure in between. Ended at 100% / 3.36.
- seed 41: 0% for most of the run. Best 100% / 5.61 at campaign 2100. Ended at 2%.

Each `best.pt` on the fixed sets (closed / tests used / cycles per campaign):

| Candidate | fake `--seed 0` | fake `--seed 5` (holdout) |
|---|---|---|
| parent `long_same_seed1/best.pt` | 300 / **1.74** / 308 | 298 / **1.94** / 318 |
| seed 30, campaign 100 | 300 / 2.57 / 269 | 300 / 2.44 / 259 |
| seed 40, campaign 1200 | 297 / 2.71 / 372 | 295 / 2.90 / 391 |
| seed 41, campaign 2100 | 300 / 5.68 / 477 | 299 / 5.73 / 486 |

## Final result
**After 3 rounds (9 training runs), no agent beats the starting one, so `long_same_seed1/best.pt` stays the agent to use.** It was evaluated on real Questa on 2026-11-16 (`runs/campaign/evalq_resume1k_best`, same weights): **30/30 campaigns closed in 1.53 tests on average (270 cycles per campaign), 46 of 46 simulations passed.** No new agent was found, so no new Questa run was needed.

| Agent | fake `--seed 0` | fake `--seed 5` | Questa `--seed 1` |
|---|---|---|---|
| **parent `long_same_seed1/best.pt` (kept)** | 300/300, 1.74 tests | 298/300, 1.94 tests | 30/30, 1.53 tests |
| best fine-tune (round 2, ε 0.3 lr 3e-6, campaign 50) | 300/300, 1.76 | 298/300, 1.96 | – |
| best fine-tune for closure (round 1, lr 3e-5, campaign 50) | 300/300, 2.27 | 300/300, 2.25 | – |
| best fresh run (round 3, seed 30, campaign 100) | 300/300, 2.57 | 300/300, 2.44 | – |

## What this tells us
1. **The step-1 learning signal cannot hold a ~1.7-test policy.** Every training run (plain resume, 6 fine-tune settings, 3 fresh runs) settles around 3 tests or collapses. The reward comes only at `END_TEST`, so the agent cannot tell which block of a plan helped. The good agents are lucky snapshots, not a stable result, and more training of any kind moves away from them.
2. **The parent is already near what this signal can find.** Best-of-8 fresh runs gave 1.74. Best-of-3 more gave 2.44–2.57. Finding a better snapshot by luck would need many more runs, with a small expected gain (the floor is 1 test, the 9-block directed plan).
3. **Checkpoint selection should use expected tests, not closure first.** `run_campaign.score` ranks closure rate first, so one extra closed campaign out of 200 outweighs doubling the tests. Suggested repo change: score on mean tests used (failed campaign = full budget), and keep every validation checkpoint. This round used a driver outside the repo for that.
4. **Next step that can actually improve the agent: step 2 (a reward for each block).** It needs the per-block covergroup instances from the verification team (see `fifo_agent_context.md`, "Step 2 requests"). Training more on the step-1 signal is not worth the CPU time.

## Files
- Runs: `fifo_rl/runs/campaign/` (git-ignored): `ft_r1_*`, `ft_r2b_*`, `fresh_r2b_steady_s30`, `fresh_r3_same_s40/41` (each with `eval.csv`, `best.pt`; `ckpt_vNNN.pt` from round 2 on), and evaluations `evalfake0_*` / `evalfake5_*`. `ft_r2_*` and `fresh_r2_steady_s30` are aborted first starts of round 2, before the selection fix; ignore them.
- Questa test folders: `C:/Users/20114/fifo_questa_work` (outside OneDrive).
