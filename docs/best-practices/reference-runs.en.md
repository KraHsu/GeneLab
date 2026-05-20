# Reference Runs

This page is the **reproducibility ground truth** for GeneLab's bundled tasks.
It lists, per registered task and per seed, the converged return, the
convergence step count, and the wall-clock budget — the numbers you should
expect to land on when you `clone → train → eval` against the same
configuration.

> **Status — 2026-05-21**: cartpole and Franka tables are populated from
> the M1.7 reference batch (8×H200, Genesis 0.4.7, `QD_GRAPH=0`); the two
> G1 tables are still **running** (3 seeds × 30 k iter each, ETA ≈ 15–17 h
> from this commit) and will be filled in the follow-up commit. The
> Franka numbers are recorded as-is with a known anomaly — see the
> **Franka status note** under its table.

## Reference tasks

The five tasks tracked here cover GeneLab's bundled locomotion +
manipulation lines:

| Task ID | Backend (default agent) | Budget | Notes |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | Tiny cartpole; sanity smoke target. |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | Harder cartpole. |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 velocity tracking on flat ground. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 motion-tracking on flat ground. |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 2048 envs | Goal-conditioned manipulation; needs offline demo prefill (see protocol below). |

> **Note**: dev pruned the example to a single SAC+HER setup. The earlier
> `Cartesian-v0` / `skrl-v0` / `sb3-v0` / `sb3-her-v0` Franka variants no
> longer register and are intentionally not part of this set.

## Reproduction protocol

### Common path (4 of 5 tasks)

Cartpole + G1 tasks are rsl_rl PPO; their reference runs use the multi-seed
CLI directly:

```bash
# 1. Train three seeds (parallel=3 only for cartpole-sized tasks; G1 needs
#    parallel=1 on a single GPU to avoid OOM).
genelab train <TASK> \
    --seeds 1,2,3 --parallel <P> \
    --log_dir logs/reference/<TASK>/<DATE>

# 2. Deterministic eval against each seed's final checkpoint.
for s in 1 2 3; do
  genelab eval <TASK> \
    "logs/reference/<TASK>/<DATE>/seed_${s}/model_final.pt" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/<TASK>/<DATE>/seed_${s}/eval.json"
done
```

`eval.json` files are the source of truth for the table numbers.

### Franka SAC+HER path

`GeneLab-Franka-Pick-And-Place-v0` is goal-conditioned SAC+HER and needs an
offline demo prefill before training, otherwise the cold-start replay
buffer never sees a successful trajectory:

```bash
# 1. Collect demos via the scripted FSM (one-shot, seed-independent).
uv run python -m genelab_franka_pick_and_place.collect_demos \
    --num-envs 32 --steps 6400 \
    --out logs/reference/franka-pp/demos.npz

# 2. Train three seeds — each child reads the demo file via
#    GENELAB_SB3_DEMO_PATH (or set agent.demo_path in cfg).
GENELAB_SB3_DEMO_PATH=logs/reference/franka-pp/demos.npz \
  genelab train GeneLab-Franka-Pick-And-Place-v0 \
    --seeds 1,2,3 --parallel 1 \
    --log_dir logs/reference/franka-pp/<DATE>

# 3. Eval each seed's saved model.zip (SB3's native format).
for s in 1 2 3; do
  genelab eval GeneLab-Franka-Pick-And-Place-v0 \
    "logs/reference/franka-pp/<DATE>/seed_${s}/model.zip" \
    --num-envs 64 --episodes 100 --seed 0 \
    --out "logs/reference/franka-pp/<DATE>/seed_${s}/eval.json"
done
```

The Franka task **cannot** currently be exported via `genelab export` —
HER's `Dict` observation is on the M1.3 limitations list. Removing that
limitation is tracked as a future M-series follow-up.

### Hardware

One CUDA GPU (≥ 12 GB VRAM) for training. CPU-only eval works for the
deterministic rollout step but is much slower than GPU-vectorized.

## Reference numbers

### `GeneLab-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 39.944 | 0.026 | 150 | ~21 min | 10.3 s |
| 2 | 39.978 | 0.002 | 150 | ~20 min | 10.1 s |
| 3 | 39.991 | 0.001 | 150 | ~19 min | 10.1 s |

Eval `length_mean = 1000.0` for all seeds (episode hits the time-limit cap
without falling), so the policy is solved at the budget cap. `success_rate`
is `null` (task does not publish `extras["is_success"]`).

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 59.980 | 0.007 | 300 | ~85 min | 12.2 s |
| 2 | 59.986 | 0.003 | 300 | ~88 min | 14.2 s |
| 3 | 59.987 | 0.002 | 300 | ~85 min | 12.6 s |

Eval `length_mean = 1200.0` for all seeds. `success_rate` is `null` (same
reason as IP).

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | _running_ | — | — | — | — |
| 2 | _running_ | — | — | — | — |
| 3 | _running_ | — | — | — | — |

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | _running_ | — | — | — | — |
| 2 | _running_ | — | — | — | — |
| 3 | _running_ | — | — | — | — |

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER, demo-prefilled)

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence timestep | Train wall-clock |
|---|---|---|---|---|---|
| 1 | −95.996 | 19.595 | 0.11 | 1,843,200 (budget cap) | 56 s |
| 2 | −97.998 | 14.000 | 0.04 | 1,843,200 (budget cap) | 56 s |
| 3 | −92.199 | 26.519 | 0.11 | 1,843,200 (budget cap) | 57 s |

Eval `length_mean = 100.0` (fixed episode length).

> **Franka status note — anomalous, not converged**:
> The 2 M-timestep budget is consumed in **≈ 56 seconds wall-clock** on
> a single H200, after which `model.learn()` returns and the success
> rate sits in **0.04–0.11**, i.e. essentially the demo-prefill
> baseline. The interaction we have not yet root-caused is between
> SB3's `HerReplayBuffer` auto-scaling, the demo prefill, and the
> `learning_starts` threshold: the gradient-step loop appears to exit
> well before `total_timesteps` of useful exploration is consumed.
>
> These rows are recorded **as-is** so the table is reproducible from
> the current `feat/m1-multi-seed-refs` branch + Genesis 0.4.7 +
> `QD_GRAPH=0`. Fixing the SAC/HER interaction is **deferred to M2**;
> the numbers will be revisited there. Do not treat the Franka row as
> a converged reference — treat it as the lower-bound smoke baseline.

## Training curves

Curves are exported from TensorBoard once the reference runs land. The
expected location:

```
logs/reference/<TASK>/<DATE>/seed_<S>/
├── events.out.tfevents.*   # TensorBoard
├── ckpts/                  # checkpoints (or `model_<N>.pt` under the dir,
│                           #   depending on backend)
├── eval.json               # written by `genelab eval`
└── (optional) curves.png   # screenshot used in this doc
```

Until the runs are done, this section is intentionally empty — the schema
above is what populated PRs should match.

## Methodology notes

- **Seeds 1, 2, 3 are GeneLab's canonical triplet.** Any task in this doc
  that ships with different seeds should explain why (e.g. seed 0 hit a
  degenerate Genesis init on this task).
- **Eval seed is fixed at 0.** This ensures the deterministic eval rollout
  is the same trajectory across seeds and across re-runs of this protocol —
  the variance in `return_mean` then reflects training variance only.
- **No `success_rate` for locomotion** at this revision. Locomotion tasks
  ship without `extras["is_success"]`; the doc reports `null` rather than
  inventing a threshold. Manipulation tasks (Franka) emit `is_success`
  from the goal-reach termination, so the field is populated there.
- **Genesis version pin.** The version used to produce these numbers is
  recorded at the top of each `eval.json` (via `evaluated_at` and the
  `params/env.json` snapshot in the same directory). Re-running with a
  different Genesis is *not* expected to reproduce the numbers exactly.

## What this doc is not

- It is **not** a benchmark suite — that is M3.8 (`genelab benchmark`).
- It is **not** a leaderboard. The numbers here are GeneLab's own
  reproducibility check; community submissions go to the benchmark suite
  once it lands.
- It is **not** a tuning guide. See `best-practices/rl-experiments` for
  curriculum, DR, and reward weight choices that are upstream of these
  numbers.
