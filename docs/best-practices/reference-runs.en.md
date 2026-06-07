# Reference Runs

The **reproducibility ground truth** for GeneLab's bundled tasks.
It lists, per registered task and per seed, the converged return, the
convergence step count, and the wall-clock budget — the numbers to
expect from a `clone → train → eval` against the same
configuration.

## Reference tasks

The six tasks tracked here cover GeneLab's bundled locomotion +
manipulation lines:

| Task ID | Backend (default agent) | Budget | Notes |
|---|---|---|---|
| `GeneLab-Inverted-Pendulum-v0` | rsl_rl PPO | 150 iter × 4096 envs | Tiny cartpole; sanity smoke target. |
| `GeneLab-Double-Inverted-Pendulum-v0` | rsl_rl PPO | 300 iter × 4096 envs | Harder cartpole. |
| `Genelab-Velocity-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 velocity tracking on flat ground. |
| `Genelab-Velocity-Rough-Unitree-G1-v0` | rsl_rl PPO | 6k iter × 4096 envs | Unitree G1 velocity tracking on a 10-level mixed-terrain curriculum. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | rsl_rl PPO | 30k iter × 4096 envs | Unitree G1 motion-tracking on flat ground. |
| `Genelab-Velocity-Flat-Unitree-Go1-v0` | rsl_rl PPO | 3k iter × 4096 envs | Unitree Go1 quadruped velocity tracking on flat ground; deployable proprioception-only actor (no base-linear-velocity sensor). |
| `Genelab-Velocity-Rough-Unitree-Go1-v0` | rsl_rl PPO | (WIP) | Unitree Go1 on the 10-level mixed-terrain curriculum. **Not yet a reference** — from-scratch 3k iters stalls in a stand-still optimum (~1–2 % of commanded speed); needs a larger budget (~6k) + an easier curriculum bootstrap. |
| `Genelab-Velocity-Flat-Unitree-Go2W-v0` | rsl_rl PPO | 3k iter × 4096 envs | Unitree Go2-W **wheeled** quadruped velocity tracking on flat ground; position-controlled legs + velocity-controlled wheels, deployable proprioception-only actor. |
| `GeneLab-Franka-Pick-And-Place-v0` | sb3 SAC + HER | 2M timesteps × 64 envs | Goal-conditioned manipulation; needs offline demo prefill (see protocol below). |

## Reproduction protocol

### Common path (5 of 6 tasks)

Cartpole + G1 tasks are rsl_rl PPO; their reference runs use the multi-seed
CLI:

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
#    --num-envs must match the task's train num_envs (currently 64); the
#    prefill loader asserts the shapes match.
python -m genelab_franka.collect_demos \
    --num-envs 64 --steps 1000 \
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

The Franka task **cannot** currently be exported via `genelab export`.
Export supports flat-tensor observations only, while SAC+HER uses a
goal-conditioned `Dict` observation.

### Hardware

One CUDA GPU (≥ 12 GB VRAM) for training. CPU-only eval works for the
deterministic rollout step but is much slower than GPU-vectorized.

!!! warning "Run the sim on the GPU backend"
    `SimulationCfg.gpu` defaults to **`False` (CPU backend)**. With the CPU backend the
    physics steps on the CPU while the policy/tensors sit on the GPU, leaving the GPU idle
    and training **~50–100× slower** (contact-heavy tasks like G1 go from a few s to hundreds
    of s per iteration). Bundled trainable tasks set `gpu=True`; **custom tasks must do the
    same**. If `nvidia-smi` shows the training GPU near 0 % during steps, this is almost
    certainly why.

!!! note "Hopper (H100/H200) and multi-GPU caveats"
    - On **Hopper (SM 90)**, set `QD_GRAPH=0` (Genesis ships no SM 90 `graph_do_while`
      fatbin); this disables CUDA-graph batching and badly slows **contact-heavy** sims. Prefer
      a non-Hopper GPU (Ada / Ampere) for locomotion reproduction.
    - Multi-GPU (`genelab train --gpus N`) gives **little speedup for G1** (per-step cost +
      PCIe all-reduce dominate). For a multi-seed sweep, run **one seed per GPU** rather than
      one seed across many GPUs.
    - RL training at 4096 envs is largely **CPU-bound** and wants the whole host; running many
      such trainings concurrently on one box oversubscribes the CPU and slows them
      super-linearly. Wall-clock suffers but rewards are deterministic, so reproduced numbers
      are unaffected by contention.

## Reference numbers

The tables below are the **v1.0** Genesis numbers. The v0.4.7 numbers are
preserved in a "Previous: v0.4.7" admonition next to each task so a
re-baseline diff is one scroll away.

!!! note "Hardware the v1.0 numbers were collected on"

    - **Cartpole IP / DIP, Franka** — NVIDIA H200 (141 GB HBM3), driver
      570.211.01, CUDA 12.8, `QD_GRAPH=0` (Hopper requires it — no
      `graph_do_while` fatbin for SM 90). Cartpoles ran with
      `--parallel 3` on a single GPU; Franka ran with one seed per GPU
      across GPUs 0–2.
    - **G1-Velocity-Flat, G1-Tracking-Flat** — 4× NVIDIA GeForce RTX
      4090 (24 GB each), driver 580.159.03, CUDA 12.8. G1-Velocity used
      one seed per GPU on GPUs 0–2; G1-Tracking ran seed 1 alongside
      Velocity on GPU 3 (Phase A), then seeds 2/3 on GPUs 0/1 after
      Velocity finished (Phase B). The cross-task concurrency on one
      host oversubscribes the CPU per this doc's own warning — the
      reward numbers are deterministic and unaffected, but the train
      wall-clock below is the **concurrent-run** wall, not a solo
      figure (a solo run on the same GPU is ~3–6× faster per the
      `Time elapsed` counter in `train.log`).
    - **Go1-Velocity-Flat**, **Go2W-Velocity-Flat** — single NVIDIA
      GeForce RTX 5060 Ti (16 GB), local workstation. **Single-seed**
      runs (seed 42, the task's cfg default) rather than the 3-seed
      cluster sweep used for the G1 tasks — smoke-grade references, not
      variance estimates.

### `GeneLab-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 39.977 | 0.002 | 150 | ~2.5 min | 12.6 s |
| 2 | 39.994 | 0.001 | 150 | ~2.5 min | 12.4 s |
| 3 | 39.986 | 0.001 | 150 | ~2.5 min | 12.6 s |

Eval `length_mean = 1000.0` for all seeds (episode hits the time-limit cap
without falling), so the policy is solved at the budget cap. `success_rate`
is `null` (task does not publish `extras["is_success"]`).

!!! note "Previous: v0.4.7"

    | Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 39.944 | 0.026 | 150 | ~21 min | 10.3 s |
    | 2 | 39.978 | 0.002 | 150 | ~20 min | 10.1 s |
    | 3 | 39.991 | 0.001 | 150 | ~19 min | 10.1 s |

### `GeneLab-Double-Inverted-Pendulum-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 59.933 | 0.020 | 300 | ~3.5 min | 16.3 s |
| 2 | 59.914 | 0.174 | 300 | ~3.5 min | 16.4 s |
| 3 | 59.897 | 0.023 | 300 | ~3.5 min | 16.4 s |

Eval `length_mean = 1200.0` for all seeds. `success_rate` is `null` (same
reason as IP).

!!! note "Previous: v0.4.7"

    | Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 59.980 | 0.007 | 300 | ~85 min | 12.2 s |
    | 2 | 59.986 | 0.003 | 300 | ~88 min | 14.2 s |
    | 3 | 59.987 | 0.002 | 300 | ~85 min | 12.6 s |

### `Genelab-Velocity-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 112.038 | 4.816 | 30 000 | ~28 h | 163.5 s |
| 2 | 112.871 | 4.918 | 30 000 | ~28 h | 160.3 s |
| 3 | 113.163 | 4.850 | 30 000 | ~28 h | 160.9 s |

Eval `length_mean = 1000.0` for all seeds (play_env `episode_length_s =
20 s` × 50 Hz). `success_rate` is `null`.

!!! note "Previous: v0.4.7"

    | Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 112.419 | 4.647 | 30 000 | ~18.7 h | 143.0 s |
    | 2 | 93.417 | 3.921 | 30 000 | ~20.6 h | 161.0 s |
    | 3 | 92.028 | 4.162 | 30 000 | ~19.8 h | 156.9 s |

### `Genelab-Velocity-Rough-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 83.95 | 23.86 | 6 000 | ~4.8 h | 145 s |
| 2 | 87.17 | 12.47 | 6 000 | ~4.9 h | 145 s |
| 3 | 84.41 | 17.67 | 6 000 | ~4.9 h | 145 s |

Eval `length_mean` per seed = 899 / 966 / 935 (of 1000 max; play_env
`episode_length_s = 20 s` × 50 Hz) — the policy walks ~90–97 % of full episodes on
the mixed rough terrain. `success_rate` is `null`. Eval terrain seed = 0
(deterministic). The curriculum self-balances at terrain level ~4.5; training runs
to convergence at 6k with no de-learning (action std holds at its 0.3 floor
throughout). Hardware: H200 (one seed per GPU); `QD_GRAPH=0` adds ~2× per-step
cost on Hopper SM90.

> Maintainer sweep protocol: `genelab train
> Genelab-Velocity-Rough-Unitree-G1-v0 --seeds 1,2,3 --parallel 1 --log_dir
> logs/reference/Genelab-Velocity-Rough-Unitree-G1-v0/<DATE>`, one seed per RTX
> 4090 GPU (4090 cluster). Eval with `genelab eval ... --num-envs 64 --episodes
> 100 --seed 0 --out <seed_dir>/eval.json` and `QT_QPA_PLATFORM=offscreen`
> (headless Qt — recording extras crash without this even after eval forces
> vis=False).

### `Genelab-Tracking-Flat-Unitree-G1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 1 | 138.444 | 0.006 | 30 000 | ~28 h | 227.3 s |
| 2 | 137.980 | 0.008 | 30 000 | ~29 h | 301.6 s |
| 3 | 138.060 | 0.004 | 30 000 | ~29 h | 236.9 s |

Eval `length_mean = 1500.0`. The tracking play_env normally sets
`episode_length_s = 1e9` for infinite viewer playback; `genelab eval`
clamps that to 30 s, so 30 s × 50 Hz = 1500 steps per episode, all hitting
the cap without termination. Very tight std across seeds — the converged
policy follows the motion clip on track under the 30 s window. `success_rate`
is `null`.

!!! note "Previous: v0.4.7"

    | Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|
    | 1 | 137.800 | 0.005 | 30 000 | ~20.8 h | 212.8 s |
    | 2 | 138.047 | 0.004 | 30 000 | ~20.6 h | 216.8 s |
    | 3 | 138.122 | 0.007 | 30 000 | ~20.9 h | 216.0 s |

### `Genelab-Velocity-Flat-Unitree-Go1-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 42 | 56.533 | 2.264 | 3 000 | ~55 min | 99.4 s |

Single seed (42, the task's cfg default) on one RTX 5060 Ti — see the hardware
note; this is a smoke-grade reference, not the 3-seed sweep, so there is no
cross-seed variance figure. Eval `length_mean = 1000.0` (play_env
`episode_length_s = 20 s` × 50 Hz — full episodes, no falls). `success_rate` is
`null`. The actor is **proprioception-only** (no base linear velocity — real Go1
has no such sensor); an asymmetric critic gets the privileged true base velocity
during training. A direction-binned rollout (256 envs, fixed commands) confirms
symmetric tracking — forward/backward/lateral/yaw all land at 88–97 % of the
commanded speed.

### `Genelab-Velocity-Rough-Unitree-Go1-v0` (work in progress)

No reference numbers yet. Trained from scratch at 3k iters the policy stalls in a
stand-still optimum — it stays upright on the curriculum terrain but translates at
only ~1–2 % of the commanded speed (the stand-still posture already collects
partial velocity-tracking credit, and 3k iters from scratch never escapes it; the
penalties are *not* the cause — foot-slip / undesired-contact terms stay
negligible). The env wiring is correct (asymmetric critic + 187-ray height scan +
`terrain_levels` curriculum, all smoke-tested); it is a training-budget /
bootstrapping gap. Converging it likely needs a larger budget (~6k, matching the
G1 rough task) plus an easier curriculum level-0 so a from-scratch policy can
bootstrap basic locomotion before the terrain hardens. Tracked separately, like
the G1 rough task was before it landed.

### `Genelab-Velocity-Flat-Unitree-Go2W-v0`

| Seed | Final `return_mean` | `return_std` | Convergence iter | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|
| 42 | 51.876 | 2.848 | 3 000 | ~97 min | 40.9 s |

Single seed (42) on one RTX 5060 Ti (hardware note) — smoke-grade, not a 3-seed
sweep. Eval `length_mean = 997.5` (of 1000; near-full episodes, almost no falls).
`success_rate` is `null`. Go2-W is a **wheeled** quadruped: the actor commands
position targets for the 12 leg joints and *velocity* targets for the 4 wheels
(the ``mdp.JointVelocityAction`` path), with a deployable proprioception-only actor
(no base linear velocity) + privileged critic, as on Go1. A direction-binned rollout
(256 envs, fixed commands) confirms symmetric tracking — forward / backward 95 %,
lateral ~89–90 %, yaw ~96 % of the commanded speed. It converged **from scratch in a
single run** (the wheels make translation natural — no tuning iterations, unlike the
Go1 flat task).

### `GeneLab-Franka-Pick-And-Place-v0` (SAC+HER, demo-prefilled)

| Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence timestep | Train wall-clock | Eval wall-clock |
|---|---|---|---|---|---|---|
| 1 | −6.122 | 13.307 | 0.990 | 2 000 000 (budget cap) | ~67 min | 10.7 s |
| 2 | −7.260 | 14.695 | 0.990 | 2 000 000 (budget cap) | ~62 min | 10.4 s |
| 3 | −8.790 | 18.275 | 0.970 | 2 000 000 (budget cap) | ~64 min | 11.1 s |

Eval `length_mean = 100.0` (fixed episode length). `success_rate` reflects
the goal-reach termination from the manipulation task. Cross-seed mean
`success_rate ≈ 0.983 ± 0.012` — meaningfully tighter than the v0.4.7
sweep, where one seed only reached 0.89 from end-effector drift on harder
goal poses. All three v1.0 seeds land in the same tight band.

!!! note "Previous: v0.4.7"

    | Seed | Final `return_mean` | `return_std` | `success_rate` | Convergence timestep | Train wall-clock | Eval wall-clock |
    |---|---|---|---|---|---|---|
    | 1 | −19.264 | 33.334 | 0.89 | 2 000 000 (budget cap) | ~68 min | 15.3 s |
    | 2 |  −4.626 |  8.297 | 1.00 | 2 000 000 (budget cap) | ~63 min | 14.3 s |
    | 3 |  −4.102 |  7.644 | 1.00 | 2 000 000 (budget cap) | ~64 min | 18.7 s |

    Mean `success_rate ≈ 0.963 ± 0.052` across the three seeds; the two
    perfect seeds reflect a fully solved policy, the 0.89 seed still
    misses ~11 % of episodes from end-effector orientation drift on
    harder goal poses.

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

- It is **not** a benchmark suite or leaderboard. The numbers here are
  GeneLab's own reproducibility check.
- It is **not** a tuning guide. See `best-practices/rl-experiments` for
  curriculum, DR, and reward weight choices that are upstream of these
  numbers.
