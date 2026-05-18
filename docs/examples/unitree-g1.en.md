# Unitree G1

`examples/unitree/` ships two PPO tasks for the Unitree G1 humanoid on flat ground.
Both tasks are ports of the equivalent `mjlab` recipes adapted to Genesis.

## Tasks

| Task id | Problem |
|---------|---------|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | Velocity-tracking baseline — follow a commanded body-frame twist. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | Motion imitation — track a recorded clip per-body (BeyondMimic style). |

## Installation

The extension depends on the `rl` extra (rsl_rl). Pick the `torch-*` extra that matches
the host.

```bash
uv sync --extra rl --extra torch-cu128
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

## Velocity tracking

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 1500

uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 \
    --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt --vis
```

## Motion imitation

The tracking task consumes an NPZ motion clip (keys: `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w` — the schema mjlab's
`csv_to_npz` emits). The default clip is the LAFAN1 retargeted `dance1_subject2` NPZ
fetched on first use by `genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2`
and cached under `.cache/assets/g1_lafan1_dance1_subject2/<md5>/`. No manual download
required — the example runs out of the box.

```bash
# Replay the reference clip frame-by-frame (no policy; robot snaps to motion each step).
# The right way to visually verify the env wiring before training.
uv run python -m genelab_unitree.replay_motion

# Train.
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 30000

# Replay the trained policy.
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt
```

`--agent` accepts `zero`, `random`, or `trained`. Without `--agent`, play defaults to
`trained` when `--checkpoint` is set, otherwise `zero`. `zero` and `random` only reset
the robot to the clip's first frame and then apply zero / random torques — the robot
will fall, not follow the motion. Use the `replay_motion` script above for pure
reference playback.

### Swapping the clip

The default `motion_file` is wired in `tracking_env_cfg.unitree_g1_tracking_env_cfg`.
Replace the `g1_lafan1_dance1_subject2()` call with any NPZ matching the schema above,
and update `motion_body_order` / `motion_joint_order` on the
`MotionCommandCfg` if the new clip uses a different body / joint axis ordering. The
[`genelab-assets`](https://github.com/KraHsu/genelab-assets) repo ships
`unitree_g1/motions/scripts/convert.sh`, a recipe that drives mjlab's `csv_to_npz`
forward-kinematics replay for arbitrary G1-retargeted CSVs.

The bundled clip inherits its upstream license (CC BY-NC-ND 4.0 — non-commercial
research use, attribution required); see `unitree_g1/motions/LICENSE.NOTICE` in the
assets repo.

## Long training runs

For multi-hour training (the tracking task defaults to `max_iterations=30_000` —
several wall-clock hours on 8×H200), set PyTorch's allocator to use growable segments
**before** launching training. This dramatically reduces fragmentation-driven
slowdown over time.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If training time drifts upward across iterations anyway, capture a profiler trace by
re-running with `GENELAB_PROFILE=1`. The trace lands under `logs/torch_profile/` and
can be loaded with `tensorboard --logdir logs/torch_profile`. Take one trace at the
start and another several hours in to compare cumulative time per section. Optional
env vars: `GENELAB_PROFILE_OUT`, `GENELAB_PROFILE_WAIT`, `GENELAB_PROFILE_WARMUP`,
`GENELAB_PROFILE_ACTIVE`, `GENELAB_PROFILE_REPEAT` (see `src/genelab/rl/_profiler.py`).
Only rank 0 emits a trace under distributed launches.

!!! tip "Smoke-test budget"
    A 5–10 iteration run with `--num-envs 64 --max-iterations 5` is enough to validate
    the env wiring end-to-end. The reward signal stays noisy at that scale;
    convergence on the velocity task needs the 1500-iteration budget above.

## Notes

- The G1 MJCF and STL meshes are fetched on first use via
  `genelab.asset_zoo.unitree_g1.UnitreeG1Cfg` from the `genelab-assets` repository and
  cached under `.cache/`. They originate from Unitree's `mujoco_menagerie` release;
  refer to that repository for licensing details.
- The default motion NPZ (`dance1_subject2`) is similarly fetched from
  `genelab-assets` on first use. The clip stores bodies and joints in mjlab's MJCF
  DFS order; the env config permutes both axes into Genesis's traversal order via
  `motion_body_order` and `motion_joint_order` on `MotionCommandCfg`.
- The motion-imitation task drops mjlab's adaptive-bin failure sampling; only
  `start` and `uniform` sampling modes are wired up. Self-collision penalties are
  omitted because GeneLab has no contact-pair sensor abstraction yet — the env still
  penalises action rate and joint-limit excursions.
- `policy` and `critic` observation groups differ in the tracking task: the critic
  receives the privileged per-body pose / orientation features that the actor does
  not see.

## Logs

Both tasks write to `logs/rsl_rl/<experiment>/<timestamp>/` exactly like the inverted
pendulum examples:

- `params/env.json` and `params/agent.json` — frozen configs at run time.
- `model_<iter>.pt` — checkpoints saved every `save_interval` iterations.
- TensorBoard event files alongside the checkpoints.

## See also

- [Inverted Pendulum](inverted-pendulum.md)
- [Actuators](../concepts/actuators.md)
- [Play and Train CLI](../cli/play-train.md)
