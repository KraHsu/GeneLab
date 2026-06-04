# Unitree G1 examples

GeneLab extension that ships three PPO tasks for the Unitree G1 humanoid:

- **`Genelab-Velocity-Flat-Unitree-G1-v0`** — velocity-tracking baseline on flat ground
  (follow a commanded body-frame twist).
- **`Genelab-Velocity-Rough-Unitree-G1-v0`** — the same velocity task on a RandomRough
  heightfield, with a torso height-scan observation and a terrain-difficulty curriculum.
- **`Genelab-Tracking-Flat-Unitree-G1-v0`** — motion imitation (BeyondMimic-style; tracks a
  recorded clip per-body).

Both tasks are ported from the equivalent ones in
[mjlab](https://github.com/Mujoco-Lab/mjlab) and adapted to Genesis.

## Layout

```
examples/unitree/
├── pyproject.toml
├── README.md
└── src/genelab_unitree/
    ├── tasks.py                     # registers all tasks into genelab's registry
    └── g1/
        ├── env_cfg.py               # velocity-tracking env cfg (flat; shared base helper)
        ├── ppo_cfg.py               # velocity-tracking PPO cfg
        ├── rough_env_cfg.py         # velocity-tracking env cfg (rough terrain)
        ├── rough_ppo_cfg.py         # velocity-tracking PPO cfg (rough terrain)
        ├── tracking_env_cfg.py      # motion-imitation env cfg
        └── tracking_ppo_cfg.py      # motion-imitation PPO cfg
```

The G1 robot itself (actuators, default pose, foot links, MJCF) is provided by
`genelab.asset_zoo.unitree_g1.UnitreeG1Cfg`, which fetches the MJCF + STL bundle from
the `genelab-assets` repository on first use and caches it under `.cache/`.

## Quickstart

```bash
# From the GeneLab repo root
uv sync --extra torch-cu128       # pick whichever torch flavor fits your GPU
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Velocity-Rough-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

### Velocity tracking

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 --num-envs 4096 --max-iterations 1500
uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

### Motion imitation

The tracking task consumes an NPZ motion clip (keys: `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`). The default clip is
the LAFAN1 retargeted `dance1_subject2` NPZ (~131 s @ 50 fps), fetched on first use by
`genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2()` into
`.cache/assets/g1_lafan1_dance1_subject2/<md5>/dance1_subject2.npz` — no manual download
required.

```bash
# Replay the reference clip frame-by-frame (no policy; robot snaps to motion each step).
# Useful for sanity-checking joint / body ordering before training.
uv run python -m genelab_unitree.replay_motion

# Train
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 30000

# Replay trained policy
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt
```

`--agent` accepts `zero`, `random`, or `trained`. Without `--agent`, play defaults to
`trained` when `--checkpoint` is set, else `zero`. Note that `--agent zero` just resets
the robot to the clip's first frame and applies zero torques — the robot will fall, not
follow the motion. Use the `replay_motion` script above to actually watch the clip play
out without a trained policy.

The bundled clip inherits its upstream license (CC BY-NC-ND 4.0 — non-commercial,
attribution required); see `unitree_g1/motions/LICENSE.NOTICE` in the
[genelab-assets](https://github.com/KraHsu/genelab-assets) repo.

#### Swapping the clip

Edit `examples/unitree/src/genelab_unitree/g1/tracking_env_cfg.py` and replace the
`motion_file=str(g1_lafan1_dance1_subject2())` line with any NPZ path matching the
schema above. The genelab-assets repo ships
`unitree_g1/motions/scripts/convert.sh`, a recipe that drives mjlab's `csv_to_npz`
forward-kinematics replay for arbitrary G1-retargeted CSVs.

## Long training runs

For multi-hour runs (the G1 tasks default to `max_iterations=30_000` ≈ several wall-clock
hours on 8×H200), set PyTorch's allocator to use growable segments **before** launching
training. This dramatically reduces fragmentation-driven slowdown over time:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If training time drifts upward across iterations anyway, capture a profiler trace by
re-running with `GENELAB_PROFILE=1` set. The trace lands under `logs/torch_profile/` and
can be loaded by `tensorboard --logdir logs/torch_profile`. Take one trace at the start
and one a few hours in to compare cumulative time per section. Optional env vars:
`GENELAB_PROFILE_OUT`, `GENELAB_PROFILE_WAIT`, `GENELAB_PROFILE_WARMUP`,
`GENELAB_PROFILE_ACTIVE`, `GENELAB_PROFILE_REPEAT` (see
`src/genelab/rl/_profiler.py`). Only rank 0 emits a trace under distributed launches.

## Notes

- The G1 MJCF and STL meshes are fetched on first use by
  `genelab.asset_zoo.unitree_g1.UnitreeG1Cfg` from the `genelab-assets` repository and
  cached under `.cache/`. They originate from Unitree's mujoco_menagerie release; refer
  to that repository for licensing details.
- The motion-imitation task drops mjlab's adaptive-bin failure sampling; only `start` and
  `uniform` sampling modes are wired up. Self-collision penalties are also omitted because
  GeneLab has no contact-sensor abstraction yet — the env still penalises action rate and
  joint-limit excursions.
- `policy` and `critic` observation groups differ in the tracking task: the critic gets the
  privileged per-body pose/orientation features that the actor does not see.
