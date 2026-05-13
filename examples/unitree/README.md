# Unitree G1 examples

GeneLab extension that ships two PPO tasks for the Unitree G1 humanoid on flat ground:

- **`Genelab-Velocity-Flat-Unitree-G1-v0`** — velocity-tracking baseline (follow a commanded
  body-frame twist).
- **`Genelab-Tracking-Flat-Unitree-G1-v0`** — motion imitation (BeyondMimic-style; tracks a
  recorded clip per-body).

Both tasks are ported from the equivalent ones in
[mjlab](https://github.com/Mujoco-Lab/mjlab) and adapted to Genesis.

## Layout

```
examples/unitree/
├── pyproject.toml
├── README.md
├── assets/g1/                       # vendored MJCF + meshes (~19 MB)
└── src/genelab_unitree/
    ├── tasks.py                     # registers both tasks into genelab's registry
    └── g1/
        ├── constants.py             # actuator gains, default pose, action scale
        ├── robot.py                 # G1RobotCfg factory
        ├── env_cfg.py               # velocity-tracking env cfg
        ├── ppo_cfg.py               # velocity-tracking PPO cfg
        ├── tracking_env_cfg.py      # motion-imitation env cfg
        └── tracking_ppo_cfg.py      # motion-imitation PPO cfg
```

## Quickstart

```bash
# From the GeneLab repo root
uv sync --extra rl --extra torch-cu128       # pick whichever torch flavor fits your GPU
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

### Velocity tracking

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 --num-envs 4096 --max-iterations 1500
uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

### Motion imitation

The tracking task needs a motion clip in mjlab's NPZ schema (keys: `joint_pos`, `joint_vel`,
`body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`). Convert your CSV with
[`mjlab.scripts.csv_to_npz`](https://github.com/Mujoco-Lab/mjlab/blob/main/src/mjlab/scripts/csv_to_npz.py),
then pass the resulting file via `--env.commands.motion.motion_file`:

```bash
# Visualise a clip with no policy (robot reset to clip frames, zero torques applied)
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent zero \
    --env.commands.motion.motion_file path/to/clip.npz \
    --vis

# Random-action sanity check (visible perturbation around the reference pose)
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent random \
    --env.commands.motion.motion_file path/to/clip.npz \
    --vis

# Train
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --env.commands.motion.motion_file path/to/clip.npz \
    --num-envs 4096 --max-iterations 30000

# Replay trained policy
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt \
    --env.commands.motion.motion_file path/to/clip.npz
```

`--agent` accepts `zero`, `random`, or `trained`. Without `--agent`, play defaults to
`trained` when `--checkpoint` is set, else `zero`.

## Notes

- The G1 MJCF and STL meshes are vendored under `assets/g1/`. They originate from Unitree's
  mujoco_menagerie release; refer to that repository for licensing details.
- The motion-imitation task drops mjlab's adaptive-bin failure sampling; only `start` and
  `uniform` sampling modes are wired up. Self-collision penalties are also omitted because
  GeneLab has no contact-sensor abstraction yet — the env still penalises action rate and
  joint-limit excursions.
- `policy` and `critic` observation groups differ in the tracking task: the critic gets the
  privileged per-body pose/orientation features that the actor does not see.
