# Unitree G1 Velocity Tracking

GeneLab extension that trains a velocity-tracking PPO policy for the Unitree G1 humanoid on flat
ground. The robot, MDP terms, and PPO hyperparameters are ported from the equivalent task in
[mjlab](https://github.com/Mujoco-Lab/mjlab) and adapted to Genesis.

## Layout

```
examples/unitree/
├── pyproject.toml
├── README.md
├── assets/g1/                  # vendored MJCF + meshes (~19 MB)
└── src/genelab_unitree/
    ├── tasks.py                # registers the task into genelab's registry
    └── g1/
        ├── constants.py        # actuator gains, default pose, action scale
        ├── robot.py            # G1RobotCfg factory
        ├── env_cfg.py          # ManagerBasedRlEnvCfg (velocity tracking)
        └── ppo_cfg.py          # RslRlOnPolicyRunnerCfg
```

## Quickstart

```bash
# From the GeneLab repo root
uv sync --extra rl --extra torch-cu128       # pick whichever torch flavor fits your GPU
uv pip install -e examples/unitree

uv run genelab list tasks                    # -> Genelab-Velocity-Flat-Unitree-G1-v0
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 --num-envs 4096 --max-iterations 1500
uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

## Notes

- The G1 MJCF and STL meshes are vendored under `assets/g1/`. They originate from Unitree's
  mujoco_menagerie release; refer to that repository for licensing details.
- `policy` and `critic` observation groups are identical in v1 — privileged terms (foot
  contacts, terrain scans, angular momentum) will land in `critic` once GeneLab grows the
  matching Genesis sensor wrappers.
- The default `max_iterations=3000` is a quick-train default. For mjlab-parity convergence,
  bump to 30000.
