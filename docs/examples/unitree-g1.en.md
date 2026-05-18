# Unitree G1

The Unitree G1 example is the advanced robot path after the inverted pendulum tutorial. It shows how
GeneLab scales to humanoid locomotion and motion imitation.

## Tasks

| Task id | Shows |
|---|---|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | Velocity tracking on flat ground. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | Reference motion tracking. |

## Install

```bash
uv pip install -e examples/unitree
uv run genelab list tasks
```

## Velocity tracking

```bash
uv run genelab play Genelab-Velocity-Flat-Unitree-G1-v0 --vis --steps 500
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 1500
```

Replay:

```bash
uv run genelab play Genelab-Velocity-Flat-Unitree-G1-v0 \
  --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

## Motion imitation

```bash
uv run python -m genelab_unitree.replay_motion
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
  --num_envs 4096 \
  --max_iterations 30000
```

The default clip is fetched through the asset zoo and cached locally.

## See also

- [RL experiments](../best-practices/rl-experiments.md)
- [Asset zoo](../concepts/asset_zoo.md)
