# Quickstart

## 1. Listing registered content

The core `genelab` package ships empty registries — robots, environments, and tasks are
contributed by extension packages. The example extension under `examples/genelab_examples/` is
discovered automatically because its `pyproject.toml` declares a `genelab.extensions` entry
point.

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

An empty registry means no extension has been imported. Install one with `uv pip install`, or
pass `--import MODULE` on the command line.

## 2. Running a task

```bash
uv run genelab play <task-id>
```

The CLI resolves `<task-id>` against the `TASKS` registry, constructs its `TaskCfg`, applies any
command-line overrides, and runs the rollout in the configured Genesis backend.

The three scene shortcuts:

```bash
uv run genelab play <task-id> --vis           # enable visualization
uv run genelab play <task-id> --steps 500     # cap the episode length
uv run genelab play <task-id> --gpu 1         # pin to a single GPU
```

Arbitrary overrides use the dotted `--a.b.c VALUE` syntax — strings are coerced to the type
declared on the target dataclass field:

```bash
uv run genelab play <task-id> \
  --env.simulation.dt 0.005 \
  --env.actions.scale 0.5
```

## 3. Training

When a task ships an RL runner (`rsl_rl` and similar), training is launched with:

```bash
uv run genelab train <task-id>
uv run genelab train <task-id> --gpus 4       # multi-GPU via torchrun
uv run genelab train <task-id> --checkpoint path/to/model.pt
```

!!! warning "Distributed training"
    `--gpus N` transparently dispatches through `torchrun` for distributed training; the target
    task's runner must be `torchrun`-compatible.

## 4. Scaffolding a project

A standalone downstream package is generated with:

```bash
uv run genelab project new my_robot_project
```

This produces `config.py`, `robots.py`, `envs.py`, `tasks.py`, and a `pyproject.toml` that
declares a `genelab.extensions` entry point — the same shape as the bundled example extension.

## 5. Advanced: end-to-end RL on Unitree G1 { #unitree-g1 }

The repository ships a production-grade RL example under `examples/unitree/` — two PPO tasks on
the Unitree G1 humanoid (velocity tracking and motion imitation), ported from mjlab and adapted
to Genesis. The following walks through install, training, and checkpoint replay in one
sitting.

### 5.1 Install the extension

The Unitree extension pulls in vendored MJCF + meshes (~19 MB). Pick the `torch-*` extra that
matches your hardware — see *Installation* for the full extras table.

```bash
uv sync --extra torch-cu128
uv pip install -e examples/unitree

uv run genelab list tasks
# -> Genelab-Velocity-Flat-Unitree-G1-v0
# -> Genelab-Tracking-Flat-Unitree-G1-v0
```

### 5.2 Velocity tracking PPO

The velocity-tracking task asks the G1 to follow a commanded body-frame twist. Train, then
replay the resulting checkpoint:

```bash
uv run genelab train Genelab-Velocity-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 1500

uv run genelab play  Genelab-Velocity-Flat-Unitree-G1-v0 \
    --checkpoint logs/rsl_rl/g1_velocity_flat/<run>/model_1500.pt
```

`--checkpoint` makes `play` route through the RL runner with `--agent trained` by default.

### 5.3 Motion imitation

The tracking task imitates a recorded clip per-body (BeyondMimic-style). The default clip
is the LAFAN1 retargeted `dance1_subject2` NPZ, fetched on first use from `genelab-assets`
via `genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2()` and cached under
`.cache/`. No manual download required.

```bash
# Replay the reference clip frame-by-frame (no policy; robot snaps to motion each step).
uv run python -m genelab_unitree.replay_motion

# Train.
uv run genelab train Genelab-Tracking-Flat-Unitree-G1-v0 \
    --num-envs 4096 --max-iterations 30000

# Replay trained policy.
uv run genelab play Genelab-Tracking-Flat-Unitree-G1-v0 \
    --agent trained \
    --checkpoint logs/rsl_rl/g1_tracking_flat/<run>/model_30000.pt
```

### 5.4 `--agent` modes

`--agent` selects the policy source for `play`:

| Value | Policy source |
|-------|--------------|
| `zero` | Constant zero action — basic sanity check; robot resets then falls under gravity. |
| `random` | Uniform-random action sampling — visible perturbation around the reference pose. |
| `trained` | Load from `--checkpoint`. This is the default whenever `--checkpoint` is set. |

## See also

- [Configs](../concepts/configs.md)
- [Extensions](../concepts/extensions.md)
- [Examples](../examples/overview.md)
