# How to Design a GeneLab Task

This guide shows how to structure a task that is easy to run, inspect, train, and extend.

## 1. Start from the task id

Pick a stable id before writing code:

```text
<Project>-<Robot>-<Objective>-v0
```

Examples:

- `GeneLab-Inverted-Pendulum-v0`
- `Genelab-Velocity-Flat-Unitree-G1-v0`
- `MyProject-PickPlace-v0`

Use the same id in `TaskCfg.name`, the `TASKS` registration, logs, and README commands.

## 2. Keep `env` and `play_env` separate

Use `env` for training defaults and `play_env` for human inspection.

| Config | Typical defaults |
|---|---|
| `env` | Many envs, viewer off, curriculum and disturbances on, training sensors only. |
| `play_env` | One env, viewer on, mouse/bridge controls on, reduced randomization, optional plots. |

Do not make users remember long overrides just to open a viewer. `genelab play TASK --vis` should
work with the task's play config.

## 3. Put robot construction behind a factory

Register a robot factory instead of a prebuilt object:

```python
register_robot(
    "my-robot",
    get_my_robot_cfg,
    description="My robot.",
    cfg_type=MyRobotCfg,
)
```

Factories keep imports light and allow the CLI to list metadata without booting Genesis.

## 4. Organize manager terms by intent

Use descriptive names in every manager dict. These names become CLI override paths and log keys.

```python
rewards_cfg = {
    "track_lin_vel": RewardTermCfg(...),
    "action_rate": RewardTermCfg(...),
    "feet_slip": RewardTermCfg(...),
}
```

Avoid names like `r1`, `penalty2`, or `tmp`. A good name is stable enough to appear in a paper,
run config, or dashboard.

## 5. Make observations explicit

Use at least a `policy` group. Add a `critic` group when the runner needs privileged information.

```python
observations_cfg = {
    "policy": ObservationGroupCfg(terms=policy_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
}
```

Keep corruption, scale, and clip settings in the term cfg so they are visible through
`genelab info TASK`.

## 6. Validate with small rollouts

Before long training, run:

```bash
uv run genelab info TASK_ID
uv run genelab play TASK_ID --steps 32
uv run genelab play TASK_ID --agent random --steps 64
uv run genelab train TASK_ID --num_envs 64 --max_iterations 2
```

This catches registration, scene construction, action dimension, observation shape, reward, and
runner wiring issues before a long job.

## Expected result

A well-designed task has a stable task id, a clear `TaskCfg`, a viewer-friendly `play_env`, readable
manager term names, documented overrides, and a short smoke command that exercises the full path.
