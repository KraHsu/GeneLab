---
hide:
  - toc
---

# Building a First Robot Experiment with GeneLab

Starting from a clean checkout, the smallest useful GeneLab loop runs end to end:
install the environment, inspect the registries, run the inverted pendulum task, understand how a
task reaches the Genesis backend, override configuration, launch a short training run, and scaffold
a downstream extension project. The Unitree G1 tasks follow as the next step toward a
full robot workflow. The end state is a clear picture of GeneLab's shape and where to
continue into detailed module, API, and best-practice documentation.

The intended reader knows Python and the command line, but is new to GeneLab. No prior understanding
of every internal module is required; each step has a concrete result to verify.

## What this builds

The loop runs the bundled `GeneLab-Inverted-Pendulum-v0` task. It is small, but it covers the core
GeneLab path:

```text
CLI
└── TASKS registry
    └── TaskCfg
        ├── env / play_env: ManagerBasedRlEnvCfg
        ├── robot: ArticulationCfg
        ├── manager terms: actions, observations, rewards, terminations, events
        └── agent: RslRlOnPolicyRunnerCfg
            └── Genesis-backed ManagerBasedRlEnv
```

Larger tasks such as Unitree G1, terrain curricula, sensors, and recording use the same route with a
larger robot, richer manager terms, and a more complete runner configuration.

## 1. Preparing the Environment

From the repository root, sync dependencies:

```bash
uv sync --extra torch-cpu
genelab --version
```

With an NVIDIA GPU, replace `torch-cpu` with the matching `torch-cu126`, `torch-cu128`, or
`torch-cu130` extra. Pick exactly one `torch-*` extra.

Expected result:

```text
genelab 0.1.0
```

Create project-local cache directories used by Genesis and plotting backends:

```bash
genelab cache
```

If dependency sync fails or torch is wrong, read [Installation](getting-started/installation.md).

## 2. Inspecting What GeneLab Knows

The core `genelab` package provides the framework. Robots, environments, and tasks are registered by
the asset zoo or extension packages. List the three registries:

```bash
genelab list robots
genelab list envs
genelab list tasks
```

The output lists bundled asset-zoo robots and any example tasks discovered through entry points. If
the task list is empty, explicitly import the inverted pendulum extension:

```bash
genelab --import genelab_inverted_pendulum.tasks list tasks
```

The registries are the first architecture boundary:

| Registry | Contents | Typical consumer |
|---|---|---|
| `ROBOTS` | Robot or asset config factories | Env configs, CLI `list robots` |
| `ENVS` | Environment factories | CLI `list envs`, downstream projects |
| `TASKS` | Task factories carrying env, play env, and agent cfg | `genelab play`, `genelab train` |

## 3. Running the First Task

Start with a short rollout to verify construction and stepping:

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 32
```

On a local desktop, open the Genesis viewer:

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
```

This step follows this path:

1. The CLI loads entry-point extensions and explicit `--import` modules.
2. The CLI resolves `GeneLab-Inverted-Pendulum-v0` in `TASKS`.
3. The task returns a `TaskCfg`; when present, `play_env` is preferred for viewer-friendly runs.
4. `ManagerBasedRlEnv` builds the Genesis scene, robot articulation, sensors, and managers.
5. The play loop produces actions and calls `env.step(action)`.

## 4. Inspecting and Overriding Task Configuration

Use `info` to inspect task metadata and overridable paths:

```bash
genelab info GeneLab-Inverted-Pendulum-v0
```

Look for `Overridable cfg paths`. Every CLI `--a.b.c VALUE` is applied to a dataclass field under
`TaskCfg`, with type coercion handled by `apply_overrides`.

Change the timestep and rollout length:

```bash
genelab play GeneLab-Inverted-Pendulum-v0 \
  --steps 128 \
  --env.simulation.dt 0.005
```

Common shortcut flags are rewritten to config paths. In `play`, they target `play_env` when the task
defines one; otherwise they target `env`.

| Shortcut | Play target with `play_env` | Train target |
|---|---|---|
| `--vis` | `play_env.simulation.vis=true` | `env.simulation.vis=true` |
| `--gpu` | `play_env.simulation.gpu=true` | `env.simulation.gpu=true` |
| `--steps N` | `play_env.simulation.steps=N` | `--max_iterations N` |
| `--dt X` | `play_env.simulation.dt=X` | `env.simulation.dt=X` |

## 5. Reading the Task Source Against the Architecture

The inverted pendulum registration entry point is
`examples/inverted_pendulum/src/genelab_inverted_pendulum/tasks.py`. It does three things:

```python
register_robot("inverted-pendulum", get_inverted_pendulum_robot_cfg, ...)
register_env("inverted-pendulum-env", lambda: ManagerBasedRlEnv(...), ...)
register_task("GeneLab-Inverted-Pendulum-v0", InvertedPendulumTask, ...)
```

The environment config lives in
`examples/inverted_pendulum/src/genelab_inverted_pendulum/single/env_cfg.py`. It shows GeneLab's
manager-based MDP shape:

```python
actions_cfg={
    "cart": JointPositionActionCfg(...)
}
observations_cfg={
    "policy": ObservationGroupCfg(terms={...}),
    "critic": ObservationGroupCfg(terms={...}),
}
rewards_cfg={
    "alive": RewardTermCfg(...),
    "pole_upright": RewardTermCfg(...),
}
terminations_cfg={
    "time_out": TerminationTermCfg(...),
    "pole_fell": TerminationTermCfg(...),
}
events_cfg={
    "reset_joints": EventTermCfg(mode="reset", ...)
}
```

This is the main GeneLab capability: RL environments are assembled from replaceable terms instead
of hiding everything inside one large `step()` function.

## 6. Launching a Short Training Run

The inverted pendulum task ships with an RSL-RL runner config. Run a tiny training job to verify the
RL pipeline can construct the env, wrap it as a VecEnv, and write logs/checkpoints:

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

After it finishes, inspect the log root:

```bash
ls logs/rsl_rl
```

The main process writes:

```text
params/env.json
params/agent.json
model_*.pt
```

Replay a trained policy by passing a checkpoint:

```bash
genelab play GeneLab-Inverted-Pendulum-v0 \
  --agent trained \
  --checkpoint logs/rsl_rl/<experiment>/<run>/model_*.pt
```

For environment health checks, use `--agent zero` or `--agent random`.

## 7. Creating a Downstream Extension Project

GeneLab expects real projects to live in their own Python packages rather than under
`src/genelab/`. Generate a scaffold:

```bash
genelab project new my_robot_project
```

The scaffold contains:

```text
my_robot_project/
├── pyproject.toml
└── src/my_robot_project/
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

The key is the entry point in `pyproject.toml`:

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project.tasks:register"
```

Install it as editable and the CLI discovers it automatically:

```bash
uv pip install -e my_robot_project
genelab list tasks
```

Without installing, import it explicitly:

```bash
PYTHONPATH=my_robot_project/src \
  genelab --import my_robot_project.tasks list tasks
```

## 8. Continuing to Unitree G1

The Unitree G1 example uses the same architecture with a humanoid robot, larger action and
observation spaces, command sampling, RSL-RL training, and motion imitation.

Install the extension:

```bash
uv pip install -e examples/unitree
genelab list tasks
```

The task list now includes:

```text
Genelab-Velocity-Flat-Unitree-G1-v0
Genelab-Tracking-Flat-Unitree-G1-v0
```

Run a visual smoke test before training:

```bash
genelab play Genelab-Velocity-Flat-Unitree-G1-v0 --vis --steps 500
```

The two Unitree tasks demonstrate the advanced side of GeneLab:

| Task | Demonstrates |
|---|---|
| `Genelab-Velocity-Flat-Unitree-G1-v0` | Velocity commands, humanoid action scaling, PPO training, checkpoint replay. |
| `Genelab-Tracking-Flat-Unitree-G1-v0` | Motion-command loading, reference replay, body pose tracking rewards. |

Read [Unitree G1](examples/unitree-g1.md) for the full training and replay commands.

## 9. GeneLab Capability Map

This completes the smallest useful loop and shows where the larger robot example fits.
GeneLab can be understood in four layers:

| Layer | Modules | Role |
|---|---|---|
| Discovery and dispatch | `genelab.registry`, `genelab.cli` | Register robots/envs/tasks; discover, run, and train from the CLI |
| Configuration and MDP | `genelab.configs`, `genelab.managers`, `genelab.mdp` | Dataclass configs, overrides, action/obs/reward/termination/event/curriculum terms |
| Simulation objects | `genelab.scene`, `genelab.entity`, `genelab.actuator`, `genelab.sensor`, `genelab.terrains` | Genesis scenes, articulations, rigid objects, actuators, sensors, terrains |
| Training and experiments | `genelab.rl`, `genelab.recording`, `genelab.bridges` | RSL-RL training/playback, recording, live plots, keyboard or GUI control |

## Next Steps

- For more commands, read [CLI overview](cli/overview.md) and [Play and Train](cli/play-train.md).
- For architecture, read [Module map](reference/module-map.md) and
  [Managers and MDP terms](concepts/managers.md).
- To write a task, read [Task design](best-practices/task-design.md) and
  [Extensions](concepts/extensions.md).
- To look up APIs, read [API Reference](api/reference.md).
- For complete demos, read [Examples](examples/overview.md).
