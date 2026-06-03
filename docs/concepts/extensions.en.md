# Extensions

Extensions are ordinary Python packages that register **robots, environments, tasks, and RL
backends** with GeneLab. They are the expected way to build real projects.

## Why extensions are separate

Keeping downstream projects outside `src/genelab/` avoids turning the framework into a collection of
project-specific code. It also lets teams version, install, and publish their robot packages
independently.

## The `genelab.extensions` API

`genelab.extensions` is the single, stable import path for everything an extension needs. The
underlying implementations live in `genelab.registry` (robots / envs / tasks) and
`genelab.rl.backends` (backends), but `genelab.extensions` is the canonical surface:

```python
from genelab.extensions import (
    register_robot, register_env, register_task, register_backend,
    ROBOTS, ENVS, TASKS,        # the registries themselves
    Runnable, Backend,          # the two Protocol contracts
)
```

| Extension kind | Register with | Stored in |
|---|---|---|
| Robot cfg | `register_robot(name, factory, *, description, …)` | `ROBOTS` |
| Env cfg | `register_env(name, factory, *, description, …)` | `ENVS` |
| Task | `register_task(name, factory, *, description, …)` | `TASKS` |
| RL backend | `register_backend(backend)` | keyed by `backend.cfg_type` |

Each `register_*` for robot / env / task takes a stable `name` and a zero-argument `factory` that
builds the config on demand (the registry stays lazy). `register_backend` takes a fully-built
backend object and keys it by the agent-config type it owns.

## Discovery mechanisms

| Mechanism | Best use |
|---|---|
| Entry point in the `genelab.extensions` group | Installed packages and daily workflows. |
| CLI `--import MODULE` | Temporary local modules or debugging entry-point loading. |
| Programmatic loading | Embedded applications. |

All mechanisms end with the same operation: a registration function calls `register_robot`,
`register_env`, `register_task`, or `register_backend`.

## The `Runnable` task contract

Every value a task factory produces must satisfy `genelab.extensions.Runnable` — the contract the
CLI's `play` / `train` commands dispatch against:

```python
class Runnable(Protocol):
    cfg: object
    def play(self, *, max_steps: int | None = None) -> None: ...
    def train(self) -> None: ...
```

A task type therefore exposes its config as `.cfg` and implements `play()` / `train()`. The bundled
task types already satisfy this; third-party task types should implement the same shape.

`play` takes a keyword-only `max_steps` — the hard cap forwarded from the `--max-steps` CLI flag.
When it is not `None` the playback loop must stop after that many steps regardless of the viewer or
the soft `simulation.steps` config; `None` (what a plain `task.play()` call passes) leaves the soft
config in charge (headless caps at `simulation.steps`, a viewer runs until the window closes). A task
that drives no step loop may accept and ignore it.

## The `Backend` contract

A backend owns one RL library and is selected by the **type** of the task's agent config (see
[RL runner](rl-runner.md)). The contract is `genelab.extensions.Backend`:

```python
@runtime_checkable
class Backend(Protocol):
    name: str
    cfg_type: type
    def train(self, ctx: TrainContext) -> Path: ...
    def play(self, ctx: PlayContext) -> None: ...
    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup: ...
```

`TrainContext` / `PlayContext` (in `genelab.rl.backends`) bundle the already-resolved env, configs,
seed, `log_dir`, and profiler knobs, so library-specific code stays inside the backend.

A minimal backend that registers itself and records a run looks like this:

```python
import json
from dataclasses import dataclass
from pathlib import Path

from genelab.extensions import register_backend
from genelab.rl.backends import PlayContext, TrainContext
from genelab.rl.backends.base import InferenceSetup
from genelab.rl.config import BackendConfig


@dataclass
class EchoAgentCfg(BackendConfig):
    """The agent-config type this backend owns; tasks set ``cfg.agent`` to an instance."""

    max_iterations: int = 10


class EchoBackend:
    name = "echo"
    cfg_type = EchoAgentCfg

    def train(self, ctx: TrainContext) -> Path:
        log_dir = ctx.log_dir or Path("logs") / ctx.task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run.json").write_text(
            json.dumps({"task": ctx.task_id, "iterations": ctx.max_iterations})
        )
        return log_dir

    def play(self, ctx: PlayContext) -> None:
        ctx.env.reset()

    def make_inference_setup(self, ctx: PlayContext) -> InferenceSetup:
        raise NotImplementedError("EchoBackend does not support eval / export")


def register() -> None:
    register_backend(EchoBackend())
```

Expose `register` as the package's entry point (or call it via `--import`), and any task whose
`cfg.agent` is an `EchoAgentCfg` routes to `EchoBackend`. `make_inference_setup` is only needed for
`genelab eval` / `genelab export`; a training-only backend may leave it unimplemented.

## Extension contract

An extension should be importable as a package, keep registry-time imports light, expose a
no-argument `register()` hook, and avoid duplicate registration in repeated loads.

## Where to continue

- [Build an Extension Project](../best-practices/extension-projects.md)
- [Project New](../cli/project-new.md)
- [Registry](registry.md)
- [RL runner](rl-runner.md)
