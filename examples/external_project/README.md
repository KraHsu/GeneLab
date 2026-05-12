# External GeneLab Project Example

This directory is a minimal standalone Python package that extends GeneLab without putting project
code under `src/genelab/`. It owns its configs, robot definitions, environment runner, and task
registration hook.

## Why `--import` Needs Python Packaging Context

The CLI option `--import my_robot_project.tasks` imports a normal Python module. It does not search
every directory in the repository. Python must be able to import `my_robot_project` first, either
because the package is installed or because its `src/` directory is on `PYTHONPATH`.

## Run Without Installing

Try the example from the GeneLab repository root without installing the package:

```bash
PYTHONPATH=examples/external_project/src uv run genelab --import my_robot_project.tasks list tasks
PYTHONPATH=examples/external_project/src uv run genelab --import my_robot_project.tasks play MyProject-PickPlace-v0 --steps 3
```

The first command should list `MyProject-PickPlace-v0`. The second command should print:

```text
Run MyPickPlaceEnv with Genesis for 3 steps
```

## Install As An Editable Package

For day-to-day use, install your external project into the environment instead of setting
`PYTHONPATH` on every command. This example package declares a `genelab.extensions` entry point, so
after an editable install the CLI can discover it without `--import`:

```bash
uv pip install -e examples/external_project
uv run genelab list tasks
uv run genelab play MyProject-PickPlace-v0 --steps 3
```

If you no longer want this example project installed in the current environment:

```bash
uv pip uninstall my-robot-project
```

## Package Layout

```text
my_robot_project/
├── pyproject.toml
└── src/my_robot_project/
    ├── __init__.py
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

If GeneLab is not published in the package index you use, point your external project's
`pyproject.toml` at a local checkout:

```toml
[project]
name = "my-robot-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["genelab"]

[tool.uv.sources]
genelab = { path = "../GeneLab", editable = true }

[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project.tasks:register"
```

## Registration Hook

Register from your package with `genelab.registry`:

```python
# src/my_robot_project/tasks.py
from genelab.configs import TaskCfg
from genelab.registry import register_env, register_robot, register_task
from my_robot_project.config import MyEnvCfg, MyRobotCfg
from my_robot_project.envs import MyPickPlaceEnv
from my_robot_project.robots import create_robot


class MyPickPlaceTask:
    def __init__(self) -> None:
        self.cfg = TaskCfg(
            name="MyProject-PickPlace-v0",
            env_name="my-project-pick-place",
            robot_name="my-project-robot",
            env=MyEnvCfg(),
            trainable=False,
        )

    def play(self) -> None:
        MyPickPlaceEnv(self.cfg.env).play()

    def train(self) -> None:
        raise NotImplementedError("add your RL runner here")


def register() -> None:
    register_robot(
        "my-project-robot",
        create_robot,
        description="Robot provided by my_robot_project.",
        cfg_type=MyRobotCfg,
    )
    register_env(
        "my-project-pick-place",
        MyPickPlaceEnv,
        description="Pick-place environment provided by my_robot_project.",
        cfg_type=MyEnvCfg,
    )
    register_task(
        "MyProject-PickPlace-v0",
        MyPickPlaceTask,
        description="Pick-place task provided by my_robot_project.",
        cfg_type=TaskCfg,
    )
```

Entry points in the `genelab.extensions` group should point to a no-argument function that performs
registration. The CLI loads installed entry-point extensions first, then any modules passed with
`--import`. Use `--no-entry-points` only when you want to debug without installed extensions.

## Troubleshooting

- `ModuleNotFoundError: No module named 'my_robot_project'` means Python cannot import the external
  package. Install the package or run with `PYTHONPATH=/path/to/my_robot_project/src`.
- `unknown task 'MyProject-PickPlace-v0'` means the package may be importable, but its registration
  hook did not run. Use `--import my_robot_project.tasks`, or install an entry point in the
  `genelab.extensions` group and rerun `uv run genelab list tasks`.
