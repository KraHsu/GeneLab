# How to Build an Extension Project

In most cases, your robot, task, or experiment should not live directly under `src/genelab/`.
Use an extension project instead. This is the recommended shape for downstream GeneLab research
code.

## 1. Generate a scaffold

```bash
genelab project new my_robot_project
```

The generated package has:

```text
my_robot_project/
├── pyproject.toml
└── src/my_robot_project/
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

## 2. Keep responsibilities separate

| File | Responsibility |
|---|---|
| `config.py` | Dataclasses owned by the project. |
| `robots.py` | Robot asset/config factories. |
| `envs.py` | Runtime env classes or env factory helpers. |
| `tasks.py` | `register()` hook and task classes. |

Avoid importing Genesis at module import time unless the object truly needs it. Registry discovery
should stay cheap.

## 3. Register through one hook

Expose a no-argument `register()` function:

```python
def register() -> None:
    register_robot(...)
    register_env(...)
    register_task(...)
```

Guard repeated registration when the hook may run more than once:

```python
if "my-robot" not in ROBOTS:
    register_robot(...)
```

## 4. Prefer entry points for daily use

In `pyproject.toml`:

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project.tasks:register"
```

Then install as editable:

```bash
uv pip install -e my_robot_project
genelab list tasks
```

Use explicit imports for temporary experiments:

```bash
PYTHONPATH=my_robot_project/src \
  genelab --import my_robot_project.tasks list tasks
```

## 5. Verify the package boundary

Run these from outside the package directory:

```bash
python -c "import my_robot_project"
genelab list tasks
genelab info MyProject-Example-v0
genelab play MyProject-Example-v0 --steps 3
```

## Expected result

The project can be imported as a normal Python package, GeneLab discovers its entry point, and all
task commands work without editing GeneLab core.
