# Project new

`genelab project new` scaffolds a downstream extension package — a self-contained Python
project that registers robots, environments, and tasks into GeneLab's global registries.

## Usage

```bash
uv run genelab project new my_robot_project
```

### Options

| Option | Default | Effect |
|--------|---------|--------|
| `--path PATH` | `./<name>` | Output directory. Created if missing. |
| `--package NAME` | derived from project name | Python import name (e.g. `my_robot_project`). |
| `--task-id ID` | `<package>/<name>-v0` | The first task ID registered by the scaffold. |
| `--force` | off | Overwrite an existing target directory. Use with care. |

## Scaffold output

```
my_robot_project/
├── pyproject.toml        # with [project.entry-points."genelab.extensions"]
├── README.md
└── src/
    └── my_robot_project/
        ├── __init__.py   # exposes register() entry-point callable
        ├── config.py     # task-specific dataclass plugged into TaskCfg.env
        ├── robots.py     # robot registrations
        ├── envs.py       # environment registrations
        └── tasks.py      # task registration (using --task-id)
```

The generated `pyproject.toml` declares:

```toml
[project.entry-points."genelab.extensions"]
my_robot_project = "my_robot_project:register"
```

Once installed (`uv pip install -e ./my_robot_project`), the extension is discovered on the
next CLI invocation without any `--import` flag.

## Post-scaffold workflow

```bash
cd my_robot_project
uv pip install -e .          # install the extension into the GeneLab venv
uv run genelab list tasks    # confirm the new task ID shows up
uv run genelab play <task-id> --vis
```

## See also

- [Extensions](../concepts/extensions.md)
- [Configs](../concepts/configs.md)
