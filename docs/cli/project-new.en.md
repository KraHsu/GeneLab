# Project New

`genelab project new` creates a standalone extension package. Use it for downstream projects rather
than editing `src/genelab/`.

## Usage

```bash
uv run genelab project new my_robot_project
```

Options:

| Option | Description |
|---|---|
| `--path PATH`, `-p PATH` | Parent directory for the generated project. |
| `--package NAME` | Python package name. Defaults to normalized `NAME`. |
| `--task-id ID` | Initial task id. Defaults to `<PackageName>-Example-v0`. |
| `--force` | Overwrite scaffold files if the target exists. |

## Generated structure

```text
my_robot_project/
├── README.md
├── pyproject.toml
└── src/my_robot_project/
    ├── __init__.py
    ├── config.py
    ├── envs.py
    ├── robots.py
    └── tasks.py
```

## After scaffolding

```bash
uv pip install -e my_robot_project
uv run genelab list tasks
uv run genelab play MyRobotProject-Example-v0 --steps 3
```

## See also

- [Build an Extension Project](../best-practices/extension-projects.md)
- [Extensions](../concepts/extensions.md)
