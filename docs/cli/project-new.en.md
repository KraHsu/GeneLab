# Project: project new

`genelab project new` creates a standalone extension package. Use it for downstream projects rather
than editing `src/genelab/`. `project` is a Typer subcommand group: `genelab project` (with no
subcommand) lists the available actions.

## Usage

```bash
genelab project new my_robot_project
```

Options:

| Option | Description |
|---|---|
| `--path PATH`, `-p PATH` | Parent directory for the generated project (default `.`). |
| `--package NAME` | Python package name. Defaults to normalized `NAME`. |
| `--task-id ID` | Initial task id. Defaults to `<PackageName>-Example-v0`. |
| `--force` | Overwrite scaffold files when the target directory exists and is non-empty. Targets that exist but are not directories still error out. |

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

The generated `pyproject.toml` references the local GeneLab source by relative path
(`tool.uv.sources`), so `uv pip install -e .` works without publishing anything.

## After scaffolding

```bash
uv pip install -e my_robot_project
genelab list tasks
genelab play MyRobotProject-Example-v0 --steps 3
```

If the target directory is non-empty and `--force` is not set, the command exits non-zero without
overwriting any existing file.

## See also

- [Build an Extension Project](../best-practices/extension-projects.md)
- [Extensions](../concepts/extensions.md)
