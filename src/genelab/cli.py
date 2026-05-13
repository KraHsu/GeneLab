"""Unified command-line entry point for registered GeneLab tasks."""

import argparse
from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
import re
from textwrap import dedent
from typing import Final, Literal, Protocol, cast, runtime_checkable

from genelab import __version__
from genelab.cache import CACHE_DIR, ensure_project_cache
from genelab.configs import apply_overrides
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    load_entrypoint_extensions,
    load_extension_module,
)

RegistryKind = Literal["robots", "envs", "tasks"]


class _TaskCfgLike(Protocol):
    env_name: str
    robot_name: str
    trainable: bool


@runtime_checkable
class _TaskLike(Protocol):
    cfg: _TaskCfgLike


@runtime_checkable
class _CfgBacked(Protocol):
    cfg: object


class _RunnableTask(Protocol):
    cfg: object

    def play(self) -> None: ...

    def train(self) -> None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="genelab", description="Run registered GeneLab tasks.")
    parser.add_argument("--version", action="version", version=f"genelab {__version__}")
    parser.add_argument(
        "--import",
        dest="extension_modules",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a downstream extension module before dispatch (can be repeated).",
    )
    parser.add_argument(
        "--no-entry-points",
        action="store_true",
        help="Skip installed extensions from the genelab.extensions entry point group.",
    )
    subparsers = parser.add_subparsers(dest="command")

    cache_parser = subparsers.add_parser("cache", help="Create project-local simulation cache directories.")
    cache_parser.set_defaults(command="cache")

    list_parser = subparsers.add_parser("list", help="List registered robots, environments, or tasks.")
    list_parser.add_argument("kind", choices=("robots", "envs", "tasks"), help="Registry kind to list.")

    play_parser = subparsers.add_parser("play", help="Run a registered task.")
    _add_common_run_args(play_parser)

    train_parser = subparsers.add_parser("train", help="Train a registered task when a runner exists.")
    _add_common_run_args(train_parser)

    project_parser = subparsers.add_parser("project", help="Create and manage GeneLab projects.")
    project_subparsers = project_parser.add_subparsers(dest="project_command", required=True)
    new_parser = project_subparsers.add_parser("new", help="Create an external project skeleton.")
    new_parser.add_argument("name", help="Directory and distribution name for the new project.")
    new_parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Parent directory where the project directory is created.",
    )
    new_parser.add_argument(
        "--package",
        help="Python package name to create. Defaults to a normalized form of NAME.",
    )
    new_parser.add_argument(
        "--task-id",
        help="Initial GeneLab task id. Defaults to <PackageName>-Example-v0.",
    )
    new_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite scaffold files when the target directory already exists.",
    )

    return parser


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="task [--path value]",
        help="Task id plus optional flags/overrides, such as --steps 5 or --env.robot.side left.",
    )


def main(argv: list[str] | None = None) -> None:
    argv = _normalize_run_flags(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print("Registered GeneLab tasks. Try `genelab list tasks` or import an extension package.")
        return

    if args.command == "project" and args.project_command == "new":
        _create_project_skeleton(args)
        return

    if args.command == "cache":
        ensure_project_cache()
        print(f"Using project cache at {CACHE_DIR}")
        return

    if not args.no_entry_points:
        load_entrypoint_extensions()
    for module_name in cast(list[str], args.extension_modules):
        load_extension_module(module_name)

    if args.command == "list":
        _print_registry(cast(RegistryKind, args.kind))
        return
    if args.command in {"play", "train"}:
        task, runner_args = _configured_task(args, args.command)
        try:
            if args.command == "play":
                _dispatch_play(task, runner_args)
            else:
                _dispatch_train(task, runner_args)
        except NotImplementedError as exc:
            raise SystemExit(str(exc)) from exc
        return

    print("Registered GeneLab tasks. Try `genelab list tasks` or import an extension package.")


def _normalize_run_flags(argv: list[str] | None) -> list[str] | None:
    if argv is None or len(argv) < 2:
        return argv
    command_index = _find_command_index(argv)
    if command_index is None or argv[command_index] not in {"play", "train"}:
        return argv
    command = argv[command_index]
    prefix = list(argv[:command_index])
    rest = list(argv[command_index + 1 :])
    task_index = _find_task_index(rest)
    if task_index is None:
        return argv
    task = rest.pop(task_index)
    return [*prefix, command, task, *rest]


def _find_command_index(argv: list[str]) -> int | None:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"cache", "list", "play", "train", "project"}:
            return index
        if token in {"-h", "--help", "--version", "--no-entry-points"}:
            index += 1
            continue
        if token == "--import":
            index += 2
            continue
        if token.startswith("--import="):
            index += 1
            continue
        if token.startswith("--"):
            return None
        return None
    return None


def _find_task_index(tokens: list[str]) -> int | None:
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-v", "--vis", "--gpu"}:
            index += 1
            continue
        if token.startswith("--"):
            index += 2
            continue
        return index
    return None


_RUNNER_KEYS: Final[frozenset[str]] = frozenset(
    {"num_envs", "checkpoint", "max_iterations", "seed", "log_dir"}
)


def _configured_task(
    args: argparse.Namespace, command: str
) -> tuple[_RunnableTask, dict[str, str]]:
    task_id, overrides = _parse_run_args(cast(list[str], args.args))
    try:
        task = cast(_RunnableTask, TASKS.get(task_id))
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    runner_args: dict[str, str] = {}
    for key in list(overrides.keys()):
        if key in _RUNNER_KEYS:
            runner_args[key] = overrides.pop(key)

    # When running in play mode, retarget the --vis / --gpu shortcuts at the play_env so users
    # don't have to write the full dotted path.
    if command == "play":
        play_env = getattr(task.cfg, "play_env", None)
        if play_env is not None:
            for short_key in ("env.scene.vis", "env.scene.gpu", "env.scene.steps", "env.scene.dt"):
                if short_key in overrides:
                    overrides[short_key.replace("env.", "play_env.", 1)] = overrides.pop(short_key)

    try:
        apply_overrides(task.cfg, overrides)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return task, runner_args


def _dispatch_play(task: _RunnableTask, runner_args: dict[str, str]) -> None:
    task_cfg = getattr(task, "cfg", None)
    agent_cfg = getattr(task_cfg, "agent", None) if task_cfg is not None else None
    checkpoint_raw = runner_args.get("checkpoint")
    num_envs_raw = runner_args.get("num_envs")
    if checkpoint_raw is None and num_envs_raw is None and agent_cfg is None:
        task.play()
        return
    from genelab.rl import play_task

    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL play helper")
    play_task(
        task_id,
        checkpoint=Path(checkpoint_raw) if checkpoint_raw is not None else None,
        num_envs=int(num_envs_raw) if num_envs_raw is not None else None,
    )


def _dispatch_train(task: _RunnableTask, runner_args: dict[str, str]) -> None:
    task_cfg = getattr(task, "cfg", None)
    agent_cfg = getattr(task_cfg, "agent", None) if task_cfg is not None else None
    if agent_cfg is None:
        task.train()
        return
    from genelab.rl import RslRlOnPolicyRunnerCfg, train_task

    if not isinstance(agent_cfg, RslRlOnPolicyRunnerCfg):
        raise SystemExit(
            f"task agent cfg has unsupported type {type(agent_cfg).__name__}; expected RslRlOnPolicyRunnerCfg"
        )
    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL train helper")

    num_envs_raw = runner_args.get("num_envs")
    max_iter_raw = runner_args.get("max_iterations")
    seed_raw = runner_args.get("seed")
    log_dir_raw = runner_args.get("log_dir")
    train_task(
        task_id,
        agent_cfg,
        num_envs=int(num_envs_raw) if num_envs_raw is not None else None,
        max_iterations=int(max_iter_raw) if max_iter_raw is not None else None,
        seed=int(seed_raw) if seed_raw is not None else None,
        log_root=Path(log_dir_raw) if log_dir_raw is not None else None,
    )


def _parse_run_args(tokens: list[str]) -> tuple[str, dict[str, str]]:
    task_id: str | None = None
    overrides: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-v", "--vis"}:
            overrides["env.scene.vis"] = "true"
            index += 1
            continue
        if token == "--gpu":
            overrides["env.scene.gpu"] = "true"
            index += 1
            continue
        if token == "--steps":
            overrides["env.scene.steps"] = _require_value(tokens, index)
            index += 2
            continue
        if token.startswith("--"):
            overrides[token[2:].replace("-", "_")] = _require_value(tokens, index)
            index += 2
            continue
        if task_id is not None:
            raise SystemExit(f"unexpected positional argument {token!r}")
        task_id = token
        index += 1
    if task_id is None:
        raise SystemExit("missing task id")
    return task_id, overrides


def _require_value(tokens: list[str], key_index: int) -> str:
    key = tokens[key_index]
    if key_index + 1 >= len(tokens):
        raise SystemExit(f"missing value for override {key}")
    value = tokens[key_index + 1]
    if value.startswith("--"):
        raise SystemExit(f"missing value for override {key}")
    return value


def _print_registry(kind: RegistryKind) -> None:
    registry = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind]
    print(f"Registered {kind}:")
    for entry in registry.entries():
        details = _entry_details(kind, entry.name)
        suffix = f" [{details}]" if details else ""
        print(f"- {entry.name}: {entry.description}{suffix}")


def _entry_details(kind: RegistryKind, name: str) -> str:
    try:
        value = {"robots": ROBOTS, "envs": ENVS, "tasks": TASKS}[kind].get(name)
    except Exception:
        return ""
    if kind == "tasks" and isinstance(value, _TaskLike):
        cfg = value.cfg
        return f"env={cfg.env_name}, robot={cfg.robot_name}, trainable={cfg.trainable}"
    cfg = value.cfg if isinstance(value, _CfgBacked) else None
    if cfg is not None:
        return _short_cfg(cfg)
    return ""


def _short_cfg(cfg: object) -> str:
    if not is_dataclass(cfg) or isinstance(cfg, type):
        return ""
    data = asdict(cfg)
    parts: list[str] = []
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, Path)) or value is None:
            parts.append(f"{key}={value}")
    return ", ".join(parts[:4])


def _create_project_skeleton(args: argparse.Namespace) -> None:
    name = cast(str, args.name)
    distribution_name = _normalize_distribution_name(name)
    package_name = _normalize_package_name(cast(str | None, args.package) or name)
    task_id = cast(str | None, args.task_id) or _default_task_id(package_name)
    parent = cast(Path, args.path).expanduser()
    target = (parent / name).resolve()
    force = cast(bool, args.force)

    if target.exists() and not target.is_dir():
        raise SystemExit(f"target exists and is not a directory: {target}")
    if target.exists() and any(target.iterdir()) and not force:
        raise SystemExit(f"target directory is not empty: {target}; use --force to overwrite scaffold files")

    package_dir = Path("src") / package_name
    genelab_source = _relative_path(_default_genelab_source_root(), target)
    robot_name = f"{distribution_name}-robot"
    env_name = f"{distribution_name}-env"
    files = {
        "README.md": _template_readme(name, task_id),
        "pyproject.toml": _template_pyproject(distribution_name, package_name, genelab_source),
        str(package_dir / "__init__.py"): '"""GeneLab extension project."""\n',
        str(package_dir / "config.py"): _template_config(),
        str(package_dir / "robots.py"): _template_robots(package_name),
        str(package_dir / "envs.py"): _template_envs(package_name),
        str(package_dir / "tasks.py"): _template_tasks(
            package_name,
            task_id=task_id,
            robot_name=robot_name,
            env_name=env_name,
        ),
    }

    target.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        path = target / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_dir():
            raise SystemExit(f"cannot overwrite directory with scaffold file: {path}")
        if path.exists() and not force:
            raise SystemExit(f"scaffold file already exists: {path}; use --force to overwrite")
        path.write_text(content, encoding="utf-8")

    print(f"Created GeneLab extension project at {target}")
    print("Next steps:")
    print(f"  cd {target}")
    print("  uv sync")
    print("  uv run genelab list tasks")
    print(f"  uv run genelab play {task_id}")


def _normalize_distribution_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "-", raw.strip().lower()).strip("-")
    if not name:
        raise SystemExit("project name must contain at least one letter or digit")
    return name


def _normalize_package_name(raw: str) -> str:
    name = re.sub(r"\W+", "_", raw.strip().lower()).strip("_")
    if not name:
        raise SystemExit("package name must contain at least one letter or digit")
    if name[0].isdigit():
        name = f"project_{name}"
    if not name.isidentifier():
        raise SystemExit(f"package name must be a valid Python identifier, got {raw!r}")
    return name


def _default_task_id(package_name: str) -> str:
    title = "".join(part.capitalize() for part in package_name.split("_") if part)
    return f"{title}-Example-v0"


def _default_genelab_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative_path(source: Path, target: Path) -> str:
    return os.path.relpath(source.resolve(), target.resolve()).replace(os.sep, "/")


def _template_readme(name: str, task_id: str) -> str:
    return dedent(
        f"""\
        # {name}

        This is a GeneLab external project skeleton.

        ```bash
        uv sync
        uv run genelab list tasks
        uv run genelab play {task_id}
        ```

        GeneLab loads this project through the `genelab.extensions` entry point in `pyproject.toml`.
        """
    )


def _template_pyproject(distribution_name: str, package_name: str, genelab_source: str) -> str:
    return dedent(
        f"""\
        [project]
        name = "{distribution_name}"
        version = "0.1.0"
        description = "GeneLab extension project."
        requires-python = ">=3.12"
        dependencies = ["genelab"]

        [project.entry-points."genelab.extensions"]
        {package_name} = "{package_name}.tasks:register"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{package_name}"]

        [tool.uv.sources]
        genelab = {{ path = "{genelab_source}", editable = true }}
        """
    )


def _template_config() -> str:
    return dedent(
        """\
        \"\"\"Configuration owned by this GeneLab extension project.\"\"\"

        from dataclasses import dataclass, field

        from genelab.configs import ManagerBasedEnvCfg, SceneCfg


        @dataclass
        class RobotCfg:
            asset_path: str = "assets/robot.usd"


        @dataclass
        class EnvCfg(ManagerBasedEnvCfg):
            scene: SceneCfg = field(default_factory=lambda: SceneCfg(steps=128))
            robot: RobotCfg = field(default_factory=RobotCfg)
        """
    )


def _template_robots(package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Robot definitions owned by this GeneLab extension project.\"\"\"

        from dataclasses import dataclass

        from {package_name}.config import RobotCfg


        @dataclass
        class Robot:
            cfg: RobotCfg


        def create_robot(cfg: RobotCfg | None = None) -> Robot:
            return Robot(cfg or RobotCfg())
        """
    )


def _template_envs(package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Environment runners owned by this GeneLab extension project.\"\"\"

        from {package_name}.config import EnvCfg


        class ExampleEnv:
            def __init__(self, cfg: EnvCfg | None = None) -> None:
                self.cfg = cfg or EnvCfg()

            def play(self) -> None:
                print(f"Run {{type(self).__name__}} for {{self.cfg.scene.steps}} steps")
        """
    )


def _template_tasks(
    package_name: str,
    *,
    task_id: str,
    robot_name: str,
    env_name: str,
) -> str:
    return dedent(
        f"""\
        \"\"\"GeneLab registration hook for this extension project.\"\"\"

        from genelab.configs import TaskCfg
        from genelab.registry import ENVS, ROBOTS, TASKS, register_env, register_robot, register_task

        from {package_name}.config import EnvCfg, RobotCfg
        from {package_name}.envs import ExampleEnv
        from {package_name}.robots import create_robot


        TASK_ID = "{task_id}"
        ROBOT_NAME = "{robot_name}"
        ENV_NAME = "{env_name}"


        class ExampleTask:
            def __init__(self) -> None:
                self.cfg = TaskCfg(
                    name=TASK_ID,
                    env_name=ENV_NAME,
                    robot_name=ROBOT_NAME,
                    env=EnvCfg(),
                    trainable=False,
                )

            def play(self) -> None:
                if not isinstance(self.cfg.env, EnvCfg):
                    raise TypeError("ExampleTask requires EnvCfg")
                ExampleEnv(self.cfg.env).play()

            def train(self) -> None:
                raise NotImplementedError(f"add your training runner for {{self.cfg.name}}")


        def register() -> None:
            if ROBOT_NAME not in ROBOTS:
                register_robot(
                    ROBOT_NAME,
                    create_robot,
                    description="Robot provided by this GeneLab extension project.",
                    cfg_type=RobotCfg,
                )
            if ENV_NAME not in ENVS:
                register_env(
                    ENV_NAME,
                    ExampleEnv,
                    description="Environment provided by this GeneLab extension project.",
                    cfg_type=EnvCfg,
                )
            if TASK_ID not in TASKS:
                register_task(
                    TASK_ID,
                    ExampleTask,
                    description="Task provided by this GeneLab extension project.",
                    cfg_type=TaskCfg,
                )
        """
    )


if __name__ == "__main__":
    main()
