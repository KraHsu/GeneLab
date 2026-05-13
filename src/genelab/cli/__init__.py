"""Unified Typer + Rich command-line entry point for registered GeneLab tasks."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Protocol, cast

import typer

from genelab import __version__
from genelab.cache import CACHE_DIR, ensure_project_cache
from genelab.cli._argv import (
    RUNNER_KEYS,
    normalize_argv,
    parse_run_args,
    split_runner_keys,
)
from genelab.cli._interactive import pick_task_interactively
from genelab.cli._render import (
    render_cache,
    render_hint,
    render_registry,
)
from genelab.cli._scaffold import create_project_skeleton
from genelab.configs import apply_overrides
from genelab.registry import TASKS, load_entrypoint_extensions, load_extension_module

__all__ = [
    "RUNNER_KEYS",
    "app",
    "main",
    "normalize_argv",
    "parse_run_args",
    "split_runner_keys",
]


class _RunnableTask(Protocol):
    cfg: object

    def play(self) -> None: ...

    def train(self) -> None: ...


@dataclass
class _RootState:
    extension_modules: list[str] = field(default_factory=list)
    no_entry_points: bool = False


class _RegistryKindArg(str, Enum):
    robots = "robots"
    envs = "envs"
    tasks = "tasks"


_AGENT_KINDS: Final[frozenset[str]] = frozenset({"zero", "random", "trained"})

_PLAY_RETARGETED_KEYS: Final[tuple[str, ...]] = (
    "env.scene.vis",
    "env.scene.gpu",
    "env.scene.steps",
    "env.scene.dt",
)


app = typer.Typer(
    name="genelab",
    help="Run registered GeneLab tasks.",
    no_args_is_help=False,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)

project_app = typer.Typer(
    name="project",
    help="Create and manage GeneLab projects.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
app.add_typer(project_app, name="project")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"genelab {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    extension_modules: Annotated[
        list[str] | None,
        typer.Option(
            "--import",
            metavar="MODULE",
            help="Import a downstream extension module before dispatch (can be repeated).",
        ),
    ] = None,
    no_entry_points: Annotated[
        bool,
        typer.Option(
            "--no-entry-points",
            help="Skip installed extensions from the genelab.extensions entry point group.",
        ),
    ] = False,
) -> None:
    _ = version  # consumed by the eager callback
    ctx.obj = _RootState(
        extension_modules=list(extension_modules or []),
        no_entry_points=no_entry_points,
    )
    if ctx.invoked_subcommand is None:
        render_hint()


@app.command("cache", help="Create project-local simulation cache directories.")
def cache_cmd() -> None:
    ensure_project_cache()
    render_cache(CACHE_DIR)


@app.command("list", help="List registered robots, environments, or tasks.")
def list_cmd(
    ctx: typer.Context,
    kind: Annotated[
        _RegistryKindArg,
        typer.Argument(help="Registry kind to list."),
    ],
) -> None:
    _load_extensions(_state(ctx))
    render_registry(kind.value)


@app.command(
    "play",
    help="Run a registered task.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def play_cmd(ctx: typer.Context) -> None:
    _load_extensions(_state(ctx))
    task, runner_args = _configured_task(list(ctx.args), command="play")
    try:
        _dispatch_play(task, runner_args)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc


@app.command(
    "train",
    help="Train a registered task when a runner exists.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_cmd(ctx: typer.Context) -> None:
    _load_extensions(_state(ctx))
    task, runner_args = _configured_task(list(ctx.args), command="train")
    try:
        _dispatch_train(task, runner_args)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc


@project_app.command("new", help="Create an external project skeleton.")
def project_new_cmd(
    name: Annotated[str, typer.Argument(help="Directory and distribution name for the new project.")],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Parent directory where the project directory is created.",
        ),
    ] = Path("."),
    package: Annotated[
        str | None,
        typer.Option(
            "--package",
            help="Python package name to create. Defaults to a normalized form of NAME.",
        ),
    ] = None,
    task_id: Annotated[
        str | None,
        typer.Option(
            "--task-id",
            help="Initial GeneLab task id. Defaults to <PackageName>-Example-v0.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite scaffold files when the target directory already exists.",
        ),
    ] = False,
) -> None:
    create_project_skeleton(
        name,
        path=path,
        package=package,
        task_id=task_id,
        force=force,
    )


def _state(ctx: typer.Context) -> _RootState:
    obj = ctx.obj
    if isinstance(obj, _RootState):
        return obj
    return _RootState()


def _load_extensions(state: _RootState) -> None:
    if not state.no_entry_points:
        load_entrypoint_extensions()
    for module_name in state.extension_modules:
        load_extension_module(module_name)


def _configured_task(
    tokens: list[str], *, command: str
) -> tuple[_RunnableTask, dict[str, str]]:
    try:
        task_id, overrides = parse_run_args(tokens)
    except SystemExit as exc:
        if str(exc) != "missing task id":
            raise
        picked = pick_task_interactively()
        if picked is None:
            raise
        task_id, overrides = parse_run_args([*tokens, picked])

    try:
        task = cast(_RunnableTask, TASKS.get(task_id))
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    runner_args = split_runner_keys(overrides)

    # In play mode, retarget the short --vis / --gpu / --steps / --dt shortcuts at the
    # task's play_env when one is configured. Keeps `genelab play TASK --vis` working
    # without forcing users to spell `play_env.scene.vis`.
    if command == "play" and getattr(task.cfg, "play_env", None) is not None:
        for short_key in _PLAY_RETARGETED_KEYS:
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
    agent_raw = runner_args.get("agent")
    if agent_raw is not None and agent_raw not in _AGENT_KINDS:
        raise SystemExit(
            f"--agent must be one of {{zero, random, trained}}; got {agent_raw!r}"
        )
    if (
        checkpoint_raw is None
        and num_envs_raw is None
        and agent_raw is None
        and agent_cfg is None
    ):
        task.play()
        return
    from genelab.rl import AgentKind, play_task

    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL play helper")
    play_task(
        task_id,
        checkpoint=Path(checkpoint_raw) if checkpoint_raw is not None else None,
        num_envs=int(num_envs_raw) if num_envs_raw is not None else None,
        agent=cast("AgentKind | None", agent_raw),
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
            f"task agent cfg has unsupported type {type(agent_cfg).__name__}; "
            "expected RslRlOnPolicyRunnerCfg"
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


def main(argv: list[str] | None = None) -> None:
    """Programmatic entry point used by tests and the console script.

    Wraps the Typer app so that:
    * an explicit ``argv`` list is forwarded as-is (instead of reading ``sys.argv``);
    * successful runs return ``None`` (Click would otherwise ``sys.exit(0)``);
    * ``SystemExit('message')`` raised by command bodies propagates with its message intact.
    """

    prepared = normalize_argv(argv) if argv is not None else None
    try:
        app(args=prepared, prog_name="genelab", standalone_mode=True)
    except SystemExit as exc:
        code = exc.code
        if code is None or (isinstance(code, int) and code == 0):
            return
        raise


if __name__ == "__main__":
    main()
