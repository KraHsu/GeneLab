"""Unified Typer + Rich command-line entry point for registered GeneLab tasks."""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Final, Protocol, cast

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
    render_entry_info,
    render_main_help,
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


_RUN_FLAGS_HELP: Final[str] = """\
Shorthand flags rewritten into env overrides:

\b
  -v, --vis        Enable the Genesis viewer (env.scene.vis=true).
  --gpu            Use the GPU backend (env.scene.gpu=true).
  --steps N        Run for N steps (env.scene.steps=N).
  --dt SECONDS     Override the sim timestep (env.scene.dt=SECONDS).
  --a.b.c VALUE    Set any dotted cfg path.

Runner flags (used when an RL runner is engaged):

\b
  --num_envs N         Parallel environments.
  --agent KIND         one of: zero, random, trained (play only).
  --checkpoint PATH    Resume from a checkpoint.
  --seed N             RNG seed.
  --log_dir PATH       Override the log directory.
  --max_iterations N   Cap training iterations (train only).
  --gpus N             Distributed training across N GPUs (train only).

Use `genelab info TASK` to see the full overridable path list for a task.
"""


_PLAY_HELP: Final[str] = "Run a registered task.\n\n" + _RUN_FLAGS_HELP
_TRAIN_HELP: Final[str] = "Train a registered task when a runner exists.\n\n" + _RUN_FLAGS_HELP


app = typer.Typer(
    name="genelab",
    help=(
        "GeneLab — Genesis robot lab CLI.\n\n"
        "Run `genelab` with no arguments for a landing page with quickstart commands "
        "and a count of registered robots, envs, and tasks."
    ),
    no_args_is_help=False,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)

project_app = typer.Typer(
    name="project",
    help="Create and manage GeneLab extension projects.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
app.add_typer(project_app, name="project", rich_help_panel="Project")


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
            help="Import an extension module before dispatch (repeatable).",
        ),
    ] = None,
    no_entry_points: Annotated[
        bool,
        typer.Option(
            "--no-entry-points",
            help="Skip installed entry points from the genelab.extensions group.",
        ),
    ] = False,
) -> None:
    _ = version  # consumed by the eager callback
    state = _RootState(
        extension_modules=list(extension_modules or []),
        no_entry_points=no_entry_points,
    )
    ctx.obj = state
    if ctx.invoked_subcommand is None:
        _load_extensions(state)
        render_main_help()


@app.command(
    "cache",
    help="Create project-local simulation cache directories.",
    rich_help_panel="Utilities",
)
def cache_cmd() -> None:
    ensure_project_cache()
    render_cache(CACHE_DIR)


@app.command(
    "list",
    help="List registered robots, envs, or tasks.",
    rich_help_panel="Registry",
)
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
    "info",
    help="Show detail for one registered task, env, or robot.",
    rich_help_panel="Registry",
)
def info_cmd(
    ctx: typer.Context,
    name: Annotated[
        str,
        typer.Argument(
            metavar="NAME",
            help="Registered task, env, or robot name.",
        ),
    ],
) -> None:
    _load_extensions(_state(ctx))
    render_entry_info(name)


@app.command(
    "play",
    help=_PLAY_HELP,
    rich_help_panel="Runtime",
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
    help=_TRAIN_HELP,
    rich_help_panel="Runtime",
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
    name: Annotated[
        str, typer.Argument(help="Directory and distribution name for the new project.")
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Parent directory under which the project directory is created.",
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


def _configured_task(tokens: list[str], *, command: str) -> tuple[_RunnableTask, dict[str, str]]:
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
        raise SystemExit(f"--agent must be one of {{zero, random, trained}}; got {agent_raw!r}")
    if checkpoint_raw is None and num_envs_raw is None and agent_raw is None and agent_cfg is None:
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

    gpus_raw = runner_args.pop("gpus", None)
    gpus = int(gpus_raw) if gpus_raw is not None else 1
    if gpus > 1 and "TORCHELASTIC_RUN_ID" not in os.environ:
        _relaunch_under_torchrun(gpus, agent_cfg, runner_args)
        return

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
        log_dir=Path(log_dir_raw) if log_dir_raw is not None else None,
    )


def _relaunch_under_torchrun(
    gpus: int,
    agent_cfg: Any,
    runner_args: dict[str, str],
) -> None:
    """Re-exec the current ``train`` invocation under torchrun.

    The parent precomputes the log directory and forwards it via ``--log-dir`` so all
    ranks land in the same directory (avoiding per-rank timestamp drift). The original
    argv is forwarded verbatim minus the ``--gpus N`` tokens so env overrides such as
    ``--env.scene.steps 5`` survive the relaunch.
    """
    from genelab.rl.runner import resolve_log_dir

    log_root_raw = runner_args.get("log_dir")
    log_root = Path(log_root_raw) if log_root_raw else Path("logs") / "rsl_rl"
    log_dir = resolve_log_dir(log_root, agent_cfg.experiment_name, agent_cfg.run_name)

    inner = _strip_gpus_flag(sys.argv[1:])
    if not _has_log_dir_flag(inner):
        inner += ["--log-dir", str(log_dir)]

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus}",
        "-m",
        "genelab.cli",
        *inner,
    ]
    os.execvp(cmd[0], cmd)


def _strip_gpus_flag(tokens: list[str]) -> list[str]:
    """Drop ``--gpus N`` (two-token form) and ``--gpus=N`` (single-token) from tokens."""
    out: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok == "--gpus":
            skip_next = True
            continue
        if tok.startswith("--gpus="):
            continue
        out.append(tok)
    return out


def _has_log_dir_flag(tokens: list[str]) -> bool:
    for t in tokens:
        if t in {"--log-dir", "--log_dir"}:
            return True
        if t.startswith("--log-dir=") or t.startswith("--log_dir="):
            return True
    return False


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
