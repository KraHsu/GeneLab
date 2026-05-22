"""Unified Typer + Rich command-line entry point for registered GeneLab tasks."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from genelab import __version__
from genelab.cache import CACHE_DIR, ensure_project_cache
from genelab.cli._argv import (
    PROF_KEYS,
    RUNNER_KEYS,
    normalize_argv,
    parse_run_args,
    split_prof_keys,
    split_runner_keys,
)
from genelab.cli._completion import complete_any_registry_name, complete_task_names
from genelab.cli._dispatch import (
    _dispatch_play as _dispatch_play,
    _dispatch_train as _dispatch_train,
)
from genelab.cli._distributed import (
    _extract_log_dir_flag as _extract_log_dir_flag,
    _has_log_dir_flag as _has_log_dir_flag,
    _relaunch_under_torchrun as _relaunch_under_torchrun,
    _resolve_per_rank_num_envs as _resolve_per_rank_num_envs,
    _strip_distributed_flags as _strip_distributed_flags,
    _strip_flag_value_pairs as _strip_flag_value_pairs,
)
from genelab.cli._multi_seed import (
    _dispatch_multi_seed_train as _dispatch_multi_seed_train,
    _parse_seed_list as _parse_seed_list,
    _resolve_multi_seed_parent as _resolve_multi_seed_parent,
    _strip_multi_seed_flags as _strip_multi_seed_flags,
)
from genelab.cli._prof import prof_app
from genelab.cli._progress import fetch_progress
from genelab.cli._render import (
    render_cache,
    render_entry_info,
    render_main_help,
    render_registry,
)
from genelab.cli._help import _PLAY_HELP, _TRAIN_HELP
from genelab.cli._resolve import _configured_task
from genelab.cli._scaffold import create_project_skeleton
from genelab.registry import (
    load_bundled_asset_zoo,
    load_entrypoint_extensions,
    load_extension_module,
)

__all__ = [
    "PROF_KEYS",
    "RUNNER_KEYS",
    "app",
    "main",
    "normalize_argv",
    "parse_run_args",
    "split_prof_keys",
    "split_runner_keys",
]


@dataclass
class _RootState:
    extension_modules: list[str] = field(default_factory=list)
    no_entry_points: bool = False


class _RegistryKindArg(str, Enum):
    robots = "robots"
    envs = "envs"
    tasks = "tasks"


app = typer.Typer(
    name="genelab",
    help=(
        "GeneLab — Genesis robot lab CLI.\n\n"
        "Run `genelab` with no arguments for a landing page with quickstart commands "
        "and a count of registered robots, envs, and tasks."
    ),
    no_args_is_help=False,
    add_completion=True,
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
app.add_typer(prof_app, name="prof", rich_help_panel="Utilities")


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
            autocompletion=complete_any_registry_name,
        ),
    ],
) -> None:
    _load_extensions(_state(ctx))
    with fetch_progress():
        render_entry_info(name)


_TASK_ARG_HELP: Final[str] = "Registered task id (omit for an interactive picker)."


@app.command(
    "play",
    help=_PLAY_HELP,
    rich_help_panel="Runtime",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def play_cmd(
    ctx: typer.Context,
    task_id: Annotated[
        str | None,
        typer.Argument(
            metavar="TASK",
            help=_TASK_ARG_HELP,
            autocompletion=complete_task_names,
        ),
    ] = None,
) -> None:
    _load_extensions(_state(ctx))
    tokens = list(ctx.args)
    if task_id is not None:
        tokens = [task_id, *tokens]
    task, runner_args, prof_args = _configured_task(tokens, command="play")
    try:
        _dispatch_play(task, runner_args, prof_args)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc


@app.command(
    "eval",
    help=(
        "Run a deterministic rollout of TASK against CHECKPOINT and write eval.json.\n\n"
        "Reads ``extras['is_success']`` per-env when the task publishes it; otherwise "
        "``success_rate`` in the output is ``null``."
    ),
    rich_help_panel="Runtime",
)
def eval_cmd(
    ctx: typer.Context,
    task_id: Annotated[
        str,
        typer.Argument(
            metavar="TASK",
            help="Registered task id.",
            autocompletion=complete_task_names,
        ),
    ],
    checkpoint: Annotated[
        Path,
        typer.Argument(
            metavar="CHECKPOINT",
            help="Trained checkpoint path (rsl_rl/skrl: .pt; sb3: .zip).",
        ),
    ],
    num_envs: Annotated[
        int,
        typer.Option("--num-envs", "--num_envs", help="Parallel envs for the rollout."),
    ] = 64,
    episodes: Annotated[
        int,
        typer.Option("--episodes", help="Minimum complete episodes to collect."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option("--seed", help="RNG seed (env-level only; policy is deterministic)."),
    ] = 0,
    deterministic: Annotated[
        bool,
        typer.Option(
            "--deterministic/--stochastic",
            help="Use deterministic policy (default) or sample from the distribution.",
        ),
    ] = True,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output JSON path."),
    ] = Path("eval.json"),
    max_steps: Annotated[
        int | None,
        typer.Option("--max-steps", help="Safety cap on rollout steps."),
    ] = None,
) -> None:
    _load_extensions(_state(ctx))
    from genelab.rl.eval_task import eval_task

    _result, payload = eval_task(
        task_id,
        checkpoint,
        num_envs=num_envs,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
        max_steps=max_steps,
        out_path=out,
    )
    typer.echo(json.dumps(payload, indent=2))


_EXPORT_FORMATS: Final[tuple[str, ...]] = ("torchscript", "onnx")


@app.command(
    "export",
    help=(
        "Export TASK's policy at CHECKPOINT to TorchScript or ONNX.\n\n"
        "The exported model takes raw obs in and emits actions; per-term scale/clip "
        "are baked in (deployment side feeds raw, training-shape obs). A sibling "
        "<OUTPUT>.metadata.json records the obs schema."
    ),
    rich_help_panel="Runtime",
)
def export_cmd(
    ctx: typer.Context,
    task_id: Annotated[
        str,
        typer.Argument(
            metavar="TASK",
            help="Registered task id.",
            autocompletion=complete_task_names,
        ),
    ],
    checkpoint: Annotated[
        Path,
        typer.Argument(
            metavar="CHECKPOINT",
            help="Trained checkpoint path (rsl_rl/skrl: .pt; sb3: .zip).",
        ),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help=f"Output format. One of {_EXPORT_FORMATS}.",
        ),
    ] = "torchscript",
    out: Annotated[
        Path,
        typer.Option("--out", help="Output file path."),
    ] = Path("policy.ts"),
    opset: Annotated[
        int,
        typer.Option("--opset", help="ONNX opset version (ignored for torchscript)."),
    ] = 17,
) -> None:
    if format not in _EXPORT_FORMATS:
        raise SystemExit(f"--format must be one of {_EXPORT_FORMATS}; got {format!r}")
    _load_extensions(_state(ctx))
    from genelab.cli._export import export_task

    fmt: Literal["torchscript", "onnx"] = "torchscript" if format == "torchscript" else "onnx"
    written = export_task(
        task_id,
        checkpoint,
        format=fmt,
        output=out,
        opset=opset,
    )
    typer.echo(f"wrote {written}")
    typer.echo(f"wrote {written.with_suffix(written.suffix + '.metadata.json')}")


@app.command(
    "train",
    help=_TRAIN_HELP,
    rich_help_panel="Runtime",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_cmd(
    ctx: typer.Context,
    task_id: Annotated[
        str | None,
        typer.Argument(
            metavar="TASK",
            help=_TASK_ARG_HELP,
            autocompletion=complete_task_names,
        ),
    ] = None,
) -> None:
    _load_extensions(_state(ctx))
    tokens = list(ctx.args)
    if task_id is not None:
        tokens = [task_id, *tokens]
    task, runner_args, prof_args = _configured_task(tokens, command="train")
    if "seeds" in runner_args:
        try:
            _dispatch_multi_seed_train(task, tokens, runner_args)
        except NotImplementedError as exc:
            raise SystemExit(str(exc)) from exc
        return
    try:
        _dispatch_train(task, runner_args, prof_args)
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
    # Bundled example robots (g1, go1, anymal-c, franka, cartpole) live in
    # genelab.asset_zoo. They are an opinionated bundle of examples, not core
    # API — load explicitly so the registration is intentional rather than a
    # side-effect of any other import.
    load_bundled_asset_zoo()
    if not state.no_entry_points:
        load_entrypoint_extensions()
    for module_name in state.extension_modules:
        load_extension_module(module_name)


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
