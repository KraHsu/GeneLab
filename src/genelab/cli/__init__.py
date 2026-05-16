"""Unified Typer + Rich command-line entry point for registered GeneLab tasks."""

import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Final, Protocol, cast

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
from genelab.cli._interactive import (
    pick_agent_kind,
    pick_name_interactively,
    pick_override_path,
    pick_task_interactively,
)
from genelab.cli._prof import prof_app
from genelab.cli._progress import fetch_progress
from genelab.cli._render import (
    iter_overridable_paths,
    render_cache,
    render_entry_info,
    render_main_help,
    render_registry,
)
from genelab.cli._scaffold import create_project_skeleton
from genelab.configs import apply_overrides
from genelab.registry import TASKS, load_entrypoint_extensions, load_extension_module

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
    "env.simulation.vis",
    "env.simulation.gpu",
    "env.simulation.steps",
    "env.simulation.dt",
)


_RUN_FLAGS_HELP: Final[str] = """\
Shorthand flags rewritten into env overrides:

\b
  -v, --vis        Enable the Genesis viewer (env.simulation.vis=true).
  --gpu            Use the GPU backend (env.simulation.gpu=true).
  --steps N        Run for N steps. Play: env.simulation.steps=N.
                   Train: alias for --max_iterations N.
  --dt SECONDS     Override the sim timestep (env.simulation.dt=SECONDS).
  --a.b.c VALUE    Set any dotted cfg path.

Runner flags (used when an RL runner is engaged):

\b
  --num_envs N          Total parallel environments across all ranks.
                        Must divide evenly by --gpus when both are set.
  --num_envs_per_gpu N  Per-rank parallel environments. Mutually exclusive
                        with --num_envs.
  --agent KIND          one of: zero, random, trained (play only).
  --checkpoint PATH     Resume from a checkpoint.
  --seed N              RNG seed.
  --log_dir PATH        Override the log directory.
  --max_iterations N    Cap training iterations (train only).
  --gpus N              Distributed training across N GPUs (train only).

Profiling flags (forwarded to torch.profiler; rank-0 only):

\b
  --prof                  Enable the profiler (overrides GENELAB_PROFILE).
  --prof-out PATH         TensorBoard trace directory (GENELAB_PROFILE_OUT).
  --prof-wait N           Schedule wait steps (GENELAB_PROFILE_WAIT, default 10).
  --prof-warmup N         Schedule warmup steps (GENELAB_PROFILE_WARMUP, default 5).
  --prof-active N         Schedule active steps (GENELAB_PROFILE_ACTIVE, default 10).
  --prof-repeat N         Schedule cycles (GENELAB_PROFILE_REPEAT, default 2).
  --prof-record-shapes    Record tensor shapes for op input attribution.
  --prof-with-stack       Capture Python stack traces (high overhead).

Use `genelab info TASK` to see the full overridable path list for a task.
Use `genelab prof open [DIR]` to launch TensorBoard against a trace directory.
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
    if not state.no_entry_points:
        load_entrypoint_extensions()
    for module_name in state.extension_modules:
        load_extension_module(module_name)


def _configured_task(
    tokens: list[str], *, command: str
) -> tuple[_RunnableTask, dict[str, str], dict[str, str]]:
    try:
        task_id, overrides = parse_run_args(tokens)
    except SystemExit as exc:
        if str(exc) != "missing task id":
            raise
        picked = pick_task_interactively()
        if picked is None:
            raise
        task_id, overrides = parse_run_args([*tokens, picked])

    task = _resolve_task(task_id)

    # In train mode, ``--steps N`` is the short form for ``--max_iterations N``.
    # ``env.simulation.steps`` is not consumed by ``train_task`` / ``ManagerBasedRlEnv``
    # (episodes are governed by ``episode_length_s``), so leaving the override on the env
    # cfg would silently no-op and the user would see iterations counted to the cfg
    # default (e.g. 0/30000) instead of stopping at N.
    if command == "train" and "env.simulation.steps" in overrides:
        if "max_iterations" in overrides:
            raise SystemExit(
                "--steps and --max_iterations conflict in train mode: drop one. "
                "(--steps is the short form for --max_iterations.)"
            )
        overrides["max_iterations"] = overrides.pop("env.simulation.steps")

    runner_args = split_runner_keys(overrides)
    prof_args = split_prof_keys(overrides)

    # In play mode, retarget the short --vis / --gpu / --steps / --dt shortcuts at the
    # task's play_env when one is configured. Keeps `genelab play TASK --vis` working
    # without forcing users to spell `play_env.simulation.vis`.
    if command == "play" and getattr(task.cfg, "play_env", None) is not None:
        for short_key in _PLAY_RETARGETED_KEYS:
            if short_key in overrides:
                overrides[short_key.replace("env.", "play_env.", 1)] = overrides.pop(short_key)

    _apply_overrides_interactively(task.cfg, overrides)
    return task, runner_args, prof_args


def _resolve_task(task_id: str) -> _RunnableTask:
    try:
        with fetch_progress():
            return cast(_RunnableTask, TASKS.get(task_id))
    except KeyError as exc:
        picked = pick_name_interactively(TASKS.names(), f"Unknown task {task_id!r}. Pick one:")
        if picked is None or picked == task_id:
            raise SystemExit(str(exc)) from exc
        with fetch_progress():
            return cast(_RunnableTask, TASKS.get(picked))


_UNKNOWN_PATH_RE: Final[re.Pattern[str]] = re.compile(r"unknown override path: '([^']+)'")


def _apply_overrides_interactively(cfg: object, overrides: dict[str, str]) -> None:
    """Apply overrides; on an unknown path, prompt the user for a correction.

    Coercion errors (e.g. ``int('abc')``) still exit immediately — those need a
    new value, not a new key.
    """
    while True:
        try:
            apply_overrides(cfg, overrides)
            return
        except ValueError as exc:
            msg = str(exc)
            match = _UNKNOWN_PATH_RE.search(msg)
            if match is None:
                raise SystemExit(msg) from exc
            bad_path = match.group(1)
            override_key = _override_key_for(bad_path, overrides)
            if override_key is None:
                raise SystemExit(msg) from exc
            candidates = [path for path, _, _ in iter_overridable_paths(cfg)]
            picked = pick_override_path(bad_path, candidates)
            if picked is None or picked == bad_path:
                raise SystemExit(msg) from exc
            overrides[picked] = overrides.pop(override_key)


def _override_key_for(bad_path: str, overrides: dict[str, str]) -> str | None:
    """Return the ``overrides`` key whose ``apply_overrides`` target equals ``bad_path``."""
    from genelab.configs import resolve_override_alias

    if bad_path in overrides:
        return bad_path
    for key in overrides:
        if resolve_override_alias(key) == bad_path:
            return key
    return None


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None) -> int | None:
    return int(raw) if raw is not None else None


def _parse_path(raw: str | None) -> Path | None:
    return Path(raw) if raw is not None else None


def _coerce_prof_kwargs(prof_args: dict[str, str]) -> dict[str, Any]:
    """Translate the raw string dict produced by ``split_prof_keys`` into typed kwargs."""
    return {
        "prof": _parse_bool(prof_args.get("prof")),
        "prof_out": _parse_path(prof_args.get("prof_out")),
        "prof_wait": _parse_int(prof_args.get("prof_wait")),
        "prof_warmup": _parse_int(prof_args.get("prof_warmup")),
        "prof_active": _parse_int(prof_args.get("prof_active")),
        "prof_repeat": _parse_int(prof_args.get("prof_repeat")),
        "prof_record_shapes": _parse_bool(prof_args.get("prof_record_shapes")),
        "prof_with_stack": _parse_bool(prof_args.get("prof_with_stack")),
    }


def _dispatch_play(
    task: _RunnableTask, runner_args: dict[str, str], prof_args: dict[str, str]
) -> None:
    task_cfg = getattr(task, "cfg", None)
    agent_cfg = getattr(task_cfg, "agent", None) if task_cfg is not None else None
    checkpoint_raw = runner_args.get("checkpoint")
    agent_raw = runner_args.get("agent")
    if agent_raw is not None and agent_raw not in _AGENT_KINDS:
        picked_agent = pick_agent_kind()
        if picked_agent is None or picked_agent not in _AGENT_KINDS:
            raise SystemExit(f"--agent must be one of {{zero, random, trained}}; got {agent_raw!r}")
        agent_raw = picked_agent
        runner_args["agent"] = picked_agent
    # play is always single-process; either flag is accepted but mapped through the same
    # resolver so the mutual-exclusion guard fires on misuse.
    num_envs_per_rank = _resolve_per_rank_num_envs(runner_args, gpus=1)
    if (
        checkpoint_raw is None
        and num_envs_per_rank is None
        and agent_raw is None
        and agent_cfg is None
        and not prof_args
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
        num_envs=num_envs_per_rank,
        agent=cast("AgentKind | None", agent_raw),
        **_coerce_prof_kwargs(prof_args),
    )


def _dispatch_train(
    task: _RunnableTask, runner_args: dict[str, str], prof_args: dict[str, str]
) -> None:
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
    num_envs_per_rank = _resolve_per_rank_num_envs(runner_args, gpus=gpus)
    if gpus > 1 and "TORCHELASTIC_RUN_ID" not in os.environ:
        _relaunch_under_torchrun(gpus, agent_cfg, runner_args, num_envs_per_rank, task_id=task_id)
        return

    max_iter_raw = runner_args.get("max_iterations")
    seed_raw = runner_args.get("seed")
    log_dir_raw = runner_args.get("log_dir")
    train_task(
        task_id,
        agent_cfg,
        num_envs=num_envs_per_rank,
        max_iterations=int(max_iter_raw) if max_iter_raw is not None else None,
        seed=int(seed_raw) if seed_raw is not None else None,
        log_dir=Path(log_dir_raw) if log_dir_raw is not None else None,
        **_coerce_prof_kwargs(prof_args),
    )


def _resolve_per_rank_num_envs(runner_args: dict[str, str], *, gpus: int) -> int | None:
    """Pop ``num_envs`` / ``num_envs_per_gpu`` from ``runner_args`` and return per-rank N.

    ``--num-envs N`` is interpreted as the **total** across all ranks: ``N // gpus``
    becomes the per-rank count and ``N`` must be divisible by ``gpus``.
    ``--num-envs-per-gpu M`` is verbatim per-rank. Passing both is a hard error so users
    don't quietly get one or the other's semantics. Returns ``None`` when neither flag
    was set so callers can defer to the cfg default.
    """
    total_raw = runner_args.pop("num_envs", None)
    per_gpu_raw = runner_args.pop("num_envs_per_gpu", None)
    if total_raw is not None and per_gpu_raw is not None:
        raise SystemExit(
            "--num-envs and --num-envs-per-gpu are mutually exclusive; pass exactly "
            "one (or neither, to use the task's cfg default)."
        )
    if per_gpu_raw is not None:
        return int(per_gpu_raw)
    if total_raw is None:
        return None
    total = int(total_raw)
    if gpus <= 0:
        raise SystemExit(f"--gpus must be a positive integer (got {gpus})")
    if total % gpus != 0:
        raise SystemExit(
            f"--num-envs {total} is not divisible by --gpus {gpus}; pick a multiple "
            f"of {gpus} or use --num-envs-per-gpu instead."
        )
    return total // gpus


def _relaunch_under_torchrun(
    gpus: int,
    agent_cfg: Any,
    runner_args: dict[str, str],
    num_envs_per_rank: int | None,
    *,
    task_id: str,
) -> None:
    """Re-exec the current ``train`` invocation under torchrun.

    The parent precomputes the log directory and forwards it via ``--log-dir`` so all
    ranks land in the same directory (avoiding per-rank timestamp drift). The original
    argv is forwarded verbatim minus the ``--gpus`` / ``--num-envs`` / ``--num-envs-per-gpu``
    tokens; if either env-count flag was set, an authoritative ``--num-envs-per-gpu N``
    is injected so each worker sees the parent-resolved per-rank value. The resolved
    ``task_id`` is also injected when missing from the forwarded argv — otherwise a
    parent-side interactive pick would force every torchrun worker back into the
    questionary picker, producing repeated CPR-probe warnings (and a blocked launch).
    """
    from genelab.rl.runner import resolve_log_dir

    log_root_raw = runner_args.get("log_dir")
    log_root = Path(log_root_raw) if log_root_raw else Path("logs") / "rsl_rl"
    log_dir = resolve_log_dir(log_root, agent_cfg.experiment_name, agent_cfg.run_name)

    inner = _strip_distributed_flags(sys.argv[1:])
    if task_id not in inner:
        try:
            insert_at = inner.index("train") + 1
        except ValueError:
            insert_at = 0
        inner.insert(insert_at, task_id)
    if num_envs_per_rank is not None:
        inner += ["--num-envs-per-gpu", str(num_envs_per_rank)]
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


_STRIPPABLE_DISTRIBUTED_FLAGS: Final[frozenset[str]] = frozenset(
    {"--gpus", "--num-envs", "--num_envs", "--num-envs-per-gpu", "--num_envs_per_gpu"}
)


def _strip_distributed_flags(tokens: list[str]) -> list[str]:
    """Drop ``--gpus`` / ``--num-envs`` / ``--num-envs-per-gpu`` (and the ``_``-spelt forms),
    in both ``--flag value`` and ``--flag=value`` shapes, from a forwarded argv slice."""
    out: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in _STRIPPABLE_DISTRIBUTED_FLAGS:
            skip_next = True
            continue
        if any(tok.startswith(f"{flag}=") for flag in _STRIPPABLE_DISTRIBUTED_FLAGS):
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
