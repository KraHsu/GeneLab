"""Unified Typer + Rich command-line entry point for registered GeneLab tasks."""

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, cast

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
from genelab.cli._distributed import (
    _extract_log_dir_flag as _extract_log_dir_flag,
    _has_log_dir_flag as _has_log_dir_flag,
    _relaunch_under_torchrun as _relaunch_under_torchrun,
    _resolve_per_rank_num_envs as _resolve_per_rank_num_envs,
    _strip_distributed_flags as _strip_distributed_flags,
    _strip_flag_value_pairs as _strip_flag_value_pairs,
)
from genelab.cli._interactive import (
    pick_agent_kind,
    pick_name_interactively,
    pick_override_path,
    pick_task_interactively,
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
    iter_overridable_paths,
    render_cache,
    render_entry_info,
    render_main_help,
    render_registry,
)
from genelab.cli._scaffold import create_project_skeleton
from genelab.configs import SimulationCfg, apply_overrides
from genelab.registry import (
    TASKS,
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
  --eval_every K        Run a deterministic eval every K iters and save
                        best_model.<ext> on improvement (train only).
  --eval_episodes N     Episodes to roll out per eval (train only, default 10).
  --eval_num_envs N     Envs used during eval (train only, default = train num).
  --eval_seed N         Seed for the eval rollout (train only, default 0).
  --seeds 1,2,3         Fan out into N independent train runs, one per seed
                        (train only). Each child gets --log_dir <parent>/seed_<S>.
  --parallel N          Cap concurrent multi-seed train workers (train only,
                        default 1). Ignored unless --seeds is set.

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
    from genelab.cli._eval import eval_task

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
        for short_key in SimulationCfg.play_retargeted_keys():
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
    # Pass the CLI's already-overridden cfg (play_env when configured): TASKS.get
    # returns a fresh task each call, so the runner re-resolving would discard the
    # --vis / --gpu / --a.env.* overrides applied above.
    play_env_cfg = getattr(task_cfg, "play_env", None)
    if play_env_cfg is None:
        play_env_cfg = getattr(task_cfg, "env", None)
    play_task(
        task_id,
        env_cfg=play_env_cfg,
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
    from genelab.rl import select_backend, train_task

    # The backend is chosen by the agent cfg type (RSL-RL, skrl, ...); an
    # unregistered type raises a clear error here instead of deep in the runner.
    try:
        backend = select_backend(agent_cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route through RL train helper")

    gpus_raw = runner_args.pop("gpus", None)
    gpus = int(gpus_raw) if gpus_raw is not None else 1
    num_envs_per_rank = _resolve_per_rank_num_envs(runner_args, gpus=gpus)
    if gpus > 1:
        if backend.name != "rsl_rl":
            raise SystemExit(
                f"multi-GPU training (--gpus {gpus}) is only supported by the RSL-RL "
                f"backend; this task uses the {backend.name!r} backend"
            )
        if "TORCHELASTIC_RUN_ID" not in os.environ:
            _relaunch_under_torchrun(
                gpus, agent_cfg, runner_args, num_envs_per_rank, task_id=task_id
            )
            return

    max_iter_raw = runner_args.get("max_iterations")
    seed_raw = runner_args.get("seed")
    log_dir_raw = runner_args.get("log_dir")
    from genelab.rl.eval_callback import EvalCallbackCfg

    eval_callback = EvalCallbackCfg.from_args(runner_args)
    train_task(
        task_id,
        agent_cfg,
        # Pass the CLI's already-overridden cfg: TASKS.get returns a fresh task
        # each call, so the runner re-resolving would discard --gpu / --a.env.* .
        env_cfg=getattr(task_cfg, "env", None),
        num_envs=num_envs_per_rank,
        max_iterations=int(max_iter_raw) if max_iter_raw is not None else None,
        seed=int(seed_raw) if seed_raw is not None else None,
        log_dir=Path(log_dir_raw) if log_dir_raw is not None else None,
        eval_callback=eval_callback,
        **_coerce_prof_kwargs(prof_args),
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
