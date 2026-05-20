"""Argv preprocessing and override parsing for the GeneLab CLI.

These helpers are extracted from ``genelab.cli`` so they can be reused (and unit-tested)
independently of the Typer wiring.
"""

from typing import Final

_COMMANDS: Final[frozenset[str]] = frozenset({"cache", "list", "play", "prof", "project", "train"})

_GLOBAL_BOOL_FLAGS: Final[frozenset[str]] = frozenset(
    {"-h", "--help", "--version", "--no-entry-points"}
)

_PROF_BOOL_FLAGS: Final[frozenset[str]] = frozenset(
    {"--prof", "--prof-record-shapes", "--prof-with-stack"}
)

_RUN_BOOL_SHORTCUTS: Final[frozenset[str]] = frozenset({"-v", "--vis", "--gpu", *_PROF_BOOL_FLAGS})

RUNNER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "num_envs",
        "num_envs_per_gpu",
        "checkpoint",
        "max_iterations",
        "seed",
        "log_dir",
        "agent",
        "gpus",
        "eval_every",
        "eval_episodes",
        "eval_num_envs",
        "eval_seed",
    }
)

PROF_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prof",
        "prof_out",
        "prof_wait",
        "prof_warmup",
        "prof_active",
        "prof_repeat",
        "prof_record_shapes",
        "prof_with_stack",
    }
)


def normalize_argv(argv: list[str] | None) -> list[str] | None:
    """Move a trailing task id to the canonical position right after ``play``/``train``.

    Example: ``["play", "--steps", "5", "TASK", "--vis"]``
    becomes  ``["play", "TASK", "--steps", "5", "--vis"]``.

    Returns ``argv`` unchanged for any other shape (other commands, missing task, etc.).
    """

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


def parse_run_args(tokens: list[str]) -> tuple[str, dict[str, str]]:
    """Split a ``play``/``train`` token slice into ``(task_id, overrides)``.

    Shortcut flags ``-v``/``--vis``/``--gpu``/``--steps`` are rewritten into the
    canonical ``env.simulation.*`` override paths. Other ``--a.b.c VALUE`` flags are
    forwarded as-is (with ``-`` replaced by ``_`` in the key).
    """

    task_id: str | None = None
    overrides: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-v", "--vis"}:
            overrides["env.simulation.vis"] = "true"
            index += 1
            continue
        if token == "--gpu":
            overrides["env.simulation.gpu"] = "true"
            index += 1
            continue
        if token == "--steps":
            overrides["env.simulation.steps"] = _require_value(tokens, index)
            index += 2
            continue
        if token in _PROF_BOOL_FLAGS:
            overrides[token[2:].replace("-", "_")] = "true"
            index += 1
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


def split_runner_keys(overrides: dict[str, str]) -> dict[str, str]:
    """Move runner-only keys out of ``overrides`` and return them as a new dict."""

    runner_args: dict[str, str] = {}
    for key in list(overrides.keys()):
        if key in RUNNER_KEYS:
            runner_args[key] = overrides.pop(key)
    return runner_args


def split_prof_keys(overrides: dict[str, str]) -> dict[str, str]:
    """Move profiler-only keys out of ``overrides`` and return them as a new dict."""

    prof_args: dict[str, str] = {}
    for key in list(overrides.keys()):
        if key in PROF_KEYS:
            prof_args[key] = overrides.pop(key)
    return prof_args


def _find_command_index(argv: list[str]) -> int | None:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _COMMANDS:
            return index
        if token in _GLOBAL_BOOL_FLAGS:
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
        if token in _RUN_BOOL_SHORTCUTS:
            index += 1
            continue
        if token.startswith("--"):
            index += 2
            continue
        return index
    return None


def _require_value(tokens: list[str], key_index: int) -> str:
    key = tokens[key_index]
    if key_index + 1 >= len(tokens):
        raise SystemExit(f"missing value for override {key}")
    value = tokens[key_index + 1]
    if value.startswith("--"):
        raise SystemExit(f"missing value for override {key}")
    return value
