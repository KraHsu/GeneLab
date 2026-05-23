"""Task resolution + interactive override application (split out of ``cli/__init__.py``).

``_configured_task`` is the shared front half of the ``play`` / ``train`` commands:
parse run args (falling back to the interactive task picker), resolve the task id to
a :class:`~genelab.registry.Runnable`, retarget train/play short flags, and apply the
overrides — prompting for a correction when an override path is misspelled. The
dispatcher (``cli/__init__.py``) imports ``_configured_task`` and calls it from both
command callbacks.

The interactive pickers are imported by name here, so a test that stubs the picker
must patch ``genelab.cli._resolve.<picker>`` (this module's binding), not only
``genelab.cli._interactive`` — see ``_patch_picker`` in ``tests/test_cli_routing.py``.
"""

import re
from typing import Final, cast

from genelab.cli._argv import parse_run_args, split_prof_keys, split_runner_keys
from genelab.cli._interactive import (
    pick_name_interactively,
    pick_override_path,
    pick_task_interactively,
)
from genelab.cli._progress import fetch_progress
from genelab.cli._render import iter_overridable_paths
from genelab.configs import SimulationCfg, apply_overrides
from genelab.registry import TASKS, Runnable

# ``_configured_task`` is consumed by ``cli/__init__.py`` (play/train) and by tests
# via ``from genelab.cli import _configured_task``; declare it exported so pyright
# does not flag the underscore-named entry point as unused at its def site.
__all__ = ["_configured_task"]


def _configured_task(
    tokens: list[str], *, command: str
) -> tuple[Runnable, dict[str, str], dict[str, str]]:
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


def _resolve_task(task_id: str) -> Runnable:
    try:
        with fetch_progress():
            return cast(Runnable, TASKS.get(task_id))
    except KeyError as exc:
        picked = pick_name_interactively(TASKS.names(), f"Unknown task {task_id!r}. Pick one:")
        if picked is None or picked == task_id:
            raise SystemExit(str(exc)) from exc
        with fetch_progress():
            return cast(Runnable, TASKS.get(picked))


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
