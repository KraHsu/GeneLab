"""Multi-seed train orchestration for ``genelab train --seeds 1,2,3``.

Fans a single ``genelab train`` invocation out into one subprocess per seed,
factored out of ``cli/__init__.py``. The CLI
dispatcher (``train_cmd``) is the only caller; this module imports the argv-strip
helpers from ``cli/_distributed.py`` and nothing from ``genelab.cli`` itself at
runtime, so no import cycle is introduced.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import typer

from genelab.cli._distributed import _extract_log_dir_flag, _strip_flag_value_pairs

if TYPE_CHECKING:
    from genelab.registry import Runnable

# These keep their leading underscores (they are CLI-package-private, re-exported
# through ``cli/__init__.py``) but are this module's external API — listing them in
# ``__all__`` marks them exported so they are not flagged as unused at the def site.
__all__ = [
    "_dispatch_multi_seed_train",
    "_parse_seed_list",
    "_resolve_multi_seed_parent",
    "_strip_multi_seed_flags",
]

_STRIPPABLE_MULTI_SEED_FLAGS: Final[frozenset[str]] = frozenset(
    {"--seeds", "--parallel", "--seed", "--log-dir", "--log_dir"}
)


def _dispatch_multi_seed_train(
    task: Runnable,
    tokens: list[str],
    runner_args: dict[str, str],
) -> None:
    """Fan out ``genelab train`` into one subprocess per seed.

    ``tokens`` is the raw ``train`` token slice (post task-id normalization) that
    ``_configured_task`` consumed. We strip ``--seeds`` / ``--parallel`` /
    ``--seed`` / ``--log[-_]dir`` from it (they are owned by this orchestrator) and
    forward the rest verbatim to each child, plus a per-child ``--seed S`` and
    ``--log_dir <parent>/seed_<S>``. Concurrency is capped by ``--parallel`` via a
    :class:`ThreadPoolExecutor` (subprocesses are the unit of work; threads block
    waiting for them, so a thread pool is the right shape).

    Raises ``SystemExit`` with a non-zero status if any child fails so CI / scripts
    can react. Successful seeds are still reported individually.
    """
    import concurrent.futures
    import datetime as _dt
    import subprocess

    seeds_raw = runner_args.pop("seeds")
    parallel_raw = runner_args.pop("parallel", None)
    seeds = _parse_seed_list(seeds_raw)
    if not seeds:
        raise SystemExit(f"--seeds must contain at least one int; got {seeds_raw!r}")
    parallel = int(parallel_raw) if parallel_raw is not None else 1
    if parallel < 1:
        raise SystemExit(f"--parallel must be ≥ 1; got {parallel}")
    parallel = min(parallel, len(seeds))

    task_cfg = getattr(task, "cfg", None)
    task_id = getattr(task_cfg, "name", None)
    if not isinstance(task_id, str):
        raise SystemExit("task config is missing 'name'; cannot route multi-seed train")

    parent_dir = _resolve_multi_seed_parent(tokens, task_id)
    parent_dir.mkdir(parents=True, exist_ok=True)
    stripped = _strip_multi_seed_flags(tokens)
    if task_id not in stripped:
        stripped = [task_id, *stripped]

    typer.echo(
        f"multi-seed train: task={task_id} seeds={seeds} parallel={parallel} parent={parent_dir}"
    )

    def _run_one(seed: int) -> tuple[int, int]:
        seed_dir = parent_dir / f"seed_{seed}"
        cmd = [
            sys.executable,
            "-m",
            "genelab.cli",
            "train",
            *stripped,
            "--seed",
            str(seed),
            "--log_dir",
            str(seed_dir),
        ]
        proc = subprocess.run(cmd, check=False)
        return seed, proc.returncode

    failures: list[int] = []
    started_at = _dt.datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(_run_one, s) for s in seeds]
        for fut in concurrent.futures.as_completed(futures):
            seed, rc = fut.result()
            status = "OK" if rc == 0 else f"FAILED rc={rc}"
            typer.echo(f"[multi-seed] seed={seed} {status}")
            if rc != 0:
                failures.append(seed)
    elapsed = (_dt.datetime.now() - started_at).total_seconds()
    typer.echo(
        f"multi-seed train: {len(seeds) - len(failures)}/{len(seeds)} ok "
        f"in {elapsed:.1f}s (parent={parent_dir})"
    )
    if failures:
        raise SystemExit(
            f"multi-seed train failed for seeds {failures} "
            f"({len(failures)}/{len(seeds)} children exited non-zero)"
        )


def _parse_seed_list(raw: str) -> list[int]:
    """Parse a comma-separated ``--seeds`` value into ``[int, ...]`` preserving order."""
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise SystemExit(
                f"--seeds entries must be ints separated by ','; got {token!r} in {raw!r}"
            ) from exc
    return seeds


def _resolve_multi_seed_parent(tokens: list[str], task_id: str) -> Path:
    """Resolve the parent directory under which per-seed log dirs are created.

    Honors a user-provided ``--log-dir`` (or ``--log_dir``) verbatim; otherwise
    creates ``logs/multi-seed/<task_id>/<YYYY-MM-DD_HH-MM-SS>/``. Each child train
    process gets ``<parent>/seed_<S>`` so all seeds of one launch land together.
    """
    import datetime as _dt

    explicit = _extract_log_dir_flag(tokens)
    if explicit is not None:
        return explicit
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("logs") / "multi-seed" / task_id / timestamp


def _strip_multi_seed_flags(tokens: list[str]) -> list[str]:
    """Drop ``--seeds`` / ``--parallel`` / ``--seed`` / ``--log[-_]dir`` from a forwarded
    argv slice. Used by the multi-seed orchestrator before re-launching one ``genelab
    train`` per seed: each child gets a fresh ``--seed S`` and ``--log_dir <parent>/seed_S``.
    """
    return _strip_flag_value_pairs(tokens, _STRIPPABLE_MULTI_SEED_FLAGS)
