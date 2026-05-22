"""``run_benchmark`` — evaluate a suite of tasks and aggregate reference numbers.

The orchestration layer behind ``genelab benchmark``: load a suite spec (a JSON list of
``{task, checkpoint, ...}`` entries), run :func:`genelab.rl.eval_task.eval_task` for each,
and aggregate the per-task metrics into one report. Optionally compare ``return_mean``
against a stored reference report and flag regressions, so ``benchmark`` can act as a CI
gate.

Lives in ``rl`` (next to ``eval_task``) so the CLI keeps the allowed ``cli -> rl``
direction and ``import genelab.cli`` stays torch-free (the command imports this module
function-locally, like ``eval``).
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genelab.rl.eval_task import eval_task


@dataclass
class BenchmarkEntry:
    """One benchmark-suite entry: a task + checkpoint and its eval settings."""

    task: str
    checkpoint: str
    episodes: int = 100
    seed: int = 0
    num_envs: int = 64


def load_suite(path: Path) -> list[BenchmarkEntry]:
    """Parse a benchmark suite JSON file — a list of entry objects.

    Each object needs ``task`` and ``checkpoint``; ``episodes`` / ``seed`` / ``num_envs``
    are optional (defaults match :func:`eval_task`).
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"benchmark suite {str(path)!r} must be a JSON list of entries")
    entries: list[BenchmarkEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "task" not in item or "checkpoint" not in item:
            raise ValueError(
                f"benchmark suite entry {i} must be an object with 'task' and 'checkpoint'"
            )
        entries.append(
            BenchmarkEntry(
                task=str(item["task"]),
                checkpoint=str(item["checkpoint"]),
                episodes=int(item.get("episodes", 100)),
                seed=int(item.get("seed", 0)),
                num_envs=int(item.get("num_envs", 64)),
            )
        )
    return entries


def _reference_returns(reference: dict[str, Any]) -> dict[str, float]:
    """Map ``task -> return_mean`` from a prior benchmark report payload."""
    out: dict[str, float] = {}
    for entry in reference.get("entries", []):
        rm = entry.get("metrics", {}).get("return_mean")
        if rm is not None:
            out[str(entry.get("task"))] = float(rm)
    return out


def detect_regressions(
    entries_out: list[dict[str, Any]], reference: dict[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    """Flag tasks whose ``return_mean`` dropped > ``tolerance`` fraction below the reference.

    Tasks absent from the reference (or with a ``null`` return) are skipped, not failed —
    a new task in the suite is not a regression.
    """
    ref_returns = _reference_returns(reference)
    regressions: list[dict[str, Any]] = []
    for entry in entries_out:
        task = str(entry.get("task"))
        cur = entry.get("metrics", {}).get("return_mean")
        ref = ref_returns.get(task)
        if cur is None or ref is None or ref == 0:
            continue
        drop = (ref - float(cur)) / abs(ref)
        if drop > tolerance:
            regressions.append(
                {
                    "task": task,
                    "metric": "return_mean",
                    "reference": ref,
                    "current": float(cur),
                    "drop_fraction": drop,
                }
            )
    return regressions


def run_benchmark(
    entries: list[BenchmarkEntry],
    *,
    out_path: Path | None = None,
    reference: dict[str, Any] | None = None,
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """Run ``eval_task`` for each entry and aggregate into one report dict.

    The report schema::

        {
            "generated_at": ISO-8601 str,
            "num_tasks": int,
            "entries": [ <eval_task payload>, ... ],   # per-task metrics
            "regressions": [ {task, metric, reference, current, drop_fraction}, ... ],
        }

    When ``reference`` (a prior report payload) is given, ``regressions`` lists tasks whose
    ``return_mean`` fell by more than ``tolerance`` of the reference value.
    """
    entries_out: list[dict[str, Any]] = []
    for entry in entries:
        _result, payload = eval_task(
            entry.task,
            Path(entry.checkpoint),
            num_envs=entry.num_envs,
            episodes=entry.episodes,
            seed=entry.seed,
        )
        entries_out.append(payload)

    report: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "num_tasks": len(entries_out),
        "entries": entries_out,
        "regressions": (
            detect_regressions(entries_out, reference, tolerance) if reference is not None else []
        ),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
    return report
