"""Tests for ``genelab.rl.benchmark`` — suite loading, aggregation, regression detection.

``eval_task`` is monkeypatched so these run without a Genesis runtime / checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genelab.rl import benchmark as bench
from genelab.rl.benchmark import BenchmarkEntry, detect_regressions, load_suite, run_benchmark


def _fake_eval_factory(returns: dict[str, float]):
    """Return a fake ``eval_task`` that yields a canned payload per task."""

    def _fake_eval(task: str, checkpoint: Path, **kwargs: Any) -> tuple[None, dict[str, Any]]:
        return None, {
            "task": task,
            "checkpoint": str(checkpoint),
            "num_episodes": kwargs.get("episodes", 100),
            "metrics": {
                "return_mean": returns[task],
                "return_std": 0.0,
                "length_mean": 100.0,
                "success_rate": None,
            },
            "seed": kwargs.get("seed", 0),
        }

    return _fake_eval


def test_load_suite_parses_entries(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            [
                {"task": "a", "checkpoint": "ck_a.pt"},
                {"task": "b", "checkpoint": "ck_b.pt", "episodes": 50, "seed": 3, "num_envs": 8},
            ]
        )
    )
    entries = load_suite(path)
    assert entries[0] == BenchmarkEntry(task="a", checkpoint="ck_a.pt")
    assert entries[1] == BenchmarkEntry(
        task="b", checkpoint="ck_b.pt", episodes=50, seed=3, num_envs=8
    )


def test_load_suite_rejects_bad_shapes(tmp_path: Path) -> None:
    bad_list = tmp_path / "obj.json"
    bad_list.write_text(json.dumps({"task": "a"}))  # not a list
    with pytest.raises(ValueError, match="must be a JSON list"):
        load_suite(bad_list)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps([{"task": "a"}]))  # no checkpoint
    with pytest.raises(ValueError, match="'task' and 'checkpoint'"):
        load_suite(missing)


def test_run_benchmark_aggregates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bench, "eval_task", _fake_eval_factory({"a": 10.0, "b": 20.0}))
    out = tmp_path / "report.json"
    report = run_benchmark(
        [BenchmarkEntry("a", "ck_a.pt"), BenchmarkEntry("b", "ck_b.pt")], out_path=out
    )
    assert report["num_tasks"] == 2
    assert [e["task"] for e in report["entries"]] == ["a", "b"]
    assert report["entries"][1]["metrics"]["return_mean"] == 20.0
    assert report["regressions"] == []
    # Written report round-trips.
    assert json.loads(out.read_text())["num_tasks"] == 2


def test_run_benchmark_flags_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reference had return 10.0; current run drops to 8.0 → 20% drop > 10% tolerance.
    reference = {"entries": [{"task": "a", "metrics": {"return_mean": 10.0}}]}
    monkeypatch.setattr(bench, "eval_task", _fake_eval_factory({"a": 8.0}))
    report = run_benchmark([BenchmarkEntry("a", "ck.pt")], reference=reference, tolerance=0.1)
    assert len(report["regressions"]) == 1
    reg = report["regressions"][0]
    assert reg["task"] == "a"
    assert reg["reference"] == 10.0 and reg["current"] == 8.0
    assert reg["drop_fraction"] == pytest.approx(0.2)


def test_run_benchmark_no_regression_within_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = {"entries": [{"task": "a", "metrics": {"return_mean": 10.0}}]}
    monkeypatch.setattr(bench, "eval_task", _fake_eval_factory({"a": 9.5}))  # 5% drop < 10%
    report = run_benchmark([BenchmarkEntry("a", "ck.pt")], reference=reference, tolerance=0.1)
    assert report["regressions"] == []


def test_detect_regressions_skips_unknown_and_improved() -> None:
    reference = {"entries": [{"task": "a", "metrics": {"return_mean": 10.0}}]}
    out = [
        {"task": "a", "metrics": {"return_mean": 12.0}},  # improved → not a regression
        {"task": "new", "metrics": {"return_mean": 1.0}},  # absent from reference → skipped
    ]
    assert detect_regressions(out, reference, tolerance=0.1) == []
