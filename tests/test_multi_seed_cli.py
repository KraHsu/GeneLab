"""Unit tests for the ``genelab train --seeds`` multi-seed orchestrator.

The full orchestrator spawns subprocesses, which is slow and Genesis-dependent
to exercise end-to-end. These tests cover its pure helpers — seed-list parsing,
argv stripping, log-dir extraction, parent-dir resolution — so the wiring stays
deterministic without standing up Genesis.

End-to-end ``--seeds`` smoke tests live alongside the M1 verification block in
the plan and require a Genesis runtime to run.
"""

from pathlib import Path

import pytest

from genelab.cli import (
    _extract_log_dir_flag,
    _parse_seed_list,
    _resolve_multi_seed_parent,
    _strip_multi_seed_flags,
)


def test_parse_seed_list_handles_whitespace_and_singletons() -> None:
    assert _parse_seed_list("1,2,3") == [1, 2, 3]
    assert _parse_seed_list(" 1 , 2 , 3 ") == [1, 2, 3]
    assert _parse_seed_list("42") == [42]
    assert _parse_seed_list("1,,2") == [1, 2]


def test_parse_seed_list_rejects_non_integer() -> None:
    with pytest.raises(SystemExit):
        _parse_seed_list("1,a,3")


def test_strip_multi_seed_flags_handles_space_and_equals_forms() -> None:
    tokens = [
        "TaskA",
        "--seeds",
        "1,2,3",
        "--parallel",
        "2",
        "--seed",
        "0",
        "--log_dir",
        "/x",
        "--max_iterations",
        "100",
    ]
    assert _strip_multi_seed_flags(tokens) == ["TaskA", "--max_iterations", "100"]

    eq_tokens = ["TaskA", "--seeds=1,2", "--parallel=2", "--log-dir=/x", "--num_envs", "64"]
    assert _strip_multi_seed_flags(eq_tokens) == ["TaskA", "--num_envs", "64"]


def test_strip_multi_seed_flags_preserves_unrelated_flags() -> None:
    tokens = ["TaskA", "--num_envs", "64", "--gpus", "1", "env.simulation.dt=0.01"]
    assert _strip_multi_seed_flags(tokens) == tokens


def test_extract_log_dir_flag_both_spellings_and_forms() -> None:
    assert _extract_log_dir_flag(["--log-dir", "/a"]) == Path("/a")
    assert _extract_log_dir_flag(["--log_dir", "/b"]) == Path("/b")
    assert _extract_log_dir_flag(["--log-dir=/c"]) == Path("/c")
    assert _extract_log_dir_flag(["--log_dir=/d"]) == Path("/d")
    assert _extract_log_dir_flag(["--seed", "0"]) is None


def test_resolve_multi_seed_parent_honors_explicit_log_dir() -> None:
    tokens = ["TaskA", "--log-dir", "/tmp/run-x", "--seeds", "1,2,3"]
    assert _resolve_multi_seed_parent(tokens, "TaskA") == Path("/tmp/run-x")


def test_resolve_multi_seed_parent_defaults_under_logs_multi_seed() -> None:
    tokens = ["TaskA", "--seeds", "1,2,3"]
    parent = _resolve_multi_seed_parent(tokens, "TaskA")
    parts = parent.parts
    # Shape: logs/multi-seed/<task>/<timestamp>
    assert parts[0] == "logs"
    assert parts[1] == "multi-seed"
    assert parts[2] == "TaskA"
    assert len(parts) == 4
