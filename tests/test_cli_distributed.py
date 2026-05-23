"""CLI distributed / torchrun plumbing: flag stripping, process-group shutdown,
log-dir detection, the torchrun relaunch command, and per-rank env-count math.

Split out of ``tests/test_cli.py`` by concern. The relaunch tests patch
``genelab.cli._distributed.{os.execvp,sys.argv}`` by absolute string path, so they
are unaffected by living in this module.
"""

import pytest


def test_strip_distributed_flags_drops_gpus_and_env_counts() -> None:
    from genelab.cli import _strip_distributed_flags

    # --gpus and --num-envs are both stripped (parent re-injects --num-envs-per-gpu).
    assert _strip_distributed_flags(["train", "TASK", "--gpus", "4", "--num-envs", "8"]) == [
        "train",
        "TASK",
    ]
    # = form is also stripped.
    assert _strip_distributed_flags(["--gpus=2", "--num-envs=8"]) == []
    # --num-envs-per-gpu also dropped; parent re-injects the authoritative value.
    assert _strip_distributed_flags(["train", "TASK", "--num-envs-per-gpu", "16"]) == [
        "train",
        "TASK",
    ]
    # Underscore spellings are recognised too.
    assert _strip_distributed_flags(["--num_envs", "8", "--num_envs_per_gpu", "4"]) == []
    # Unrelated flags pass through.
    assert _strip_distributed_flags(["--vis", "--seed", "42"]) == ["--vis", "--seed", "42"]


def test_shutdown_process_group_is_a_noop_when_not_initialized() -> None:
    """Single-GPU runs never call ``init_process_group``; the helper must no-op."""
    from genelab.utils.distributed import shutdown_process_group

    # The test environment is single-process; nothing was initialized.
    # The call should return cleanly without raising or trying to destroy anything.
    shutdown_process_group()


def test_shutdown_process_group_destroys_when_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distributed train runs need an explicit ``destroy_process_group`` to avoid the
    ``WARNING: destroy_process_group() was not called before program exit`` resource-leak
    notice that rsl_rl's ``OnPolicyRunner`` triggers on shutdown."""

    import torch.distributed as dist

    from genelab.utils import distributed

    destroyed = {"count": 0}

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda: destroyed.__setitem__("count", destroyed["count"] + 1),
    )

    distributed.shutdown_process_group()

    assert destroyed["count"] == 1


def test_has_log_dir_flag_recognises_both_spellings() -> None:
    from genelab.cli import _has_log_dir_flag

    assert _has_log_dir_flag(["--log-dir", "/tmp/x"])
    assert _has_log_dir_flag(["--log_dir", "/tmp/x"])
    assert _has_log_dir_flag(["--log-dir=/tmp/x"])
    assert _has_log_dir_flag(["--log_dir=/tmp/x"])
    assert not _has_log_dir_flag(["--num-envs", "8"])


def test_relaunch_under_torchrun_builds_expected_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    from genelab.cli import _relaunch_under_torchrun

    @dataclass
    class _FakeAgentCfg:
        experiment_name: str = "test-exp"
        run_name: str = "unit"

    captured: dict[str, object] = {}

    def _fake_execvp(file: str, args: list[str]) -> None:
        captured["file"] = file
        captured["args"] = args

    monkeypatch.setattr("genelab.cli._distributed.os.execvp", _fake_execvp)
    monkeypatch.setattr(
        "genelab.cli._distributed.sys.argv",
        ["genelab", "train", "TASK_ID", "--gpus", "4", "--num-envs", "8"],
    )

    _relaunch_under_torchrun(
        4, _FakeAgentCfg(), runner_args={}, num_envs_per_rank=2, task_id="TASK_ID"
    )

    args = captured["args"]
    assert isinstance(args, list)
    # python -m torch.distributed.run --standalone --nproc_per_node=4 -m genelab.cli ...
    assert args[1:7] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "genelab.cli",
    ]
    # --gpus and the original --num-envs were stripped from the forwarded tokens.
    assert "--gpus" not in args
    assert "--num-envs" not in args
    # The per-rank value (8 total / 4 ranks = 2) was injected as --num-envs-per-gpu.
    assert "--num-envs-per-gpu" in args
    per_gpu_index = args.index("--num-envs-per-gpu")
    assert args[per_gpu_index + 1] == "2"
    assert "train" in args and "TASK_ID" in args
    # A --log-dir was injected because the original argv had none
    assert "--log-dir" in args
    log_dir_index = args.index("--log-dir")
    assert "test-exp" in args[log_dir_index + 1]


def test_relaunch_under_torchrun_preserves_explicit_log_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    from genelab.cli import _relaunch_under_torchrun

    @dataclass
    class _FakeAgentCfg:
        experiment_name: str = "exp"
        run_name: str = ""

    captured: dict[str, object] = {}

    def _fake_execvp(file: str, args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("genelab.cli._distributed.os.execvp", _fake_execvp)
    monkeypatch.setattr(
        "genelab.cli._distributed.sys.argv",
        ["genelab", "train", "TASK", "--gpus", "2", "--log-dir", "/tmp/keep-this"],
    )

    _relaunch_under_torchrun(
        2,
        _FakeAgentCfg(),
        runner_args={"log_dir": "/tmp/keep-this"},
        num_envs_per_rank=None,
        task_id="TASK",
    )

    args = captured["args"]
    assert isinstance(args, list)
    # The user's explicit --log-dir is forwarded; we should not inject another.
    assert args.count("--log-dir") == 1
    log_dir_index = args.index("--log-dir")
    assert args[log_dir_index + 1] == "/tmp/keep-this"


def test_relaunch_under_torchrun_injects_interactively_picked_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the parent resolved the task via the interactive picker, the original argv
    has no task id; the relaunch must inject it so torchrun workers don't re-enter
    the picker (which probes the tty for CPR and floods stderr with warnings)."""

    from dataclasses import dataclass

    from genelab.cli import _relaunch_under_torchrun

    @dataclass
    class _FakeAgentCfg:
        experiment_name: str = "exp"
        run_name: str = ""

    captured: dict[str, object] = {}

    def _fake_execvp(file: str, args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("genelab.cli._distributed.os.execvp", _fake_execvp)
    # No task id in argv — parent picked it interactively.
    monkeypatch.setattr(
        "genelab.cli._distributed.sys.argv",
        ["genelab", "train", "--num_envs", "8192", "--steps", "20", "--gpus", "8"],
    )

    _relaunch_under_torchrun(
        8,
        _FakeAgentCfg(),
        runner_args={},
        num_envs_per_rank=1024,
        task_id="Picked-Task-v0",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "Picked-Task-v0" in args, "resolved task id must be forwarded to torchrun workers"
    # It should land immediately after the `train` token so child argv normalization is happy.
    train_index = args.index("train")
    assert args[train_index + 1] == "Picked-Task-v0"
    # It should not be duplicated even though sys.argv didn't contain it.
    assert args.count("Picked-Task-v0") == 1


def test_relaunch_under_torchrun_does_not_duplicate_explicit_task_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    from genelab.cli import _relaunch_under_torchrun

    @dataclass
    class _FakeAgentCfg:
        experiment_name: str = "exp"
        run_name: str = ""

    captured: dict[str, object] = {}

    def _fake_execvp(file: str, args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr("genelab.cli._distributed.os.execvp", _fake_execvp)
    monkeypatch.setattr(
        "genelab.cli._distributed.sys.argv",
        ["genelab", "train", "Explicit-Task-v0", "--gpus", "2"],
    )

    _relaunch_under_torchrun(
        2,
        _FakeAgentCfg(),
        runner_args={},
        num_envs_per_rank=None,
        task_id="Explicit-Task-v0",
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert args.count("Explicit-Task-v0") == 1


def test_resolve_per_rank_num_envs_divides_total_by_gpus() -> None:
    from genelab.cli import _resolve_per_rank_num_envs

    runner_args: dict[str, str] = {"num_envs": "8"}
    assert _resolve_per_rank_num_envs(runner_args, gpus=4) == 2
    assert runner_args == {}  # both keys consumed
    assert _resolve_per_rank_num_envs({"num_envs": "4096"}, gpus=1) == 4096


def test_resolve_per_rank_num_envs_returns_per_gpu_verbatim() -> None:
    from genelab.cli import _resolve_per_rank_num_envs

    assert _resolve_per_rank_num_envs({"num_envs_per_gpu": "1024"}, gpus=4) == 1024
    # Per-gpu bypasses divisibility checks — the user took responsibility.
    assert _resolve_per_rank_num_envs({"num_envs_per_gpu": "37"}, gpus=4) == 37


def test_resolve_per_rank_num_envs_errors_on_mutual_exclusion() -> None:
    from genelab.cli import _resolve_per_rank_num_envs

    with pytest.raises(SystemExit) as excinfo:
        _resolve_per_rank_num_envs({"num_envs": "8", "num_envs_per_gpu": "2"}, gpus=2)
    assert "mutually exclusive" in str(excinfo.value)


def test_resolve_per_rank_num_envs_errors_on_non_divisible_total() -> None:
    from genelab.cli import _resolve_per_rank_num_envs

    with pytest.raises(SystemExit) as excinfo:
        _resolve_per_rank_num_envs({"num_envs": "4097"}, gpus=2)
    assert "not divisible" in str(excinfo.value)


def test_resolve_per_rank_num_envs_returns_none_when_unset() -> None:
    from genelab.cli import _resolve_per_rank_num_envs

    assert _resolve_per_rank_num_envs({}, gpus=4) is None
