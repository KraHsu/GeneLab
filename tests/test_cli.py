import re
import sys
from pathlib import Path

import pytest

from genelab import __doc__ as genelab_doc
from genelab.cli import main
from genelab.configs import ManagerBasedEnvCfg, apply_overrides
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    load_entrypoint_extensions,
    load_extension_module,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_cli_outputs_registered_hint(capsys: pytest.CaptureFixture[str]) -> None:
    main([])

    assert "genelab list tasks" in capsys.readouterr().out


def test_bare_invocation_prints_landing_page(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--no-entry-points"])

    out = capsys.readouterr().out
    assert "GeneLab" in out
    assert "Quickstart" in out
    assert "Commands" in out
    assert "Registered:" in out
    assert "genelab list tasks" in out
    assert "genelab info NAME" in out


def test_play_help_documents_runner_keys(capsys: pytest.CaptureFixture[str]) -> None:
    from genelab.cli import RUNNER_KEYS

    main(["play", "--help"])

    out = _strip_ansi(capsys.readouterr().out)
    for key in RUNNER_KEYS:
        assert key in out, f"runner key {key!r} missing from `play --help`"


def test_play_help_documents_short_flag_grammar(capsys: pytest.CaptureFixture[str]) -> None:
    main(["play", "--help"])

    out = _strip_ansi(capsys.readouterr().out)
    assert "--vis" in out
    assert "--gpu" in out
    assert "--steps" in out
    assert "env.simulation" in out


def test_register_task_accepts_examples_kwarg() -> None:
    from genelab.registry import Registry

    isolated: Registry[object] = Registry("test-task")
    entry = isolated.register(
        "Examples-Roundtrip-v0",
        lambda: None,
        description="Examples round-trip test entry.",
        examples=["genelab play Examples-Roundtrip-v0"],
    )

    assert entry.examples == ("genelab play Examples-Roundtrip-v0",)


def test_register_task_examples_default_to_empty_tuple() -> None:
    from genelab.registry import Registry

    isolated: Registry[object] = Registry("test-task")
    entry = isolated.register(
        "Defaults-v0",
        lambda: None,
        description="Default examples test entry.",
    )

    assert entry.examples == ()


def test_info_renders_examples_and_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "--no-entry-points",
            "--import",
            "tests.fake_extension",
            "info",
            "External-Fake-Task-v0",
        ]
    )

    out = capsys.readouterr().out
    assert "External-Fake-Task-v0" in out
    assert "Task from a fake external package." in out
    assert "genelab play External-Fake-Task-v0" in out
    assert "--steps 7" in out
    # cfg introspection surfaces the simulation fields that overrides walk through.
    assert "env.simulation.steps" in out


def test_info_unknown_name_errors(capsys: pytest.CaptureFixture[str]) -> None:
    try:
        main(["--no-entry-points", "info", "definitely-not-a-registered-name"])
    except SystemExit as exc:
        assert "definitely-not-a-registered-name" in str(exc)
        assert "available" in str(exc)
    else:
        raise AssertionError("expected info to exit on unknown name")
    capsys.readouterr()  # drain any output


def test_core_does_not_register_example_tasks_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--no-entry-points", "list", "tasks"])

    assert "GeneLab-Rubiks-Play-v0" not in capsys.readouterr().out


def test_example_extension_registers_robots_envs_and_tasks() -> None:
    load_extension_module("genelab_examples.tasks")

    assert "rubiks-cube" in ROBOTS.names()
    assert "wuji-hand" in ROBOTS.names()
    assert "rubiks-play" in ENVS.names()
    assert "wuji-hand-playback" in ENVS.names()
    assert "GeneLab-Rubiks-Play-v0" in TASKS.names()
    assert "GeneLab-Wuji-Hand-Playback-v0" in TASKS.names()


def test_list_tasks_shows_registered_bindings(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_examples.tasks", "list", "tasks"])

    out = capsys.readouterr().out
    assert "GeneLab-Rubiks-Play-v0" in out
    assert "env=rubiks-play" in out
    assert "robot=rubiks-cube" in out
    assert "GeneLab-Wuji-Hand-Playback-v0" in out


def test_package_positions_genelab_as_genesis_powered_lab_api() -> None:
    assert genelab_doc is not None
    assert "Isaac Lab API" in genelab_doc
    assert ManagerBasedEnvCfg.__name__ == "ManagerBasedEnvCfg"


def test_config_overrides_update_nested_task_config() -> None:
    load_extension_module("genelab_examples.tasks")
    task = TASKS.get("GeneLab-Wuji-Hand-Playback-v0")

    apply_overrides(
        task.cfg,
        {"env.robot.side": "left", "env.simulation.steps": "3", "env.reset_interval": "0"},
    )

    assert task.cfg.env.robot.side == "left"
    assert task.cfg.env.simulation.steps == 3
    assert task.cfg.env.reset_interval == 0


def test_cli_run_args_accept_flags_after_task() -> None:
    from genelab.cli import normalize_argv, parse_run_args

    argv = normalize_argv(
        ["play", "--steps", "5", "GeneLab-Rubiks-Play-v0", "--vis", "--env.robot.gap", "0.002"]
    )
    assert argv is not None
    task_id, overrides = parse_run_args(argv[1:])

    assert task_id == "GeneLab-Rubiks-Play-v0"
    assert overrides == {
        "env.simulation.steps": "5",
        "env.simulation.vis": "true",
        "env.robot.gap": "0.002",
    }


def test_cli_parses_agent_flag_value() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(
        ["External-Fake-Task-v0", "--agent", "random", "--num_envs", "4"]
    )

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {"agent": "random", "num_envs": "4"}


def test_cli_parses_gpus_flag_into_runner_args() -> None:
    from genelab.cli import RUNNER_KEYS, parse_run_args, split_runner_keys

    assert "gpus" in RUNNER_KEYS
    task_id, overrides = parse_run_args(["External-Fake-Task-v0", "--gpus", "4"])

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {"gpus": "4"}

    runner_args = split_runner_keys(overrides)
    assert runner_args == {"gpus": "4"}
    assert overrides == {}


def test_configured_task_train_mode_retargets_steps_to_max_iterations() -> None:
    """In train mode, ``--steps N`` is the short form for ``--max_iterations N``.

    ``env.simulation.steps`` is not consumed by ``train_task`` / ``ManagerBasedRlEnv`` —
    leaving the override on the env cfg would silently no-op and the user would see
    iterations counted to the cfg default instead of stopping at N.
    """
    from genelab.cli import _configured_task

    load_extension_module("genelab_examples.tasks")
    task, runner_args, _ = _configured_task(
        ["GeneLab-Wuji-Hand-Playback-v0", "--steps", "20"],
        command="train",
    )

    assert runner_args == {"max_iterations": "20"}
    # The env cfg's simulation.steps must NOT be set to 20; it should keep its default.
    assert task.cfg.env.simulation.steps != 20


def test_configured_task_train_mode_rejects_steps_with_explicit_max_iterations() -> None:
    """Passing both ``--steps`` and ``--max_iterations`` in train mode is a hard error."""
    from genelab.cli import _configured_task

    load_extension_module("genelab_examples.tasks")
    with pytest.raises(SystemExit) as excinfo:
        _configured_task(
            ["GeneLab-Wuji-Hand-Playback-v0", "--steps", "20", "--max_iterations", "100"],
            command="train",
        )
    assert "conflict" in str(excinfo.value)


def test_configured_task_play_mode_keeps_steps_as_env_simulation_steps() -> None:
    """Play mode behavior is unchanged: ``--steps`` lands on ``env.simulation.steps``."""
    from genelab.cli import _configured_task

    load_extension_module("genelab_examples.tasks")
    task, runner_args, _ = _configured_task(
        ["GeneLab-Wuji-Hand-Playback-v0", "--steps", "5"],
        command="play",
    )

    assert task.cfg.env.simulation.steps == 5
    assert "max_iterations" not in runner_args


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


def test_cli_rejects_invalid_agent_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab import cli as cli_module

    def _fake_play(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("play_task should not run when --agent value is invalid")

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(__import__("sys").modules, "genelab.rl", fake_rl)

    try:
        cli_module.main(
            [
                "--import",
                "tests.fake_extension",
                "play",
                "External-Fake-Task-v0",
                "--agent",
                "bogus",
            ]
        )
    except SystemExit as exc:
        assert "--agent" in str(exc)
    else:
        raise AssertionError("expected invalid --agent value to exit")


def test_cli_routes_agent_through_play_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab import cli as cli_module

    captured: dict[str, object] = {}

    def _fake_play(task_id: str, **kwargs: object) -> None:
        captured["task_id"] = task_id
        captured.update(kwargs)

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(__import__("sys").modules, "genelab.rl", fake_rl)

    cli_module.main(
        ["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--agent", "random"]
    )

    assert captured["task_id"] == "External-Fake-Task-v0"
    assert captured["agent"] == "random"


def test_core_namespaces_do_not_import_example_objects() -> None:
    import genelab.envs as envs
    import genelab.robots as robots
    import genelab.tasks as tasks

    assert not hasattr(robots, "create_rubiks_robot")
    assert not hasattr(envs, "create_rubiks_env")
    assert not hasattr(tasks, "rubiks_play_task_cfg")


def test_cli_import_loads_external_task(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "tests.fake_extension", "list", "tasks"])

    out = capsys.readouterr().out
    assert "External-Fake-Task-v0" in out
    assert "env=fake-extension-env" in out
    assert "robot=fake-extension-robot" in out


def test_cli_imported_external_task_can_run(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--steps", "7"])

    assert "played External-Fake-Task-v0 for 7 steps" in capsys.readouterr().out


def test_project_new_creates_importable_external_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["project", "new", "sample-project", "--path", str(tmp_path)])

    project = tmp_path / "sample-project"
    package_src = project / "src" / "sample_project"
    assert (project / "pyproject.toml").exists()
    assert (package_src / "tasks.py").exists()
    assert "genelab = { path = " in (project / "pyproject.toml").read_text()

    monkeypatch.syspath_prepend(str(project / "src"))
    main(["--import", "sample_project.tasks", "list", "tasks"])

    assert "SampleProject-Example-v0" in capsys.readouterr().out


def test_entrypoint_extensions_load_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeEntryPoint:
        name = "fake"
        value = "tests.fake_entrypoint_extension:register"
        dist = None

        def load(self):
            def register():
                register_task(
                    "EntryPoint-Fake-Task-v0",
                    lambda: None,
                    description="Task from a fake entry point.",
                )

            return register

    def fake_entry_points(*, group: str | None = None) -> list[_FakeEntryPoint]:
        assert group == "genelab.extensions"
        return [_FakeEntryPoint()]

    from genelab import registry
    from genelab.registry import register_task

    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        fake_entry_points,
    )  # pyright: ignore[reportUnknownArgumentType]

    load_entrypoint_extensions()

    assert "EntryPoint-Fake-Task-v0" in TASKS.names()


def test_train_registered_task_reports_unimplemented() -> None:
    try:
        main(["--import", "genelab_examples.tasks", "train", "GeneLab-Rubiks-Play-v0"])
    except SystemExit as exc:
        assert "training is not implemented" in str(exc)
    else:
        raise AssertionError("expected train to report unimplemented")


def test_parse_run_args_prof_toggle() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(["External-Fake-Task-v0", "--prof"])

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {"prof": "true"}


def test_parse_run_args_prof_value_flags() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(
        [
            "External-Fake-Task-v0",
            "--prof-out",
            "/tmp/profx",
            "--prof-wait",
            "20",
            "--prof-warmup",
            "3",
            "--prof-active",
            "7",
            "--prof-repeat",
            "1",
        ]
    )

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {
        "prof_out": "/tmp/profx",
        "prof_wait": "20",
        "prof_warmup": "3",
        "prof_active": "7",
        "prof_repeat": "1",
    }


def test_parse_run_args_prof_record_shapes_and_with_stack_are_bool_flags() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(
        ["External-Fake-Task-v0", "--prof-record-shapes", "--prof-with-stack"]
    )

    assert task_id == "External-Fake-Task-v0"
    assert overrides == {"prof_record_shapes": "true", "prof_with_stack": "true"}


def test_split_prof_keys_extracts_only_prof_entries() -> None:
    from genelab.cli import PROF_KEYS, split_prof_keys

    assert "prof" in PROF_KEYS
    overrides = {
        "prof": "true",
        "prof_out": "/tmp/x",
        "env.simulation.vis": "true",
        "num_envs": "8",
    }
    prof_args = split_prof_keys(overrides)

    assert prof_args == {"prof": "true", "prof_out": "/tmp/x"}
    assert overrides == {"env.simulation.vis": "true", "num_envs": "8"}


def test_normalize_argv_with_prof_keeps_task_after_command() -> None:
    from genelab.cli import normalize_argv

    argv = normalize_argv(["play", "--prof", "GeneLab-Rubiks-Play-v0", "--vis"])

    assert argv == ["play", "GeneLab-Rubiks-Play-v0", "--prof", "--vis"]


def test_maybe_profile_disabled_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab.rl.profiler import maybe_profile

    monkeypatch.delenv("GENELAB_PROFILE", raising=False)
    with maybe_profile() as step:
        assert step is None


def test_maybe_profile_kwarg_overrides_env_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    from genelab.rl import profiler as profiler_mod
    from genelab.rl.profiler import maybe_profile

    monkeypatch.setenv("GENELAB_PROFILE", "1")
    monkeypatch.setattr(profiler_mod, "is_main_process", lambda: True)
    with maybe_profile(enabled=False) as step:
        assert step is None


def test_prof_open_missing_dir_exits(tmp_path: Path) -> None:
    from genelab.cli import main

    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit) as excinfo:
        main(["prof", "open", str(missing)])
    assert "not found" in str(excinfo.value)


def _patch_picker(monkeypatch: pytest.MonkeyPatch, attr: str, value: str | None) -> None:
    """Replace a picker at both consumer sites (top-level import + late import)."""

    def fake(*_args: object, **_kwargs: object) -> str | None:
        return value

    monkeypatch.setattr(f"genelab.cli._interactive.{attr}", fake)
    # Patch every consumer module that imported the picker by name (each holds its
    # own binding to the original function). `pick_agent_kind` is consumed in
    # `cli._dispatch`; the task/name/override pickers in `cli` itself.
    for module in ("genelab.cli", "genelab.cli._dispatch"):
        if hasattr(__import__(module, fromlist=[attr]), attr):
            monkeypatch.setattr(f"{module}.{attr}", fake)


def test_play_unknown_task_falls_back_to_picker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_picker(monkeypatch, "pick_name_interactively", "External-Fake-Task-v0")
    main(["--import", "tests.fake_extension", "play", "not-a-task"])

    out = capsys.readouterr().out
    assert "played External-Fake-Task-v0" in out


def test_play_unknown_task_exits_without_picker_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default test stdin is non-TTY -> picker returns None -> original KeyError surfaces.
    _patch_picker(monkeypatch, "pick_name_interactively", None)
    with pytest.raises(SystemExit) as excinfo:
        main(["--import", "tests.fake_extension", "play", "not-a-task"])
    assert "not-a-task" in str(excinfo.value)


def test_info_unknown_name_falls_back_to_picker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_picker(monkeypatch, "pick_name_interactively", "External-Fake-Task-v0")
    main(["--import", "tests.fake_extension", "info", "definitely-not-a-name"])

    out = capsys.readouterr().out
    assert "External-Fake-Task-v0" in out
    assert "Task from a fake external package." in out


def test_play_invalid_agent_falls_back_to_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_play(task_id: str, **kwargs: object) -> None:
        captured["task_id"] = task_id
        captured.update(kwargs)

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(sys.modules, "genelab.rl", fake_rl)
    _patch_picker(monkeypatch, "pick_agent_kind", "zero")

    main(
        [
            "--import",
            "tests.fake_extension",
            "play",
            "External-Fake-Task-v0",
            "--agent",
            "bogus",
        ]
    )

    assert captured["agent"] == "zero"


def test_play_invalid_agent_without_picker_still_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_picker(monkeypatch, "pick_agent_kind", None)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--import",
                "tests.fake_extension",
                "play",
                "External-Fake-Task-v0",
                "--agent",
                "bogus",
            ]
        )
    assert "--agent" in str(excinfo.value)


def test_play_unknown_override_path_falls_back_to_picker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --env.simulation.step (typo for "steps"); picker corrects to the real path.
    _patch_picker(monkeypatch, "pick_override_path", "env.simulation.steps")

    main(
        [
            "--import",
            "tests.fake_extension",
            "play",
            "External-Fake-Task-v0",
            "--env.simulation.step",
            "3",
        ]
    )

    assert "played External-Fake-Task-v0 for 3 steps" in capsys.readouterr().out


def test_play_unknown_override_path_exits_without_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_picker(monkeypatch, "pick_override_path", None)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--import",
                "tests.fake_extension",
                "play",
                "External-Fake-Task-v0",
                "--env.simulation.step",
                "3",
            ]
        )
    assert "env.simulation.step" in str(excinfo.value)


def test_play_task_argument_accepts_either_ordering(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["--import", "tests.fake_extension", "play", "External-Fake-Task-v0", "--steps", "2"])
    first = capsys.readouterr().out
    main(["--import", "tests.fake_extension", "play", "--steps", "2", "External-Fake-Task-v0"])
    second = capsys.readouterr().out

    assert "played External-Fake-Task-v0 for 2 steps" in first
    assert "played External-Fake-Task-v0 for 2 steps" in second


def test_play_task_argument_preserves_dashed_runner_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_play(task_id: str, **kwargs: object) -> None:
        captured["task_id"] = task_id
        captured.update(kwargs)

    fake_rl = type("FakeRl", (), {"play_task": staticmethod(_fake_play), "AgentKind": str})
    monkeypatch.setitem(sys.modules, "genelab.rl", fake_rl)

    main(
        [
            "--import",
            "tests.fake_extension",
            "play",
            "External-Fake-Task-v0",
            "--num-envs",
            "8",
        ]
    )

    assert captured["task_id"] == "External-Fake-Task-v0"
    assert captured["num_envs"] == 8


def test_complete_task_names_returns_registered_ids() -> None:
    from genelab.cli._completion import complete_task_names

    load_extension_module("tests.fake_extension")

    assert "External-Fake-Task-v0" in complete_task_names("Ex")
    assert complete_task_names("zzz") == []


def test_complete_any_registry_name_returns_union() -> None:
    from genelab.cli._completion import complete_any_registry_name

    load_extension_module("tests.fake_extension")
    completions = complete_any_registry_name("")

    assert "External-Fake-Task-v0" in completions
    assert "fake-extension-env" in completions
    assert "fake-extension-robot" in completions


def test_completion_callbacks_swallow_extension_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genelab.cli import _completion

    def _boom() -> None:
        raise RuntimeError("entry point exploded")

    monkeypatch.setattr(_completion, "load_entrypoint_extensions", _boom)

    # Both callbacks must catch the failure and return any names already in the registry.
    # The exception itself must not propagate.
    _completion.complete_task_names("")
    _completion.complete_any_registry_name("")


def test_list_kind_argument_remains_enum_for_completion() -> None:
    """Regression: ``list KIND`` stays an Enum so Typer auto-completes its values."""
    from genelab.cli import _RegistryKindArg

    assert {member.value for member in _RegistryKindArg} == {"robots", "envs", "tasks"}
