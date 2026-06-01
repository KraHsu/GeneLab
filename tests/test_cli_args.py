"""CLI argument handling: run-arg parsing, config overrides, ``_configured_task``
train/play step semantics, task-argument ordering, and profiler-flag parsing.

Split out of ``tests/test_cli.py`` by concern.
"""

import sys
from pathlib import Path

import pytest

from genelab.cli import main
from genelab.configs import apply_overrides
from genelab.registry import TASKS, load_extension_module


def test_config_overrides_update_nested_task_config() -> None:
    load_extension_module("genelab_wuji.tasks")
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


def test_cli_headless_flag_forces_vis_false() -> None:
    from genelab.cli import parse_run_args

    task_id, overrides = parse_run_args(
        ["GeneLab-Inverted-Pendulum-v0", "--agent", "trained", "--headless"]
    )

    assert task_id == "GeneLab-Inverted-Pendulum-v0"
    assert overrides["env.simulation.vis"] == "false"


@pytest.mark.parametrize("order", [["--vis", "--headless"], ["--headless", "--vis"]])
def test_cli_vis_and_headless_are_mutually_exclusive(order: list[str]) -> None:
    from genelab.cli import parse_run_args

    with pytest.raises(SystemExit, match="mutually exclusive"):
        parse_run_args(["GeneLab-Rubiks-Play-v0", *order])


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

    load_extension_module("genelab_wuji.tasks")
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

    load_extension_module("genelab_wuji.tasks")
    with pytest.raises(SystemExit) as excinfo:
        _configured_task(
            ["GeneLab-Wuji-Hand-Playback-v0", "--steps", "20", "--max_iterations", "100"],
            command="train",
        )
    assert "conflict" in str(excinfo.value)


def test_configured_task_play_mode_keeps_steps_as_env_simulation_steps() -> None:
    """Play mode behavior is unchanged: ``--steps`` lands on ``env.simulation.steps``."""
    from genelab.cli import _configured_task

    load_extension_module("genelab_wuji.tasks")
    task, runner_args, _ = _configured_task(
        ["GeneLab-Wuji-Hand-Playback-v0", "--steps", "5"],
        command="play",
    )

    assert task.cfg.env.simulation.steps == 5
    assert "max_iterations" not in runner_args


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
            "External-Fake-RL-Task-v0",
            "--num-envs",
            "8",
        ]
    )

    assert captured["task_id"] == "External-Fake-RL-Task-v0"
    assert captured["num_envs"] == 8


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
