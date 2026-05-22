"""CLI task routing: agent-kind validation/dispatch, extension + entry-point
loading, ``project new`` scaffolding, ``train`` reporting, and the interactive
picker fallback for unknown task / agent / override-path inputs.

Split out of ``tests/test_cli.py`` by concern.
"""

import sys
from pathlib import Path

import pytest

from genelab.cli import main
from genelab.registry import TASKS, load_entrypoint_extensions


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
