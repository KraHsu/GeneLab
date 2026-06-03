"""CLI surface: registered hints, landing page, ``play --help`` docs, task
registration, ``info`` / ``list`` rendering, and shell-completion callbacks.

Split out of ``tests/test_cli.py`` by concern.
"""

import re

import pytest

from genelab import __doc__ as genelab_doc
from genelab.cli import main
from genelab.configs import ManagerBasedEnvCfg
from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module

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

    # The help spells some flags with hyphens (e.g. ``--max-steps``) and others with
    # underscores (``--num_envs``); the freeform parser accepts both. Normalize hyphens to
    # underscores so this coverage check is about *documentation presence*, not spelling.
    out = _strip_ansi(capsys.readouterr().out).replace("-", "_")
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
    # The registry singletons accumulate across the shared pytest session (other
    # tests call ``load_extension_module("genelab_examples.tasks")``), so this
    # absence check snapshots, clears, and restores them to assert against a *fresh*
    # registry — exactly what a real ``genelab --no-entry-points`` process sees.
    registries = (TASKS, ROBOTS, ENVS)
    saved = [dict(reg._entries) for reg in registries]
    for reg in registries:
        reg._entries.clear()
    try:
        main(["--no-entry-points", "list", "tasks"])
        assert "GeneLab-Rubiks-Play-v0" not in capsys.readouterr().out
    finally:
        for reg, entries in zip(registries, saved, strict=True):
            reg._entries.clear()
            reg._entries.update(entries)


def test_example_extension_registers_robots_envs_and_tasks() -> None:
    load_extension_module("genelab_examples.tasks")

    assert "rubiks-cube" in ROBOTS.names()
    assert "rubiks-play" in ENVS.names()
    assert "GeneLab-Rubiks-Play-v0" in TASKS.names()


def test_wuji_extension_registers_robots_envs_and_tasks() -> None:
    load_extension_module("genelab_wuji.tasks")

    assert "wuji-hand" in ROBOTS.names()
    assert "wuji-hand-playback" in ENVS.names()
    assert "GeneLab-Wuji-Hand-Playback-v0" in TASKS.names()


def test_list_tasks_shows_registered_bindings(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--import", "genelab_examples.tasks", "list", "tasks"])

    out = capsys.readouterr().out
    assert "GeneLab-Rubiks-Play-v0" in out
    assert "env=rubiks-play" in out
    assert "robot=rubiks-cube" in out


def test_package_positions_genelab_as_genesis_powered_lab_api() -> None:
    assert genelab_doc is not None
    assert "Isaac Lab API" in genelab_doc
    assert ManagerBasedEnvCfg.__name__ == "ManagerBasedEnvCfg"


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
