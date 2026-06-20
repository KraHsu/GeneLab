"""Unit tests for the unified ``wuji`` console-script dispatcher.

Covers verb routing (including the shared hand_utils check/home prefix), the
help/usage/unknown-command exit codes, and that each command lazily imports its
target module (so e.g. `wuji play` does not require the camera SDK).
"""

from __future__ import annotations

import sys
import types

import pytest

from genelab_wuji import cli


def test_commands_map_to_real_modules():
    # Every advertised verb maps to a deploy.scripts module + a help line.
    for name, (module_name, prefix) in cli._COMMANDS.items():
        assert module_name.startswith("genelab_wuji.deploy.scripts.")
        assert isinstance(prefix, tuple)
        assert name in cli._HELP


@pytest.mark.parametrize(
    "argv,code",
    [
        (["--help"], 0),
        (["help"], 0),
        ([], 2),
        (["bogus"], 2),
    ],
)
def test_top_level_exit_codes(argv, code):
    assert cli.main(argv) == code


def test_usage_lists_every_command(capsys):
    cli.main(["--help"])
    out = capsys.readouterr().out
    for name in cli._COMMANDS:
        assert name in out


def _install_stub(monkeypatch, target_module: str):
    """Replace ``importlib.import_module`` so the dispatched module is a stub
    that records the argv it was handed instead of importing real hardware code."""
    seen: dict[str, list[str]] = {}
    stub = types.ModuleType(target_module)

    def _main(argv):
        seen["argv"] = list(argv)
        return 0

    stub.main = _main  # type: ignore[attr-defined]
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: stub)
    return seen


def test_play_routes_args_through(monkeypatch):
    seen = _install_stub(monkeypatch, "genelab_wuji.deploy.scripts.play_real")
    rc = cli.main(["play", "--ckpt", "p.onnx", "--mock"])
    assert rc == 0
    assert seen["argv"] == ["--ckpt", "p.onnx", "--mock"]


def test_check_prepends_subcommand(monkeypatch):
    # `wuji check` / `wuji home` share hand_utils.main via a prepended verb.
    seen = _install_stub(monkeypatch, "genelab_wuji.deploy.scripts.hand_utils")
    cli.main(["check", "--foo"])
    assert seen["argv"] == ["check", "--foo"]
    cli.main(["home"])
    assert seen["argv"] == ["home"]


def test_main_reads_sys_argv_when_argv_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["wuji", "--help"])
    assert cli.main() == 0
