"""CLI ``--help`` snapshot baseline (ROADMAP §9 Phase R0.1).

Locks the user-facing ``--help`` output of every Typer command so later
refactors — R3 (domain-owned parsing) and R4 (CLI decomposition) in
particular — can prove they are structural, not behavioural, via an
empty snapshot diff.

To intentionally change ``--help`` text (e.g. when adding a new flag),
regenerate the snapshots::

    UPDATE_SNAPSHOTS=1 pytest tests/test_cli_help_snapshots.py

The test invokes the CLI as ``python -m genelab.cli`` in a fresh
subprocess under a pinned environment so Rich's auto-detected terminal
width and colour mode cannot leak host-specific state into the
snapshots.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_SNAPSHOTS") == "1"

# (snapshot_name, argv_after_module). Mirrors ``genelab <cmd> --help``
# end-user invocations. Order matches the Typer panels rendered by the
# root ``--help`` (Utilities → Registry → Runtime → Project) plus root
# itself first and the ``project new`` subcommand last.
HELP_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("root", ("--help",)),
    ("cache", ("cache", "--help")),
    ("prof", ("prof", "--help")),
    ("list", ("list", "--help")),
    ("info", ("info", "--help")),
    ("play", ("play", "--help")),
    ("eval", ("eval", "--help")),
    ("export", ("export", "--help")),
    ("train", ("train", "--help")),
    ("project", ("project", "--help")),
    ("project_new", ("project", "new", "--help")),
)


def _run_help(args: tuple[str, ...]) -> str:
    """Run ``python -m genelab.cli <args>`` with a deterministic env.

    The pinned env is what makes the snapshots reproducible:

    - ``NO_COLOR=1`` disables ANSI colour escape codes.
    - ``TERM=dumb`` keeps Rich's terminal detection off the
      colour/styling code path.
    - ``COLUMNS=100`` pins Rich's panel width so wrapping is identical
      on every host.
    - ``PYTHONDONTWRITEBYTECODE=1`` avoids ``__pycache__`` churn in the
      working tree under repeated CI runs.
    """
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "TERM": "dumb",
        "COLUMNS": "100",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "genelab.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`python -m genelab.cli {' '.join(args)}` exited "
        f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc.stdout


@pytest.mark.parametrize(
    ("name", "argv"),
    HELP_COMMANDS,
    ids=[name for name, _ in HELP_COMMANDS],
)
def test_cli_help_snapshot(name: str, argv: tuple[str, ...]) -> None:
    """``genelab <cmd> --help`` must match its frozen baseline byte-for-byte."""
    snapshot_path = SNAPSHOT_DIR / f"help-{name}.txt"
    actual = _run_help(argv)

    if UPDATE_SNAPSHOTS:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual, encoding="utf-8")
        return

    assert snapshot_path.exists(), (
        f"missing snapshot {snapshot_path}; regenerate with "
        f"`UPDATE_SNAPSHOTS=1 pytest {Path(__file__).name}`"
    )
    expected = snapshot_path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{name} --help drifted from snapshot. If this change is "
        f"intentional, regenerate via "
        f"`UPDATE_SNAPSHOTS=1 pytest {Path(__file__).name}`."
    )
