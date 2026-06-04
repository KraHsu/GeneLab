"""CLI ``--help`` snapshot baseline.

Locks the user-facing ``--help`` output of every Typer command so later
refactors — R3 (domain-owned parsing) and R4 (CLI decomposition) in
particular — can prove they are structural, not behavioural, via an
empty snapshot diff.

To intentionally change ``--help`` text (e.g. when adding a new flag),
regenerate the snapshots::

    UPDATE_SNAPSHOTS=1 pytest tests/test_cli_help_snapshots.py

The test invokes the CLI in a fresh subprocess through a small wrapper
that pins two ``typer.rich_utils`` module constants before importing
``genelab.cli``:

- ``MAX_WIDTH = 100``: makes Typer construct its Rich console with an
  explicit width, bypassing every auto-detection branch.  Without this
  the width drifts between hosts because Rich's
  ``shutil.get_terminal_size()`` / ``is_dumb_terminal`` / TTY ioctl
  probes pick different fallbacks depending on environment.
- ``FORCE_TERMINAL = False``: forces Rich into non-terminal mode so it
  never emits ANSI style escapes (bold / dim / colour).  ``NO_COLOR=1``
  alone is insufficient because some CI environments (GitHub Actions
  among them) set ``FORCE_COLOR=1`` which Rich honours over
  ``NO_COLOR``.

Together these pins make the captured snapshot byte-identical across
local workstations, headless CI runners, and any future host.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_SNAPSHOTS") == "1"

# Width to pin Typer's Rich help console at.  100 cols renders without
# wrapping most option descriptions; any value works as long as it is
# the same here and in CI.
HELP_WIDTH = 100

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
    ("benchmark", ("benchmark", "--help")),
    ("export", ("export", "--help")),
    ("train", ("train", "--help")),
    ("project", ("project", "--help")),
    ("project_new", ("project", "new", "--help")),
)


# Wrapper executed by the subprocess: pin Typer's Rich-console width
# and force non-terminal mode (no ANSI styles), set ``sys.argv`` to the
# desired ``genelab <cmd> --help`` invocation, then hand off to
# ``genelab.cli.main`` exactly the way the console-script does.
_WRAPPER = (
    "import sys\n"
    "import typer.rich_utils as _r\n"
    "_r.MAX_WIDTH = {width}\n"
    "_r.FORCE_TERMINAL = False\n"
    "sys.argv = {argv!r}\n"
    "from genelab.cli import main\n"
    "main()\n"
)


def _run_help(args: tuple[str, ...]) -> str:
    """Run ``genelab <args>`` with a width-pinned Rich help console."""
    wrapper = _WRAPPER.format(width=HELP_WIDTH, argv=["genelab", *args])
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`genelab {' '.join(args)}` exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
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
