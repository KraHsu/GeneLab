"""Optional interactive prompts for the GeneLab CLI.

All pickers are TTY-guarded and return ``str | None``. ``None`` means
"non-TTY, no candidates, or user cancelled" — callers should treat that as
"no choice made" and re-raise whatever non-interactive error they would have
raised otherwise. This keeps CI / pipes / pytest behavior identical to a
build without these helpers.
"""

import sys
from collections.abc import Iterable, Sequence

import questionary

from genelab.registry import TASKS


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def pick_name_interactively(names: Sequence[str], label: str) -> str | None:
    """Prompt the user to pick one of ``names`` and return it."""

    if not _tty() or not names:
        return None
    answer = questionary.select(label, choices=list(names)).ask()
    if answer is None:
        return None
    return str(answer)


def pick_task_interactively() -> str | None:
    """Show a picker for registered task ids."""

    return pick_name_interactively(TASKS.names(), "Select a task:")


def pick_agent_kind() -> str | None:
    """Prompt the user to pick a ``--agent`` value."""

    return pick_name_interactively(("zero", "random", "trained"), "Select agent kind:")


def pick_override_path(typed: str, candidates: Iterable[str]) -> str | None:
    """Offer matching override paths when the user typed an unknown one.

    Filtering keeps paths whose head segment matches the typed head, or whose
    tail equals the typed tail. Returning ``None`` (no matches, or user
    cancelled) lets the caller re-raise the original ``ValueError``.
    """

    head, _, tail = typed.partition(".")
    candidate_list = list(candidates)
    matches = [
        path
        for path in candidate_list
        if path == typed or head in path.split(".") or (tail != "" and path.endswith(tail))
    ]
    if not matches:
        return None
    return pick_name_interactively(matches, f"No such override path {typed!r}. Pick one:")
