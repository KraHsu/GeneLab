"""Optional interactive prompts for the GeneLab CLI."""

import sys

import questionary

from genelab.registry import TASKS


def pick_task_interactively() -> str | None:
    """Show a questionary picker for registered task ids.

    Returns ``None`` when running outside a TTY, when no tasks are registered, or when
    the user cancels the prompt. Callers should treat ``None`` as "no task chosen" and
    fall back to whatever non-interactive behavior they want (typically: exit with a
    "missing task id" message).
    """

    if not sys.stdin.isatty():
        return None
    names = TASKS.names()
    if not names:
        return None
    answer = questionary.select("Select a task:", choices=names).ask()
    if answer is None:
        return None
    return str(answer)
