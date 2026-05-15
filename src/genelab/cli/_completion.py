"""Shell-completion callbacks for the GeneLab CLI.

Callbacks run inside a shell subprocess where stderr noise pollutes the
user's prompt. They must therefore be fast and silent: registry side effects
are limited to the entry-point group, and any exception during loading
collapses to an empty completion list rather than propagating.
"""

from genelab.registry import ENVS, ROBOTS, TASKS, load_entrypoint_extensions


def _safe_load() -> None:
    try:
        load_entrypoint_extensions()
    except Exception:
        pass


def complete_task_names(incomplete: str) -> list[str]:
    _safe_load()
    return [name for name in TASKS.names() if name.startswith(incomplete)]


def complete_any_registry_name(incomplete: str) -> list[str]:
    _safe_load()
    names = sorted({*TASKS.names(), *ENVS.names(), *ROBOTS.names()})
    return [name for name in names if name.startswith(incomplete)]
