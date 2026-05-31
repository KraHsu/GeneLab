"""GeneLab: an Isaac Lab API for RL and robotics research powered by Genesis."""

from typing import TYPE_CHECKING

__version__ = "0.3.1"

__all__ = [
    "ManagerBasedEnvCfg",
    "ManagerBasedEnvProtocol",
    "TaskCfg",
    "__version__",
]

if TYPE_CHECKING:
    from genelab.lab import ManagerBasedEnvCfg, ManagerBasedEnvProtocol, TaskCfg


# Re-export the lab dataclasses lazily so that `import genelab` (and therefore the
# CLI entry point) does not eagerly drag in torch via `genelab.lab → genelab.actuator`.
# First access to any of these resolves and caches the real attribute on this module.
_LAZY_LAB_EXPORTS = frozenset({"ManagerBasedEnvCfg", "ManagerBasedEnvProtocol", "TaskCfg"})


def __getattr__(name: str) -> object:
    if name in _LAZY_LAB_EXPORTS:
        from genelab import lab

        value = getattr(lab, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_LAB_EXPORTS)
