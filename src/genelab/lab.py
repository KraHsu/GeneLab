"""Public Isaac Lab API surface backed by Genesis."""

from dataclasses import dataclass
from typing import Protocol

from genelab.configs import ManagerBasedEnvCfg, TaskCfg, apply_overrides
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    Registry,
    RegistryEntry,
    load_builtin_registries,
    load_entrypoint_extensions,
    load_extension_module,
)


class ManagerBasedEnv(Protocol):
    """Minimal runtime protocol for registered manager-based environments."""

    cfg: ManagerBasedEnvCfg

    def play(self) -> None:
        """Run the environment with its current configuration."""
        ...


@dataclass(frozen=True)
class GenesisBackendCfg:
    """Small descriptor for the Genesis backend used by GeneLab runners."""

    name: str = "genesis"
    precision: str = "32"


__all__ = [
    "ENVS",
    "ROBOTS",
    "TASKS",
    "GenesisBackendCfg",
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "Registry",
    "RegistryEntry",
    "TaskCfg",
    "apply_overrides",
    "load_builtin_registries",
    "load_entrypoint_extensions",
    "load_extension_module",
]
