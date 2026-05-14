"""Public Isaac Lab API surface backed by Genesis."""

from dataclasses import dataclass
from typing import Protocol

from genelab.configs import (
    InteractiveSceneCfg,
    ManagerBasedEnvCfg,
    SimulationCfg,
    TaskCfg,
    apply_overrides,
)
from genelab.entity import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    RobotState,
)
from genelab.mdp.noise import Gnoise, NoiseCfg, Unoise
from genelab.scene import InteractiveScene
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
from genelab.sensor import (
    ContactData,
    ContactSensor,
    ContactSensorCfg,
    GridPattern,
    RayCastData,
    RayCastSensor,
    RayCastSensorCfg,
    Sensor,
    SensorCfg,
    TerrainHeightSensor,
    TerrainHeightSensorCfg,
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
    "Articulation",
    "ArticulationCfg",
    "ContactData",
    "ContactSensor",
    "ContactSensorCfg",
    "GenesisBackendCfg",
    "Gnoise",
    "GridPattern",
    "InteractiveScene",
    "InteractiveSceneCfg",
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "NoiseCfg",
    "RayCastData",
    "RayCastSensor",
    "RayCastSensorCfg",
    "Registry",
    "RegistryEntry",
    "RigidObject",
    "RigidObjectCfg",
    "RobotState",
    "Sensor",
    "SensorCfg",
    "SimulationCfg",
    "TaskCfg",
    "TerrainHeightSensor",
    "TerrainHeightSensorCfg",
    "Unoise",
    "apply_overrides",
    "load_builtin_registries",
    "load_entrypoint_extensions",
    "load_extension_module",
]
