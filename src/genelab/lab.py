"""Public Isaac Lab API surface backed by Genesis."""

from dataclasses import dataclass
from typing import Protocol

from genelab.actuator import (
    ActuatorBase,
    ActuatorBaseCfg,
    DCMotorActuator,
    DCMotorActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitPDActuator,
    ImplicitPDActuatorCfg,
)
from genelab.asset_zoo import (
    AnymalCCfg,
    CartpoleCfg,
    FrankaPandaCfg,
    UnitreeG1Cfg,
    UnitreeGo1Cfg,
)
from genelab.bridges import (
    Bridge,
    BridgeCfg,
    KeyboardCommandBridge,
    KeyboardCommandBridgeCfg,
)
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
    CameraData,
    CameraSensor,
    CameraSensorCfg,
    ContactData,
    ContactSensor,
    ContactSensorCfg,
    FrameTransformerData,
    FrameTransformerSensor,
    FrameTransformerSensorCfg,
    GridPattern,
    HemispherePattern,
    IMUData,
    IMUSensor,
    IMUSensorCfg,
    RayCastData,
    RayCastSensor,
    RayCastSensorCfg,
    RingPattern,
    Sensor,
    SensorCfg,
    TargetFrameCfg,
    TerrainHeightSensor,
    TerrainHeightSensorCfg,
)
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SubTerrainCfg,
    TerrainGenerator,
    TerrainGeneratorCfg,
    TerrainImporter,
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
    "ActuatorBase",
    "ActuatorBaseCfg",
    "AnymalCCfg",
    "Articulation",
    "ArticulationCfg",
    "Bridge",
    "BridgeCfg",
    "CameraData",
    "CameraSensor",
    "CameraSensorCfg",
    "CartpoleCfg",
    "ContactData",
    "ContactSensor",
    "ContactSensorCfg",
    "DCMotorActuator",
    "DCMotorActuatorCfg",
    "FlatPatchCfg",
    "FrameTransformerData",
    "FrameTransformerSensor",
    "FrameTransformerSensorCfg",
    "FrankaPandaCfg",
    "GenesisBackendCfg",
    "Gnoise",
    "GridPattern",
    "HemispherePattern",
    "IMUData",
    "IMUSensor",
    "IMUSensorCfg",
    "IdealPDActuator",
    "IdealPDActuatorCfg",
    "ImplicitPDActuator",
    "ImplicitPDActuatorCfg",
    "InteractiveScene",
    "InteractiveSceneCfg",
    "KeyboardCommandBridge",
    "KeyboardCommandBridgeCfg",
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "NoiseCfg",
    "PyramidStairsCfg",
    "RandomRoughCfg",
    "RayCastData",
    "RayCastSensor",
    "RayCastSensorCfg",
    "Registry",
    "RegistryEntry",
    "RigidObject",
    "RigidObjectCfg",
    "RingPattern",
    "RobotState",
    "Sensor",
    "SensorCfg",
    "SimulationCfg",
    "SubTerrainCfg",
    "TargetFrameCfg",
    "TaskCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainHeightSensor",
    "TerrainHeightSensorCfg",
    "TerrainImporter",
    "UnitreeG1Cfg",
    "UnitreeGo1Cfg",
    "Unoise",
    "apply_overrides",
    "load_builtin_registries",
    "load_entrypoint_extensions",
    "load_extension_module",
]
