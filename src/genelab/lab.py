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
    MlpResidualActuator,
    MlpResidualActuatorCfg,
)
from genelab.bridges import (
    Bridge,
    BridgeCfg,
    KeyboardTwistBridge,
    KeyboardTwistBridgeCfg,
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
from genelab.mdp.noise import BiasDrift, CorrelatedNoise, Gnoise, NoiseCfg, ScaledNoise, Unoise
from genelab.scene import InteractiveScene
from genelab.registry import (
    ENVS,
    ROBOTS,
    TASKS,
    Registry,
    RegistryEntry,
    load_bundled_asset_zoo,
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
    ForceTorqueData,
    ForceTorqueSensor,
    ForceTorqueSensorCfg,
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
    "Articulation",
    "ArticulationCfg",
    "BiasDrift",
    "Bridge",
    "BridgeCfg",
    "CameraData",
    "CameraSensor",
    "CameraSensorCfg",
    "ContactData",
    "ContactSensor",
    "ContactSensorCfg",
    "CorrelatedNoise",
    "DCMotorActuator",
    "DCMotorActuatorCfg",
    "FlatPatchCfg",
    "ForceTorqueData",
    "ForceTorqueSensor",
    "ForceTorqueSensorCfg",
    "FrameTransformerData",
    "FrameTransformerSensor",
    "FrameTransformerSensorCfg",
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
    "KeyboardTwistBridge",
    "KeyboardTwistBridgeCfg",
    "ManagerBasedEnv",
    "ManagerBasedEnvCfg",
    "MlpResidualActuator",
    "MlpResidualActuatorCfg",
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
    "ScaledNoise",
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
    "Unoise",
    "apply_overrides",
    "load_bundled_asset_zoo",
    "load_entrypoint_extensions",
    "load_extension_module",
]
