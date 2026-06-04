"""Public Isaac Lab API surface backed by Genesis.

The bulk of this surface is re-exported **lazily**: rather than eagerly importing
every domain symbol (which made this file co-change with every sensor/actuator
addition and dragged ``torch`` in at ``import genelab.lab`` time), names are
resolved on first access by :func:`__getattr__`, which searches the source
packages in :data:`_SOURCE_MODULES` and caches the result. ``__all__`` stays an
explicit literal — it is the *blessed* public surface (guarded by
``tests/test_public_api_snapshot.py`` and honoured by ``from genelab.lab import
*``), while ``__getattr__`` makes any public name in a source package reachable
without editing this file (ADR-0017).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Type-only re-export. At runtime these names are resolved lazily by
    # ``__getattr__`` (which searches ``_SOURCE_MODULES``); this block exists solely
    # so type-checkers and IDEs see the public surface without the eager import cost.
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
    from genelab.materials import (
        BsdfCfg,
        ColorTextureCfg,
        EmissionCfg,
        FemClothCfg,
        FemElasticCfg,
        FemMuscleCfg,
        FemOptionsCfg,
        GlassCfg,
        HybridMaterialCfg,
        ImageTextureCfg,
        KinematicMaterialCfg,
        MaterialCfg,
        MetalCfg,
        MpmElasticCfg,
        MpmElastoPlasticCfg,
        MpmLiquidCfg,
        MpmMuscleCfg,
        MpmOptionsCfg,
        MpmSandCfg,
        MpmSnowCfg,
        PbdClothCfg,
        PbdElasticCfg,
        PbdLiquidCfg,
        PbdOptionsCfg,
        PbdParticleCfg,
        PlasticCfg,
        RigidMaterialCfg,
        SfOptionsCfg,
        SfSmokeCfg,
        SolverOptionsCfg,
        SphLiquidCfg,
        SphOptionsCfg,
        SurfaceCfg,
        TextureCfg,
        ToolMaterialCfg,
        ToolOptionsCfg,
    )
    from genelab.mdp.noise import BiasDrift, CorrelatedNoise, Gnoise, NoiseCfg, ScaledNoise, Unoise
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
    from genelab.scene import InteractiveScene
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
        MeshGridPattern,
        MeshRayCastData,
        MeshRayCastSensor,
        MeshRayCastSensorCfg,
        MeshSphericalPattern,
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
        DiscreteObstaclesCfg,
        FlatPatchCfg,
        FractalCfg,
        PyramidStairsCfg,
        RandomRoughCfg,
        SteppingStonesCfg,
        SubTerrainCfg,
        TerrainGenerator,
        TerrainGeneratorCfg,
        TerrainImporter,
    )


class ManagerBasedEnvProtocol(Protocol):
    """Minimal runtime protocol for registered manager-based environments."""

    cfg: ManagerBasedEnvCfg

    def play(self) -> None:
        """Run the environment with its current configuration."""
        ...


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
    "BsdfCfg",
    "CameraData",
    "CameraSensor",
    "CameraSensorCfg",
    "ColorTextureCfg",
    "ContactData",
    "ContactSensor",
    "ContactSensorCfg",
    "CorrelatedNoise",
    "DCMotorActuator",
    "DCMotorActuatorCfg",
    "DiscreteObstaclesCfg",
    "EmissionCfg",
    "FemClothCfg",
    "FemElasticCfg",
    "FemMuscleCfg",
    "FemOptionsCfg",
    "FlatPatchCfg",
    "ForceTorqueData",
    "ForceTorqueSensor",
    "ForceTorqueSensorCfg",
    "FractalCfg",
    "FrameTransformerData",
    "FrameTransformerSensor",
    "FrameTransformerSensorCfg",
    "GlassCfg",
    "Gnoise",
    "GridPattern",
    "HemispherePattern",
    "HybridMaterialCfg",
    "IMUData",
    "IMUSensor",
    "IMUSensorCfg",
    "IdealPDActuator",
    "IdealPDActuatorCfg",
    "ImageTextureCfg",
    "ImplicitPDActuator",
    "ImplicitPDActuatorCfg",
    "InteractiveScene",
    "InteractiveSceneCfg",
    "KeyboardTwistBridge",
    "KeyboardTwistBridgeCfg",
    "KinematicMaterialCfg",
    "ManagerBasedEnvCfg",
    "ManagerBasedEnvProtocol",
    "MaterialCfg",
    "MeshGridPattern",
    "MeshRayCastData",
    "MeshRayCastSensor",
    "MeshRayCastSensorCfg",
    "MeshSphericalPattern",
    "MetalCfg",
    "MlpResidualActuator",
    "MlpResidualActuatorCfg",
    "MpmElasticCfg",
    "MpmElastoPlasticCfg",
    "MpmLiquidCfg",
    "MpmMuscleCfg",
    "MpmOptionsCfg",
    "MpmSandCfg",
    "MpmSnowCfg",
    "NoiseCfg",
    "PbdClothCfg",
    "PbdElasticCfg",
    "PbdLiquidCfg",
    "PbdOptionsCfg",
    "PbdParticleCfg",
    "PlasticCfg",
    "PyramidStairsCfg",
    "RandomRoughCfg",
    "RayCastData",
    "RayCastSensor",
    "RayCastSensorCfg",
    "Registry",
    "RegistryEntry",
    "RigidMaterialCfg",
    "RigidObject",
    "RigidObjectCfg",
    "RingPattern",
    "RobotState",
    "ScaledNoise",
    "Sensor",
    "SensorCfg",
    "SfOptionsCfg",
    "SfSmokeCfg",
    "SimulationCfg",
    "SolverOptionsCfg",
    "SphLiquidCfg",
    "SphOptionsCfg",
    "SteppingStonesCfg",
    "SubTerrainCfg",
    "SurfaceCfg",
    "TargetFrameCfg",
    "TaskCfg",
    "TerrainGenerator",
    "TerrainGeneratorCfg",
    "TerrainHeightSensor",
    "TerrainHeightSensorCfg",
    "TerrainImporter",
    "TextureCfg",
    "ToolMaterialCfg",
    "ToolOptionsCfg",
    "Unoise",
    "apply_overrides",
    "load_bundled_asset_zoo",
    "load_entrypoint_extensions",
    "load_extension_module",
]

# Source packages searched, in order, by ``__getattr__``. A public name defined in
# any of these is reachable as ``genelab.lab.<name>`` without editing this file;
# ``__all__`` above is the *blessed* subset (snapshot-guarded, used by ``import *``).
_SOURCE_MODULES = (
    "genelab.actuator",
    "genelab.bridges",
    "genelab.configs",
    "genelab.entity",
    "genelab.materials",
    "genelab.mdp.noise",
    "genelab.registry",
    "genelab.scene",
    "genelab.sensor",
    "genelab.terrains",
)

_MISSING = object()


def __getattr__(name: str) -> object:
    """Resolve a public domain symbol lazily from the source packages.

    First access imports the owning package, caches the value on this module (so
    later access is a normal global lookup), and returns it. The locally-defined
    ``ManagerBasedEnvProtocol`` never reaches here.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for module_path in _SOURCE_MODULES:
        value = getattr(importlib.import_module(module_path), name, _MISSING)
        if value is not _MISSING:
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
