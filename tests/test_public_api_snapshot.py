"""Public-API ``__all__`` snapshot baseline.

Pins the two ``__all__``-guarded public facades — :mod:`genelab.lab` (the
Isaac-Lab-mirroring surface) and :mod:`genelab.extensions` (the plugin port)
— against checked-in expected name sets. Any add or removal must
update the expected set in this file in the same PR, making public-surface
changes explicit and reviewable.

This is the guardrail later changes rely on: clean-cut renames/removals
(e.g. dropping ``GenesisBackendCfg``, renaming the env protocol) update the
expected set here in the same change. The test guards against *accidental*
drift, not against deliberate, reviewed changes.

The expected sets are inline ``frozenset`` literals (not external snapshot
files) so drift surfaces as a readable code diff right next to the assertion.
"""

from __future__ import annotations

import genelab.extensions
import genelab.lab

# Frozen baseline of ``genelab.lab.__all__``. Update in the same PR that
# intentionally changes the public surface.
EXPECTED_LAB_ALL = frozenset(
    {
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
        "DiscreteObstaclesCfg",
        "FlatPatchCfg",
        "ForceTorqueData",
        "ForceTorqueSensor",
        "ForceTorqueSensorCfg",
        "FractalCfg",
        "FrameTransformerData",
        "FrameTransformerSensor",
        "FrameTransformerSensorCfg",
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
        "ManagerBasedEnvCfg",
        "ManagerBasedEnvProtocol",
        "MeshGridPattern",
        "MeshRayCastData",
        "MeshRayCastSensor",
        "MeshRayCastSensorCfg",
        "MeshSphericalPattern",
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
        "SteppingStonesCfg",
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
    }
)

# Frozen baseline of ``genelab.extensions.__all__`` (the plugin port).
EXPECTED_EXTENSIONS_ALL = frozenset(
    {
        "ENVS",
        "ROBOTS",
        "TASKS",
        "Backend",
        "Runnable",
        "register_backend",
        "register_env",
        "register_robot",
        "register_task",
    }
)


def test_lab_all_matches_snapshot() -> None:
    """``genelab.lab.__all__`` must match its frozen baseline exactly."""
    assert set(genelab.lab.__all__) == EXPECTED_LAB_ALL, (
        "genelab.lab public surface drifted from the snapshot. If this change "
        "is intentional, update EXPECTED_LAB_ALL in this file in the same PR."
    )


def test_extensions_all_matches_snapshot() -> None:
    """``genelab.extensions.__all__`` must match its frozen baseline exactly."""
    assert set(genelab.extensions.__all__) == EXPECTED_EXTENSIONS_ALL, (
        "genelab.extensions public surface drifted from the snapshot. If this "
        "change is intentional, update EXPECTED_EXTENSIONS_ALL in the same PR."
    )


def test_public_all_lists_have_no_duplicates() -> None:
    """A name listed twice in ``__all__`` is a copy-paste slip, not a surface."""
    assert len(genelab.lab.__all__) == len(set(genelab.lab.__all__))
    assert len(genelab.extensions.__all__) == len(set(genelab.extensions.__all__))


def test_public_names_are_importable() -> None:
    """Every advertised name must resolve on its module (no dangling ``__all__``)."""
    for name in genelab.lab.__all__:
        assert hasattr(genelab.lab, name), f"genelab.lab.__all__ lists missing {name!r}"
    for name in genelab.extensions.__all__:
        assert hasattr(genelab.extensions, name), (
            f"genelab.extensions.__all__ lists missing {name!r}"
        )
