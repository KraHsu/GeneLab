"""Unit tests for the bundled ``sample-arm`` xacro showcase."""

from pathlib import Path

import pytest

from genelab.asset_zoo import SampleArmCfg
from genelab.entity import ArticulationCfg


def test_sample_arm_cfg_returns_independent_instances() -> None:
    a = SampleArmCfg()
    b = SampleArmCfg()
    assert isinstance(a, ArticulationCfg)
    assert isinstance(b, ArticulationCfg)
    # Distinct dataclass instances so callers can mutate ``init_pos`` etc. without
    # leaking state across envs.
    assert a is not b


def test_sample_arm_cfg_uses_urdf_path_not_mjcf() -> None:
    cfg = SampleArmCfg()
    assert cfg.urdf_path != ""
    assert cfg.mjcf_path == ""


def test_sample_arm_xacro_file_is_packaged() -> None:
    cfg = SampleArmCfg()
    path = Path(cfg.urdf_path)
    assert path.exists()
    assert path.suffix == ".xacro"
    assert path.read_text().startswith("<?xml")


@pytest.mark.parametrize("link_length", [0.2, 0.3, 0.5])
def test_link_length_propagates_to_xacro_args(link_length: float) -> None:
    cfg = SampleArmCfg(link_length=link_length)
    assert cfg.xacro_args == {"link_length": f"{link_length}"}


def test_xacro_preprocess_substitutes_link_length() -> None:
    """End-to-end ``xacro:arg`` round-trip: pass an argument, parse the file with
    the same ``xacro`` package Genesis uses, and assert the geometry block carries
    the substituted size — proving the cfg's args actually flow through."""
    xacro = pytest.importorskip("xacro")

    cfg = SampleArmCfg(link_length=0.42)
    doc = xacro.process_file(cfg.urdf_path, mappings=dict(cfg.xacro_args))
    xml = doc.toxml()
    # ``${L}`` and ``${L * 0.6}`` should have been substituted with literal floats.
    assert "0.42" in xml  # the upper link uses ``${L}``
    assert "0.252" in xml  # the forearm uses ``${L * 0.6}``


def test_actuator_group_targets_only_revolute_joints() -> None:
    """Genesis's URDF morph adds an implicit ``root_joint`` for the floating base;
    the cfg's actuator regex must not match it (it would error at bind time when the
    actuator binder tries to discover gains for a 6-DoF free joint)."""
    cfg = SampleArmCfg()
    arm_actuator = cfg.actuators["arm"]
    assert "shoulder_yaw" in arm_actuator.target_names_expr
    assert "elbow_pitch" in arm_actuator.target_names_expr
    # ``.*`` would catch ``root_joint`` — we want the explicit list.
    assert ".*" not in arm_actuator.target_names_expr


def test_register_robot_registers_sample_arm() -> None:
    from genelab.registry import ROBOTS

    # Import side-effect from ``genelab.asset_zoo`` registers every robot.
    import genelab.asset_zoo  # noqa: F401

    assert "sample-arm" in ROBOTS.names()
    entry = ROBOTS.entry("sample-arm")
    assert "xacro" in entry.description.lower()
