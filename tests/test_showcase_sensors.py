"""Unit test for the ForceTorqueSensor wired into the Sensors-Showcase env.

Asserts the showcase env cfg registers a ``ForceTorqueSensorCfg`` and that, bound
against a Franka-style fake env (no Genesis), it resolves exactly the 7 arm joints
and produces ``data.force`` of shape ``(num_envs, 7)``. Mirrors the fake-env
pattern in ``tests/test_sensor.py``.
"""

import pytest

torch = pytest.importorskip("torch")

from genelab.sensor import ForceTorqueSensorCfg  # noqa: E402

# Franka actuated joints: 7 arm + 2 finger. The showcase regex ``^joint[1-7]$``
# must select the arm joints only.
_FRANKA_JOINTS = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)


class _FakeForceRobot:
    def __init__(self, force: torch.Tensor) -> None:
        self._force = force

    def get_dofs_force(self) -> torch.Tensor:
        return self._force


class _FakeArticulation:
    def __init__(self, actuated_dof_ids: torch.Tensor, joint_names: list[str]) -> None:
        self.actuated_dof_ids = actuated_dof_ids
        self.joint_names = joint_names


class _FakeFrankaEnv:
    """Fixed-base Franka: 9 actuated joints at global DoFs 0..8 (no floating base)."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.joint_names = list(_FRANKA_JOINTS)
        n = len(_FRANKA_JOINTS)
        # Distinct per-DoF values so a wrong slice would change the result.
        force = torch.arange(num_envs * n, dtype=torch.float32).reshape(num_envs, n)
        self.articulation = _FakeArticulation(
            torch.arange(n, dtype=torch.long), list(_FRANKA_JOINTS)
        )
        self.robot = _FakeForceRobot(force)
        self.sensors: dict[str, object] = {}


def _showcase_force_torque_cfg() -> ForceTorqueSensorCfg:
    from genelab_showcase.sensors.env_cfg import sensors_showcase_env_cfg

    fts = [
        s for s in sensors_showcase_env_cfg().scene.sensors if isinstance(s, ForceTorqueSensorCfg)
    ]
    assert len(fts) == 1, f"expected exactly one ForceTorqueSensorCfg, got {len(fts)}"
    return fts[0]


def test_sensors_showcase_registers_force_torque_sensor() -> None:
    cfg = _showcase_force_torque_cfg()
    assert cfg.name == "arm_joint_ft"
    assert cfg.joint_names_expr == r"^joint[1-7]$"


def test_sensors_showcase_force_torque_resolves_arm_joints() -> None:
    env = _FakeFrankaEnv(num_envs=2)
    sensor = _showcase_force_torque_cfg().build()
    sensor.bind(env)

    assert sensor.joint_names == [f"joint{i}" for i in range(1, 8)]
    force = sensor.data.force
    assert force.shape == (2, 7)
    # In this fake env actuated_dof_ids == arange(9), so the 7 arm joints map to
    # global DoFs 0..6 and the 2 finger joints (7, 8) must be excluded.
    assert torch.allclose(force, env.robot.get_dofs_force()[:, 0:7])
