"""Tests for ``genelab.entity.root_velocity`` — the #242 regression seam.

Genesis's ``RigidEntity`` exposes ``get_vel`` / ``get_ang`` but no ``set_vel`` /
``set_ang`` (those live on FEM / tool entities only), so every root-velocity write
must route through ``set_dofs_velocity`` on the base free joint's 6 DoFs. The fakes
here mirror that Genesis 1.2 API surface — deliberately *without* ``set_vel`` /
``set_ang`` — so a regression back to the old ``getattr`` idiom fails these tests
instead of silently no-opping.
"""

from typing import Any

import torch

from genelab.entity._articulation_writer import ArticulationWriter
from genelab.entity.root_velocity import base_dof_indices, write_root_velocity
from genelab.mdp.events import push_by_setting_velocity, reset_root_state_uniform


class _FakeJoint:
    def __init__(self, n_dofs: int, dofs_idx_local: list[int]) -> None:
        self.n_dofs = n_dofs
        self.dofs_idx_local = dofs_idx_local


class _FakeRigidEntity:
    """Genesis-1.2-shaped rigid handle: free joint + hinges, no ``set_vel`` / ``set_ang``."""

    def __init__(self, free_base: bool = True) -> None:
        hinge_offset = 6 if free_base else 0
        self.joints = ([_FakeJoint(6, [0, 1, 2, 3, 4, 5])] if free_base else []) + [
            _FakeJoint(1, [hinge_offset]),
            _FakeJoint(1, [hinge_offset + 1]),
        ]
        self.dofs_velocity_calls: list[tuple[torch.Tensor, list[int], Any]] = []
        self.pos_calls: list[torch.Tensor] = []
        self.quat_calls: list[torch.Tensor] = []

    def set_dofs_velocity(
        self, velocity: torch.Tensor, dofs_idx_local: Any = None, envs_idx: Any = None
    ) -> None:
        self.dofs_velocity_calls.append((velocity.clone(), list(dofs_idx_local), envs_idx))

    def set_pos(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self.pos_calls.append(value.clone())

    def set_quat(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self.quat_calls.append(value.clone())


class _FakeSoftEntity:
    """FEM / tool-entity shape: direct ``set_vel`` / ``set_ang``, no DoF API."""

    def __init__(self) -> None:
        self.vel: torch.Tensor | None = None
        self.ang: torch.Tensor | None = None

    def set_vel(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self.vel = value.clone()

    def set_ang(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self.ang = value.clone()


# ------------------------------------------------------------- base_dof_indices


def test_base_dof_indices_free_base() -> None:
    assert base_dof_indices(_FakeRigidEntity()) == [0, 1, 2, 3, 4, 5]


def test_base_dof_indices_fixed_base_is_none() -> None:
    assert base_dof_indices(_FakeRigidEntity(free_base=False)) is None


def test_base_dof_indices_no_joints_is_none() -> None:
    assert base_dof_indices(object()) is None


# ------------------------------------------------------------ write_root_velocity


def test_write_root_velocity_routes_through_free_joint_dofs() -> None:
    handle = _FakeRigidEntity()
    lin = torch.tensor([[1.0, 2.0, 3.0]])
    ang = torch.tensor([[0.1, 0.2, 0.3]])
    env_ids = torch.tensor([4])
    assert write_root_velocity(handle, lin, ang, env_ids) is True
    (velocity, idx, envs_idx) = handle.dofs_velocity_calls[0]
    assert torch.equal(velocity, torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]))
    assert idx == [0, 1, 2, 3, 4, 5]
    assert torch.equal(envs_idx, env_ids)


def test_write_root_velocity_fixed_base_writes_nothing() -> None:
    handle = _FakeRigidEntity(free_base=False)
    zeros = torch.zeros(1, 3)
    assert write_root_velocity(handle, zeros, zeros, torch.tensor([0])) is False
    assert handle.dofs_velocity_calls == []


def test_write_root_velocity_direct_setter_fallback() -> None:
    handle = _FakeSoftEntity()
    lin = torch.tensor([[1.0, 0.0, 0.0]])
    ang = torch.tensor([[0.0, 0.0, 2.0]])
    assert write_root_velocity(handle, lin, ang, torch.tensor([0])) is True
    assert handle.vel is not None and torch.equal(handle.vel, lin)
    assert handle.ang is not None and torch.equal(handle.ang, ang)


# ------------------------------------------------------------------ event terms


class _FakeEntityCfg:
    init_pos = (0.0, 0.0, 0.5)
    init_quat = (1.0, 0.0, 0.0, 0.0)


class _FakeArticulation:
    cfg = _FakeEntityCfg()


class _FakeEnv:
    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.articulation = _FakeArticulation()
        self.device = "cpu"


def test_push_by_setting_velocity_writes_sampled_base_velocity() -> None:
    """#242: the push must land in the free-joint DoFs of a handle without set_vel/set_ang."""
    torch.manual_seed(0)
    robot = _FakeRigidEntity()
    env = _FakeEnv(robot)
    env_ids = torch.arange(32)
    push_by_setting_velocity(env, env_ids, velocity_range={"x": (0.5, 1.0), "yaw": (-0.2, -0.1)})
    assert len(robot.dofs_velocity_calls) == 1
    velocity, idx, envs_idx = robot.dofs_velocity_calls[0]
    assert velocity.shape == (32, 6)
    assert idx == [0, 1, 2, 3, 4, 5]
    assert torch.equal(envs_idx, env_ids)
    assert torch.all(velocity[:, 0] >= 0.5) and torch.all(velocity[:, 0] <= 1.0)
    assert torch.all(velocity[:, 5] >= -0.2) and torch.all(velocity[:, 5] <= -0.1)
    # Unspecified axes overwrite to zero (Isaac Lab parity).
    assert torch.equal(velocity[:, 1:5], torch.zeros(32, 4))


def test_push_by_setting_velocity_empty_env_ids_is_noop() -> None:
    robot = _FakeRigidEntity()
    push_by_setting_velocity(_FakeEnv(robot), torch.tensor([], dtype=torch.long))
    assert robot.dofs_velocity_calls == []


def test_reset_root_state_uniform_writes_base_velocity() -> None:
    torch.manual_seed(0)
    robot = _FakeRigidEntity()
    env = _FakeEnv(robot)
    env_ids = torch.arange(16)
    reset_root_state_uniform(env, env_ids, velocity_range={"z": (0.1, 0.2)})
    assert len(robot.pos_calls) == 1  # pose write still happens
    assert len(robot.dofs_velocity_calls) == 1
    velocity, idx, _ = robot.dofs_velocity_calls[0]
    assert idx == [0, 1, 2, 3, 4, 5]
    assert torch.all(velocity[:, 2] >= 0.1) and torch.all(velocity[:, 2] <= 0.2)


# ------------------------------------------------------------------ writer seam


def test_writer_write_root_state_routes_velocity_via_free_joint() -> None:
    robot = _FakeRigidEntity()
    writer = ArticulationWriter(
        robot,
        actuated_dof_idx=torch.tensor([6, 7]),
        default_joint_pos=torch.zeros(2),
        joint_pos_target=torch.zeros(1, 2),
        actuators={},
        device="cpu",
    )
    env_ids = torch.tensor([0])
    writer.write_root_state(
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[0.1, 0.2, 0.3]]),
        env_ids,
    )
    assert len(robot.pos_calls) == 1 and len(robot.quat_calls) == 1
    velocity, idx, envs_idx = robot.dofs_velocity_calls[0]
    assert torch.equal(velocity, torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]]))
    assert idx == [0, 1, 2, 3, 4, 5]
    assert torch.equal(envs_idx, env_ids)


# ------------------------------------------------------------- real Genesis seam


def test_write_root_velocity_round_trips_on_real_genesis(genesis_runtime: Any) -> None:
    """The definitive #242 check: write through the free joint, read back via get_vel/get_ang."""
    del genesis_runtime  # fixture only guards EGL/runtime availability
    from genelab.configs import InteractiveSceneCfg, SimulationCfg
    from genelab.entity import RigidObjectCfg
    from genelab.scene import InteractiveScene

    sim_cfg = SimulationCfg(num_envs=2, dt=0.01, substeps=1, vis=False, gpu=False)
    scene_cfg = InteractiveSceneCfg(
        env_spacing=(2.0, 2.0),
        # ``fixed=False``: only a floating box has the free joint this write targets.
        entities={
            "box": RigidObjectCfg(
                morph="box", size=(0.1, 0.1, 0.1), init_pos=(0, 0, 1.0), fixed=False
            )
        },
    )
    scene = InteractiveScene(sim_cfg, scene_cfg, device_hint="cpu")
    try:
        scene.build()
        handle = scene.rigid_objects["box"].gs_handle
        lin = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, 0.0]])
        ang = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.5]])
        assert write_root_velocity(handle, lin, ang, torch.tensor([0, 1])) is True
        got_lin = torch.as_tensor(handle.get_vel()).cpu()
        got_ang = torch.as_tensor(handle.get_ang()).cpu()
        assert torch.allclose(got_lin, lin, atol=1e-5)
        assert torch.allclose(got_ang, ang, atol=1e-5)
    finally:
        scene.close()
