"""``Articulation.refresh`` must update :class:`RobotState` in place.

Rebinding the underlying tensors on every step is what the velocity-tracking hot loop
used to do, and the resulting allocator churn over a multi-hour training run is one of
the leading suspects for the progressive slowdown reported on the 8×H200 box. This test
locks in the contract: the buffers stay the same Python objects (same ``data_ptr``)
across consecutive ``refresh`` calls, even as their contents change.
"""

from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from genelab.entity.articulation import Articulation, ArticulationCfg  # noqa: E402


class _FakeJoint:
    def __init__(self, name: str, dof_start: int) -> None:
        self.name = name
        self.n_dofs = 1
        self.dofs_idx_local = (dof_start,)


class _FakeLink:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRefreshHandle:
    """Implements just the getters ``refresh`` reads. Holds simple counters so each call
    returns slightly different values — the test asserts both ``data_ptr`` stability
    and that the new values land inside the same buffer."""

    def __init__(self, joint_names: list[str], num_envs: int) -> None:
        self.joints = [_FakeJoint(name, i) for i, name in enumerate(joint_names)]
        self.links = [_FakeLink("base"), _FakeLink("tip")]
        self.n_dofs = len(joint_names)
        self._num_envs = num_envs
        self._tick = 0.0

    def _bump(self) -> float:
        self._tick += 1.0
        return self._tick

    def set_dofs_kp(self, *_args: Any, **_kwargs: Any) -> None: ...
    def set_dofs_kv(self, *_args: Any, **_kwargs: Any) -> None: ...
    def set_dofs_force_range(self, *_args: Any, **_kwargs: Any) -> None: ...

    def get_pos(self) -> torch.Tensor:
        return torch.full((self._num_envs, 3), self._bump())

    def get_quat(self) -> torch.Tensor:
        # Identity-ish quat — values don't need to be physically meaningful for this test.
        q = torch.zeros(self._num_envs, 4)
        q[:, 0] = 1.0
        return q

    def get_vel(self) -> torch.Tensor:
        return torch.full((self._num_envs, 3), self._bump())

    def get_ang(self) -> torch.Tensor:
        return torch.full((self._num_envs, 3), self._bump())

    def get_dofs_position(self) -> torch.Tensor:
        return torch.full((self._num_envs, self.n_dofs), self._bump())

    def get_dofs_velocity(self) -> torch.Tensor:
        return torch.full((self._num_envs, self.n_dofs), self._bump())

    def get_links_pos(self) -> torch.Tensor:
        return torch.full((self._num_envs, len(self.links), 3), self._bump())

    def get_links_quat(self) -> torch.Tensor:
        q = torch.zeros(self._num_envs, len(self.links), 4)
        q[..., 0] = 1.0
        return q

    def get_links_vel(self) -> torch.Tensor:
        return torch.full((self._num_envs, len(self.links), 3), self._bump())

    def get_links_ang(self) -> torch.Tensor:
        return torch.full((self._num_envs, len(self.links), 3), self._bump())


def _build_articulation_with_refresh_handle(num_envs: int = 4) -> Articulation:
    from genelab.actuator import ImplicitPDActuatorCfg

    joint_names = ["j0", "j1"]
    cfg = ArticulationCfg(
        mjcf_path="/dev/null",
        default_joint_pos={name: 0.0 for name in joint_names},
        actuators={
            "all": ImplicitPDActuatorCfg(
                target_names_expr=(".*",), stiffness=1.0, damping=0.1
            ),
        },
    )
    art = Articulation(cfg, name="test")
    art._gs_handle = cast(Any, _FakeRefreshHandle(joint_names, num_envs))  # noqa: SLF001
    art.bind(num_envs=num_envs, device="cpu")
    return art


def test_refresh_updates_robot_state_in_place() -> None:
    art = _build_articulation_with_refresh_handle()
    rs = art.data

    ptrs_before = {
        "root_pos": rs.root_pos.data_ptr(),
        "root_quat": rs.root_quat.data_ptr(),
        "root_lin_vel_w": rs.root_lin_vel_w.data_ptr(),
        "root_ang_vel_w": rs.root_ang_vel_w.data_ptr(),
        "joint_pos": rs.joint_pos.data_ptr(),
        "joint_vel": rs.joint_vel.data_ptr(),
        "link_pos": rs.link_pos.data_ptr(),
        "link_quat_w": rs.link_quat_w.data_ptr(),
        "link_lin_vel_w": rs.link_lin_vel_w.data_ptr(),
        "link_ang_vel_w": rs.link_ang_vel_w.data_ptr(),
    }

    art.refresh()
    first_root_pos = rs.root_pos.clone()
    art.refresh()
    second_root_pos = rs.root_pos.clone()

    ptrs_after = {
        "root_pos": rs.root_pos.data_ptr(),
        "root_quat": rs.root_quat.data_ptr(),
        "root_lin_vel_w": rs.root_lin_vel_w.data_ptr(),
        "root_ang_vel_w": rs.root_ang_vel_w.data_ptr(),
        "joint_pos": rs.joint_pos.data_ptr(),
        "joint_vel": rs.joint_vel.data_ptr(),
        "link_pos": rs.link_pos.data_ptr(),
        "link_quat_w": rs.link_quat_w.data_ptr(),
        "link_lin_vel_w": rs.link_lin_vel_w.data_ptr(),
        "link_ang_vel_w": rs.link_ang_vel_w.data_ptr(),
    }

    assert ptrs_before == ptrs_after, (
        "RobotState tensors were reallocated by refresh — the in-place copy was lost"
    )
    assert not torch.equal(first_root_pos, second_root_pos), (
        "FakeRefreshHandle ticks per call, so consecutive refreshes must produce different "
        "root_pos values (sanity check that data is actually being copied in)"
    )


def test_refresh_without_genesis_getters_keeps_buffers_at_zero() -> None:
    """``AttributeError`` from missing getters must short-circuit cleanly (used by tests
    in ``test_actuator.py``). The in-place change preserves this fallback."""

    from genelab.actuator import ImplicitPDActuatorCfg

    class _BareHandle:
        def __init__(self) -> None:
            self.joints = [_FakeJoint("j0", 0)]
            self.links = [_FakeLink("base")]
            self.n_dofs = 1

        def set_dofs_kp(self, *_args: Any, **_kwargs: Any) -> None: ...
        def set_dofs_kv(self, *_args: Any, **_kwargs: Any) -> None: ...
        def set_dofs_force_range(self, *_args: Any, **_kwargs: Any) -> None: ...

    cfg = ArticulationCfg(
        mjcf_path="/dev/null",
        default_joint_pos={"j0": 0.0},
        actuators={
            "all": ImplicitPDActuatorCfg(
                target_names_expr=(".*",), stiffness=1.0, damping=0.1
            ),
        },
    )
    art = Articulation(cfg, name="bare")
    art._gs_handle = cast(Any, _BareHandle())  # noqa: SLF001
    art.bind(num_envs=2, device="cpu")
    art.refresh()
    # All getters raised AttributeError → buffers stay at their RobotState init values.
    assert torch.equal(art.data.root_pos, torch.zeros(2, 3))
    assert torch.allclose(art.data.root_quat[:, 0], torch.ones(2))
