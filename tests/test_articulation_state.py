"""Behaviour-pinning tests for :class:`~genelab.entity.articulation.Articulation`.

ADR-0019 PR-A: the prerequisite test net for the Phase 5 decomposition. These
tests run **without Genesis** by injecting a stateful fake robot handle that
stores what the write-back methods push and returns it from the read getters,
so a ``write_* -> refresh -> data`` cycle is a true round-trip. They pin:

* binding / joint enumeration and the actuator-coverage validation contract
  (uncovered, conflicting, zero-match, and empty actuator groups all raise);
* joint-state and root-state write -> refresh round-trips (incl. partial
  ``env_ids``);
* ``reset`` restoring the home pose + zeroing velocity + the cached target;
* ``build_per_joint_tensor`` regex mapping;
* the implicit-PD branch of ``write_joint_targets_partial``;
* the empty-``env_ids`` no-op guards on every write path.

The split in PR-B/C/D must keep all of these green.
"""

from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from genelab.actuator import ImplicitPDActuatorCfg  # noqa: E402
from genelab.entity.articulation import Articulation, ArticulationCfg  # noqa: E402


class _FakeJoint:
    def __init__(self, name: str, dof: int) -> None:
        self.name = name
        self.n_dofs = 1
        self.dofs_idx_local = (dof,)


class _FakeLink:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRobot:
    """Stateful stand-in for a Genesis ``RigidEntity``.

    Each actuated joint owns exactly one DoF (``dofs_idx_local=(i,)``), so the
    articulation's ``_actuated_dof_idx`` is ``[0, 1, ...]`` and indexing the DoF
    buffers by it is the identity — keeping the round-trip assertions direct.
    Write setters mutate the buffers in place; read getters return them, so
    ``write_* -> refresh`` reflects exactly what was written.
    """

    def __init__(
        self,
        joint_names: tuple[str, ...],
        num_envs: int,
        *,
        with_limits: bool = False,
    ) -> None:
        self.joints = [_FakeJoint(name, i) for i, name in enumerate(joint_names)]
        self.links = [_FakeLink("base"), _FakeLink("tip")]
        self.n_dofs = len(joint_names)
        self._with_limits = with_limits
        n = self.n_dofs
        self._dof_pos = torch.zeros(num_envs, n)
        self._dof_vel = torch.zeros(num_envs, n)
        self._root_pos = torch.zeros(num_envs, 3)
        self._root_quat = torch.zeros(num_envs, 4)
        self._root_quat[:, 0] = 1.0
        self._root_vel = torch.zeros(num_envs, 3)
        self._root_ang = torch.zeros(num_envs, 3)
        self.control_target: torch.Tensor | None = None

    # -- static-parameter setters invoked by ActuatorBase.initialize (no-ops) --
    def set_dofs_kp(self, *_a: Any, **_k: Any) -> None: ...
    def set_dofs_kv(self, *_a: Any, **_k: Any) -> None: ...
    def set_dofs_force_range(self, *_a: Any, **_k: Any) -> None: ...

    # -- joint-state write-back --
    def set_dofs_position(
        self, value: torch.Tensor, dof_idx: torch.Tensor, envs_idx: Any = None
    ) -> None:
        if envs_idx is None:
            self._dof_pos[:, dof_idx] = value
        else:
            self._dof_pos[envs_idx.unsqueeze(1), dof_idx.unsqueeze(0)] = value

    def set_dofs_velocity(
        self, value: torch.Tensor, dof_idx: torch.Tensor, envs_idx: Any = None
    ) -> None:
        if envs_idx is None:
            self._dof_vel[:, dof_idx] = value
        else:
            self._dof_vel[envs_idx.unsqueeze(1), dof_idx.unsqueeze(0)] = value

    # -- root-state write-back --
    def _set_root(self, buf: torch.Tensor, value: torch.Tensor, envs_idx: Any) -> None:
        if envs_idx is None:
            buf[:] = value
        else:
            buf[envs_idx] = value

    def set_pos(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self._set_root(self._root_pos, value, envs_idx)

    def set_quat(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self._set_root(self._root_quat, value, envs_idx)

    def set_vel(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self._set_root(self._root_vel, value, envs_idx)

    def set_ang(self, value: torch.Tensor, envs_idx: Any = None) -> None:
        self._set_root(self._root_ang, value, envs_idx)

    # -- control routing --
    def control_dofs_position(self, target: torch.Tensor, _dofs: torch.Tensor) -> None:
        self.control_target = target.clone()

    # -- read getters --
    def get_dofs_position(self) -> torch.Tensor:
        return self._dof_pos

    def get_dofs_velocity(self) -> torch.Tensor:
        return self._dof_vel

    def get_pos(self) -> torch.Tensor:
        return self._root_pos

    def get_quat(self) -> torch.Tensor:
        return self._root_quat

    def get_vel(self) -> torch.Tensor:
        return self._root_vel

    def get_ang(self) -> torch.Tensor:
        return self._root_ang

    def get_dofs_limit(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._with_limits:
            raise AttributeError("get_dofs_limit")
        lower = torch.full((self.n_dofs,), -1.0)
        upper = torch.full((self.n_dofs,), 1.0)
        return lower, upper


def _make_articulation(
    *,
    joint_names: tuple[str, ...] = ("j0", "j1"),
    num_envs: int = 2,
    actuators: dict[str, Any] | None = None,
    default_joint_pos: dict[str, float] | None = None,
    joint_vel_limit: float | None = None,
    with_limits: bool = False,
) -> Articulation:
    """Construct a bound ``Articulation`` backed by a stateful fake robot handle."""
    if actuators is None:
        actuators = {
            "all": ImplicitPDActuatorCfg(target_names_expr=(".*",), stiffness=1.0, damping=0.1),
        }
    cfg = ArticulationCfg(
        mjcf_path="/dev/null",
        default_joint_pos=default_joint_pos or {name: 0.0 for name in joint_names},
        actuators=actuators,
        joint_vel_limit=joint_vel_limit,
    )
    art = Articulation(cfg, name="test")
    art._gs_handle = cast(Any, _FakeRobot(joint_names, num_envs, with_limits=with_limits))  # noqa: SLF001
    art.bind(num_envs=num_envs, device="cpu")
    return art


# --------------------------------------------------------------------- binding


def test_bind_enumerates_joints_links_and_state() -> None:
    art = _make_articulation(joint_names=("j0", "j1", "j2"), num_envs=4)
    assert art.num_dofs == 3
    assert art.joint_names == ["j0", "j1", "j2"]
    assert torch.equal(art.actuated_dof_ids, torch.tensor([0, 1, 2]))
    assert art.link_names == ["base", "tip"]
    assert art.num_links == 2
    assert art.action_scale_tensor.shape == (3,)
    rs = art.data
    assert rs.joint_pos.shape == (4, 3)
    assert rs.root_pos.shape == (4, 3)
    assert rs.link_pos.shape == (4, 2, 3)


def test_bind_default_joint_pos_resolves_patterns() -> None:
    art = _make_articulation(default_joint_pos={"j0": 0.5, "j1": -0.3})
    assert torch.equal(art.default_joint_pos, torch.tensor([0.5, -0.3]))


def test_bind_reads_joint_pos_limits_from_handle() -> None:
    art = _make_articulation(with_limits=True)
    assert torch.equal(art.joint_pos_limits, torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]))


def test_bind_joint_vel_limits_default_infinite() -> None:
    art = _make_articulation()
    assert torch.isinf(art.joint_vel_limits).all()
    art2 = _make_articulation(joint_vel_limit=10.0)
    assert torch.equal(art2.joint_vel_limits, torch.tensor([10.0, 10.0]))


# ------------------------------------------------- joint matching / coverage


def test_actuator_groups_partition_joints() -> None:
    art = _make_articulation(
        actuators={
            "a": ImplicitPDActuatorCfg(target_names_expr=("j0",), stiffness=1.0, damping=0.1),
            "b": ImplicitPDActuatorCfg(target_names_expr=("j1",), stiffness=1.0, damping=0.1),
        },
    )
    assert set(art.actuators) == {"a", "b"}
    assert art.actuators["a"].joint_names == ["j0"]
    assert art.actuators["b"].joint_names == ["j1"]


def test_bind_raises_on_uncovered_joint() -> None:
    with pytest.raises(ValueError, match="not.*covered"):
        _make_articulation(
            actuators={
                "a": ImplicitPDActuatorCfg(target_names_expr=("j0",), stiffness=1.0, damping=0.1),
            },
        )


def test_bind_raises_on_conflicting_groups() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _make_articulation(
            actuators={
                "a": ImplicitPDActuatorCfg(target_names_expr=(".*",), stiffness=1.0, damping=0.1),
                "b": ImplicitPDActuatorCfg(target_names_expr=("j0",), stiffness=1.0, damping=0.1),
            },
        )


def test_bind_raises_on_zero_match_group() -> None:
    with pytest.raises(ValueError, match="zero joints"):
        _make_articulation(
            actuators={
                "a": ImplicitPDActuatorCfg(target_names_expr=("nope",), stiffness=1.0, damping=0.1),
            },
        )


def test_bind_raises_on_empty_actuators() -> None:
    with pytest.raises(ValueError, match="actuators is empty"):
        _make_articulation(actuators={})


# --------------------------------------------------------------- round-trips


def test_write_joint_state_round_trips_through_refresh() -> None:
    art = _make_articulation(num_envs=3)
    pos = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    vel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    env_ids = torch.tensor([0, 2], dtype=torch.long)

    art.write_joint_state(pos, vel, env_ids)
    art.refresh()

    assert torch.equal(art.data.joint_pos.index_select(0, env_ids), pos)
    assert torch.equal(art.data.joint_vel.index_select(0, env_ids), vel)
    # The untouched env (1) stays at its initial zeros.
    assert torch.equal(art.data.joint_pos[1], torch.zeros(2))


def test_write_root_state_round_trips_through_refresh() -> None:
    art = _make_articulation(num_envs=2)
    root_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    lin = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ang = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    art.write_root_state(root_pos, root_quat, lin, ang, env_ids)
    art.refresh()

    assert torch.equal(art.data.root_pos, root_pos)
    assert torch.equal(art.data.root_quat, root_quat)
    assert torch.equal(art.data.root_lin_vel_w, lin)
    assert torch.equal(art.data.root_ang_vel_w, ang)


# --------------------------------------------------------------------- reset


def test_reset_restores_home_pose_and_target() -> None:
    art = _make_articulation(num_envs=2, default_joint_pos={"j0": 0.5, "j1": -0.3})
    default = torch.tensor([0.5, -0.3])

    # Disturb both envs away from home.
    disturb = torch.tensor([[9.0, 9.0], [9.0, 9.0]])
    art.write_joint_state(disturb, disturb, torch.tensor([0, 1], dtype=torch.long))

    art.reset(torch.tensor([0], dtype=torch.long))
    art.refresh()

    assert torch.equal(art.data.joint_pos[0], default)
    assert torch.equal(art.data.joint_vel[0], torch.zeros(2))
    # Cached PD target for the reset env returns to the home pose.
    assert torch.equal(art._joint_pos_target[0], default)  # noqa: SLF001
    # The non-reset env keeps its disturbed state.
    assert torch.equal(art.data.joint_pos[1], disturb[1])


# ----------------------------------------------------- build_per_joint_tensor


def test_build_per_joint_tensor_maps_patterns_with_default() -> None:
    art = _make_articulation(joint_names=("j0", "j1", "j2"), num_envs=2)
    out = art.build_per_joint_tensor({"j1": 5.0}, default=1.0)
    assert torch.equal(out, torch.tensor([1.0, 5.0, 1.0]))
    out_all = art.build_per_joint_tensor({".*": 2.0}, default=0.0)
    assert torch.equal(out_all, torch.tensor([2.0, 2.0, 2.0]))


# ------------------------------------------------ write_joint_targets_partial


def test_write_joint_targets_partial_routes_implicit_pd() -> None:
    art = _make_articulation(num_envs=2)
    local_joint_ids = torch.tensor([0, 1], dtype=torch.long)
    target = torch.tensor([[0.2, 0.3], [0.4, 0.5]])

    art.write_joint_targets_partial(local_joint_ids, target)

    # Target is stashed for the next step...
    assert torch.equal(art._joint_pos_target, target)  # noqa: SLF001
    # ...and routed to the implicit-PD control channel.
    handle = cast(Any, art.gs_handle)
    assert handle.control_target is not None
    assert torch.equal(handle.control_target, target)


# ------------------------------------------------------- empty-env_ids guards


def test_write_paths_are_noops_for_empty_env_ids() -> None:
    art = _make_articulation(num_envs=2)
    empty = torch.empty(0, dtype=torch.long)
    handle = cast(Any, art.gs_handle)
    before_pos = handle._dof_pos.clone()  # noqa: SLF001
    before_root = handle._root_pos.clone()  # noqa: SLF001

    art.write_joint_state(torch.zeros(0, 2), torch.zeros(0, 2), empty)
    art.write_root_state(
        torch.zeros(0, 3), torch.zeros(0, 4), torch.zeros(0, 3), torch.zeros(0, 3), empty
    )
    art.reset(empty)

    assert torch.equal(handle._dof_pos, before_pos)  # noqa: SLF001
    assert torch.equal(handle._root_pos, before_root)  # noqa: SLF001
