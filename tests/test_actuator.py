"""Acceptance tests for the M2 :mod:`genelab.actuator` package.

The tests sidestep Genesis by driving :class:`Articulation` with a fake gs handle that
records every ``set_dofs_*`` / ``control_dofs_*`` invocation. Each test pinpoints a single
behavioural promise made by the actuator design — regex partitioning, fail-fast on
mis-configured joints, and the three torque models.
"""

from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from genelab.actuator import (  # noqa: E402
    DCMotorActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitPDActuator,
    ImplicitPDActuatorCfg,
)
from genelab.entity.articulation import Articulation, ArticulationCfg  # noqa: E402


class _FakeJoint:
    """Mimics the subset of ``gs.joints[i]`` that :meth:`Articulation._enumerate_joints_and_links`
    reads. ``dofs_idx_local`` is what Genesis would expose for an MJCF without a free joint."""

    def __init__(self, name: str, dof_start: int) -> None:
        self.name = name
        self.n_dofs = 1
        self.dofs_idx_local = (dof_start,)


class _FakeLink:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeGenesisHandle:
    """Records every actuator-relevant Genesis call so tests can assert on it."""

    def __init__(self, joint_names: list[str]) -> None:
        self.joints = [_FakeJoint(name, i) for i, name in enumerate(joint_names)]
        self.links = [_FakeLink("base")]
        self.n_dofs = len(joint_names)
        self.kp_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.kv_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.force_range_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.position_target_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.force_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def set_dofs_kp(self, kp: torch.Tensor, dof_ids: torch.Tensor) -> None:
        self.kp_calls.append((kp.clone(), dof_ids.clone()))

    def set_dofs_kv(self, kv: torch.Tensor, dof_ids: torch.Tensor) -> None:
        self.kv_calls.append((kv.clone(), dof_ids.clone()))

    def set_dofs_force_range(
        self, lower: torch.Tensor, upper: torch.Tensor, dof_ids: torch.Tensor
    ) -> None:
        self.force_range_calls.append((lower.clone(), upper.clone(), dof_ids.clone()))

    def control_dofs_position(self, target: torch.Tensor, dof_ids: torch.Tensor) -> None:
        self.position_target_calls.append((target.clone(), dof_ids.clone()))

    def control_dofs_force(self, effort: torch.Tensor, dof_ids: torch.Tensor) -> None:
        self.force_calls.append((effort.clone(), dof_ids.clone()))


def _build_articulation(
    joint_names: list[str], actuators: dict[str, Any]
) -> tuple[Articulation, _FakeGenesisHandle]:
    cfg = ArticulationCfg(
        mjcf_path="/dev/null",
        default_joint_pos={name: 0.0 for name in joint_names},
        actuators=actuators,
    )
    art = Articulation(cfg, name="test")
    handle = _FakeGenesisHandle(joint_names)
    art._gs_handle = cast(Any, handle)  # noqa: SLF001 — direct injection for tests
    art.bind(num_envs=2, device="cpu")
    return art, handle


# ---------------------------------------------------------------------- regex matching


def test_actuator_regex_partitions_joints_disjointly() -> None:
    joint_names = ["left_hip", "right_hip", "torso"]
    actuators = {
        "hips": ImplicitPDActuatorCfg(target_names_expr=(".*_hip",), stiffness=10.0, damping=1.0),
        "torso": ImplicitPDActuatorCfg(target_names_expr=("torso",), stiffness=20.0, damping=2.0),
    }
    art, _ = _build_articulation(joint_names, actuators)
    grp = art.actuators
    hip_ids = grp["hips"].joint_ids.tolist()
    torso_ids = grp["torso"].joint_ids.tolist()
    assert sorted(hip_ids) == [0, 1]
    assert torso_ids == [2]
    assert set(hip_ids).isdisjoint(set(torso_ids))


# ---------------------------------------------------------------------- fail-fast paths


def test_unmatched_joint_raises_value_error() -> None:
    joint_names = ["a", "b", "c"]
    actuators = {
        "ab": ImplicitPDActuatorCfg(target_names_expr=("a", "b"), stiffness=1.0, damping=0.1),
    }
    with pytest.raises(ValueError, match="not covered by any actuator group"):
        _build_articulation(joint_names, actuators)


def test_conflicting_actuators_raise_value_error() -> None:
    joint_names = ["a", "b"]
    actuators = {
        "group_x": ImplicitPDActuatorCfg(target_names_expr=("a", "b"), stiffness=1.0, damping=0.1),
        "group_y": ImplicitPDActuatorCfg(target_names_expr=("a",), stiffness=2.0, damping=0.2),
    }
    with pytest.raises(ValueError, match="conflicts on joints"):
        _build_articulation(joint_names, actuators)


# ---------------------------------------------------------------------- ImplicitPD


def test_implicit_pd_writes_kp_kv_to_sim_and_compute_returns_none() -> None:
    joint_names = ["j0", "j1"]
    cfg = ImplicitPDActuatorCfg(target_names_expr=(".*",), stiffness=42.0, damping=3.5)
    art, handle = _build_articulation(joint_names, {"all": cfg})

    assert len(handle.kp_calls) == 1
    kp_tensor, kp_ids = handle.kp_calls[0]
    assert torch.allclose(kp_tensor, torch.tensor([42.0, 42.0]))
    assert kp_ids.tolist() == [0, 1]

    kv_tensor, _ = handle.kv_calls[0]
    assert torch.allclose(kv_tensor, torch.tensor([3.5, 3.5]))

    actuator = cast(ImplicitPDActuator, art.actuators["all"])
    out = actuator.compute(
        joint_pos=torch.zeros(2, 2),
        joint_vel=torch.zeros(2, 2),
        target_pos=torch.zeros(2, 2),
    )
    assert out is None


# ---------------------------------------------------------------------- IdealPD


def test_ideal_pd_unit_impulse_and_damping() -> None:
    cfg = IdealPDActuatorCfg(
        target_names_expr=("j0", "j1"),
        stiffness=100.0,
        damping=2.0,
        effort_limit=10000.0,
    )
    art, handle = _build_articulation(["j0", "j1"], {"all": cfg})
    actuator = cast(IdealPDActuator, art.actuators["all"])

    # tau = kp*(q*-q) - kv*qd. q=0, q*=1, qd=0 -> 100*1 - 0 = 100.
    out = actuator.compute(
        joint_pos=torch.zeros(1, 2),
        joint_vel=torch.zeros(1, 2),
        target_pos=torch.ones(1, 2),
    )
    assert out is not None and torch.allclose(out, torch.tensor([[100.0, 100.0]]))

    # q=1, q*=1, qd=10, kv=2 -> 100*0 - 2*10 = -20.
    out = actuator.compute(
        joint_pos=torch.ones(1, 2),
        joint_vel=torch.full((1, 2), 10.0),
        target_pos=torch.ones(1, 2),
    )
    assert out is not None and torch.allclose(out, torch.tensor([[-20.0, -20.0]]))

    # Force-channel actuators zero the simulator-side PD.
    kp_tensor, _ = handle.kp_calls[0]
    assert torch.allclose(kp_tensor, torch.zeros(2))
    kv_tensor, _ = handle.kv_calls[0]
    assert torch.allclose(kv_tensor, torch.zeros(2))


def test_ideal_pd_clips_to_effort_limit() -> None:
    cfg = IdealPDActuatorCfg(
        target_names_expr=("j0",),
        stiffness=1000.0,
        damping=0.0,
        effort_limit=15.0,
    )
    art, _ = _build_articulation(["j0"], {"all": cfg})
    actuator = cast(IdealPDActuator, art.actuators["all"])
    # tau_pd would be 1000.0; clipped to 15.0.
    out = actuator.compute(
        joint_pos=torch.zeros(1, 1),
        joint_vel=torch.zeros(1, 1),
        target_pos=torch.ones(1, 1),
    )
    assert out is not None and torch.allclose(out, torch.tensor([[15.0]]))
    # Same in reverse.
    out = actuator.compute(
        joint_pos=torch.ones(1, 1),
        joint_vel=torch.zeros(1, 1),
        target_pos=torch.zeros(1, 1),
    )
    assert out is not None and torch.allclose(out, torch.tensor([[-15.0]]))


# ---------------------------------------------------------------------- DCMotor


def test_dc_motor_saturation_curve() -> None:
    """Linear de-rating only in the driving direction; reverse braking keeps full effort."""
    cfg = DCMotorActuatorCfg(
        target_names_expr=("j0",),
        stiffness=1000.0,
        damping=0.0,
        effort_limit=20.0,
        velocity_limit=10.0,
        saturation_effort=20.0,
    )
    art, _ = _build_articulation(["j0"], {"all": cfg})
    actuator = art.actuators["all"]

    target = torch.full((1, 1), 1.0)  # tau_pd → 1000 (large, will be clipped)

    # q̇ = 0 → derate=1, drive cap = 20, brake cap = 20 → tau = 20 (clipped from 1000).
    tau = actuator.compute(torch.zeros(1, 1), torch.zeros(1, 1), target)
    assert tau is not None and pytest.approx(tau.item(), rel=1e-6) == 20.0

    # q̇ = 5 (half of v_max), driving direction (tau_pd>0, qd>0) → derate=0.5 → drive cap = 10.
    tau = actuator.compute(torch.zeros(1, 1), torch.full((1, 1), 5.0), target)
    assert tau is not None and pytest.approx(tau.item(), rel=1e-6) == 10.0

    # q̇ ≥ v_max in driving direction → drive cap = 0 → tau = 0.
    tau = actuator.compute(torch.zeros(1, 1), torch.full((1, 1), 10.0), target)
    assert tau is not None and pytest.approx(tau.item(), abs=1e-6) == 0.0

    # Reverse braking: tau_pd>0 but q̇<0 → upper bound = brake_limit = 20 → tau = 20 (full).
    tau = actuator.compute(torch.zeros(1, 1), torch.full((1, 1), -5.0), target)
    assert tau is not None and pytest.approx(tau.item(), rel=1e-6) == 20.0
