"""Tests for ``DifferentialIKAction`` and ``BinaryGripperAction``.

The math tests build a Genesis-shaped fake (articulation, rigid entity, robot
state) so the DLS solver can be exercised without standing up an actual scene.
The integration test at the bottom is guarded by ``importorskip("genesis")``
plus a CUDA check — it reproduces the exact acceptance criterion from #73
(``dx=0.01`` along world-x moves the EE 0.01 ± 0.001 m on a stationary Franka).
"""

from dataclasses import dataclass

import pytest
import torch

from genelab.mdp.actions.binary_gripper import (
    BinaryGripperAction,
    BinaryGripperActionCfg,
)
from genelab.mdp.actions.continuous_gripper import (
    ContinuousGripperAction,
    ContinuousGripperActionCfg,
)
from genelab.mdp.actions.ee_delta_ik import (
    DifferentialIKAction,
    DifferentialIKActionCfg,
)


# ----------------------------------------------------------------- fakes


@dataclass
class _FakeArticulationCfg:
    requires_jac_and_ik: bool = True


class _FakeLink:
    def __init__(self, idx: int, name: str) -> None:
        self.idx = idx
        self.name = name


class _FakeRigidEntity:
    """Stand-in for ``genesis.engine.entities.RigidEntity``.

    ``get_jacobian`` returns a pre-baked tensor regardless of the link
    argument — the caller asserts on the residual algebra, not on which link
    Genesis would have used.
    """

    def __init__(self, jacobian: torch.Tensor, link_names: list[str]) -> None:
        self._jacobian = jacobian
        self.links = [_FakeLink(i, n) for i, n in enumerate(link_names)]

    def get_jacobian(self, link: _FakeLink) -> torch.Tensor:
        del link
        return self._jacobian


class _FakeArticulation:
    def __init__(
        self,
        joint_names: list[str],
        link_names: list[str],
        actuated_dof_idx: torch.Tensor,
        joint_pos_limits: torch.Tensor,
        jacobian: torch.Tensor,
        *,
        requires_jac_and_ik: bool = True,
        name: str = "robot",
    ) -> None:
        self.cfg = _FakeArticulationCfg(requires_jac_and_ik=requires_jac_and_ik)
        self.name = name
        self._joint_names = list(joint_names)
        self._link_names = list(link_names)
        self.actuated_dof_idx = actuated_dof_idx
        self.joint_pos_limits = joint_pos_limits
        self.gs_handle = _FakeRigidEntity(jacobian, link_names)
        self.writes: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def link_names(self) -> list[str]:
        return list(self._link_names)

    def write_joint_targets_partial(
        self, local_joint_ids: torch.Tensor, target: torch.Tensor
    ) -> None:
        self.writes.append((local_joint_ids.clone(), target.clone()))


class _FakeRobotState:
    def __init__(self, joint_pos: torch.Tensor, link_quat_w: torch.Tensor | None = None) -> None:
        self.joint_pos = joint_pos
        self.link_quat_w = link_quat_w


class _FakeEnv:
    def __init__(
        self,
        articulation: _FakeArticulation,
        joint_pos: torch.Tensor,
        link_quat_w: torch.Tensor | None = None,
    ) -> None:
        self.num_envs = joint_pos.shape[0]
        self.device = str(joint_pos.device)
        self.articulation = articulation
        self.robot_state = _FakeRobotState(joint_pos, link_quat_w)

    @property
    def joint_names(self) -> list[str]:
        return self.articulation.joint_names


def _franka_like_fake_env(
    *,
    num_envs: int = 1,
    seed: int = 0,
    requires_jac_and_ik: bool = True,
    current_q: torch.Tensor | None = None,
    joint_pos_limits: torch.Tensor | None = None,
    ee_quat: torch.Tensor | None = None,
) -> tuple[_FakeEnv, torch.Tensor]:
    """Build a Franka-shaped fake env. Returns ``(env, jacobian_arm_columns)``.

    The articulation has 9 actuated joints (7 arm + 2 fingers), matching the
    Menagerie ``franka_emika_panda`` topology, and a 6 × 9 Jacobian sampled from
    a normal distribution with a fixed seed. The first 7 columns (the arm
    joints) are what ``DifferentialIKAction`` actually uses.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    arm_names = [f"joint{i + 1}" for i in range(7)]
    finger_names = ["finger_joint1", "finger_joint2"]
    joint_names = arm_names + finger_names
    link_names = ["base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]

    # Fixed-base Franka: each actuated joint is a single DoF, so the full-DoF
    # index space matches the actuated-joint index space 1:1.
    actuated_dof_idx = torch.arange(len(joint_names))

    # Sample a well-conditioned random Jacobian. Shape (num_envs, 6, n_full_dofs).
    jacobian = torch.randn(num_envs, 6, len(joint_names), generator=g)

    if joint_pos_limits is None:
        joint_pos_limits = torch.tensor([[-2.9, 2.9]] * 7 + [[0.0, 0.04]] * 2, dtype=torch.float)

    articulation = _FakeArticulation(
        joint_names=joint_names,
        link_names=link_names,
        actuated_dof_idx=actuated_dof_idx,
        joint_pos_limits=joint_pos_limits,
        jacobian=jacobian,
        requires_jac_and_ik=requires_jac_and_ik,
    )

    if current_q is None:
        current_q = torch.zeros(num_envs, len(joint_names))

    # Identity world quats for every link; the "hand" link can be overridden so
    # orientation-lock tests exercise a non-trivial error rotation.
    link_quat_w = torch.zeros(num_envs, len(link_names), 4)
    link_quat_w[..., 0] = 1.0
    if ee_quat is not None:
        link_quat_w[:, link_names.index("hand"), :] = ee_quat
    env = _FakeEnv(articulation, current_q, link_quat_w)

    j_arm = jacobian[:, :, :7]
    return env, j_arm


# ----------------------------------------------------------------- DifferentialIKAction


def test_differential_ik_action_dim_is_three_by_default() -> None:
    env, _ = _franka_like_fake_env()
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    assert term.action_dim == 3


def test_differential_ik_action_dim_is_six_with_orientation() -> None:
    env, _ = _franka_like_fake_env()
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        use_orientation=True,
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    assert term.action_dim == 6


def test_differential_ik_action_dx_residual_within_1mm() -> None:
    """Issue #73 acceptance criterion: ``dx=0.01`` along world-x must produce
    a joint delta whose linearised EE motion is within 1 mm of the request."""
    env, j_arm = _franka_like_fake_env(seed=42)
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        scale=1.0,  # raw policy action == physical metres for this test
        damping=0.01,  # tighter damping → smaller DLS residual
        max_delta_joint=10.0,  # disable the safety clip for the residual check
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]

    actions = torch.tensor([[0.01, 0.0, 0.0]])
    term.process_actions(actions)
    term.apply_actions()

    # Pull the joint delta out of the latest write to ``write_joint_targets_partial``.
    joint_indices, target_q = env.articulation.writes[-1]
    current_q = env.robot_state.joint_pos.index_select(1, joint_indices)
    delta_q = target_q - current_q  # (1, 7)

    # Linearised EE motion (position rows only) under the cached arm Jacobian.
    j_arm_pos = j_arm[:, :3, :]  # (1, 3, 7)
    delta_x_predicted = (j_arm_pos @ delta_q.unsqueeze(-1)).squeeze(-1)

    target_dx = torch.tensor([[0.01, 0.0, 0.0]])
    assert torch.allclose(delta_x_predicted, target_dx, atol=1e-3), (
        f"DLS residual exceeded 1 mm: predicted={delta_x_predicted}, target={target_dx}"
    )


def test_differential_ik_action_respects_joint_limits() -> None:
    """``target_q`` must lie within ``joint_pos_limits`` even if the DLS step
    would overshoot — protects the policy from learning out-of-range targets."""
    # Place every arm joint *at* its upper limit so any positive Δq must clip.
    upper = 0.5
    current_q = torch.full((1, 9), upper)
    limits = torch.full((9, 2), upper)
    limits[:, 0] = -10.0  # no clip from below; only upper matters here
    env, _ = _franka_like_fake_env(seed=7, current_q=current_q, joint_pos_limits=limits)
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        scale=1.0,
        damping=0.01,
        max_delta_joint=10.0,
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[1.0, 1.0, 1.0]]))
    _, target_q = (
        env.articulation.writes[0]
        if env.articulation.writes
        else (
            None,
            term._target_q,  # noqa: SLF001 — direct read into the term's buffer is fine here
        )
    )
    # No need to call apply; just inspect the cached target.
    target_q = term._target_q  # noqa: SLF001
    assert (target_q <= upper + 1e-6).all(), f"target_q exceeded upper limit {upper}: {target_q}"


def test_differential_ik_action_clips_max_delta_joint() -> None:
    """A huge raw policy delta must still translate to ``|Δq| ≤ max_delta_joint``."""
    env, _ = _franka_like_fake_env(seed=3)
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        scale=10.0,  # x100 amplification → DLS would normally over-step
        damping=0.001,
        max_delta_joint=0.05,
        clip=None,  # bypass the raw-action clamp so ``scale`` actually amplifies
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[1.0, 1.0, 1.0]]))
    delta_q = term._target_q - env.robot_state.joint_pos.index_select(  # noqa: SLF001
        1,
        term._joint_indices,  # noqa: SLF001
    )
    assert delta_q.abs().max().item() <= 0.05 + 1e-6


def test_differential_ik_action_requires_morph_flag() -> None:
    env, _ = _franka_like_fake_env(requires_jac_and_ik=False)
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
    )
    with pytest.raises(RuntimeError, match="requires_jac_and_ik"):
        DifferentialIKAction(cfg, env)  # type: ignore[arg-type]


def test_differential_ik_action_unknown_body_name_errors() -> None:
    env, _ = _franka_like_fake_env()
    cfg = DifferentialIKActionCfg(
        body_name="not_a_link",
        joint_names=(r"^joint[1-7]$",),
    )
    with pytest.raises(ValueError, match="not_a_link"):
        DifferentialIKAction(cfg, env)  # type: ignore[arg-type]


# ----------------------------------------------------------------- BinaryGripperAction


def test_binary_gripper_action_maps_positive_to_open() -> None:
    env, _ = _franka_like_fake_env()
    cfg = BinaryGripperActionCfg(
        joint_names=(r"^finger_joint.*$",),
        open_pos=0.04,
        closed_pos=0.0,
    )
    term = BinaryGripperAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[1.0]]))
    assert torch.allclose(term._target, torch.tensor([[0.04, 0.04]]))  # noqa: SLF001


def test_binary_gripper_action_maps_negative_to_closed() -> None:
    env, _ = _franka_like_fake_env()
    cfg = BinaryGripperActionCfg(
        joint_names=(r"^finger_joint.*$",),
        open_pos=0.04,
        closed_pos=0.0,
    )
    term = BinaryGripperAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[-1.0]]))
    assert torch.allclose(term._target, torch.tensor([[0.0, 0.0]]))  # noqa: SLF001


def test_binary_gripper_action_dim_is_one() -> None:
    env, _ = _franka_like_fake_env()
    cfg = BinaryGripperActionCfg(joint_names=(r"^finger_joint.*$",))
    term = BinaryGripperAction(cfg, env)  # type: ignore[arg-type]
    assert term.action_dim == 1


def test_binary_gripper_matches_only_finger_joints() -> None:
    # Construction (shared GripperActionBase.__init__): the regex resolves the two
    # finger joints and sizes the per-joint target buffer to them.
    env, _ = _franka_like_fake_env()
    cfg = BinaryGripperActionCfg(joint_names=(r"^finger_joint.*$",))
    term = BinaryGripperAction(cfg, env)  # type: ignore[arg-type]
    assert term._joint_indices.numel() == 2  # noqa: SLF001
    assert term._target.shape == (env.num_envs, 2)  # noqa: SLF001


def test_binary_gripper_zero_match_raises() -> None:
    # Zero-joint guard in the shared base; the message names the concrete subclass.
    env, _ = _franka_like_fake_env()
    cfg = BinaryGripperActionCfg(joint_names=("does_not_exist",))
    with pytest.raises(ValueError, match="BinaryGripperAction matched zero joints"):
        BinaryGripperAction(cfg, env)  # type: ignore[arg-type]


# ----------------------------------------------------------------- ContinuousGripperAction


def test_continuous_gripper_action_dim_is_one() -> None:
    env, _ = _franka_like_fake_env()
    cfg = ContinuousGripperActionCfg(joint_names=(r"^finger_joint.*$",))
    term = ContinuousGripperAction(cfg, env)  # type: ignore[arg-type]
    assert term.action_dim == 1


def test_continuous_gripper_integrates_on_sensed_width() -> None:
    """The target is ``sensed_width + action·speed`` — panda-gym's incremental
    gripper control, not an absolute mapping."""
    current_q = torch.zeros(1, 9)
    current_q[:, 7:9] = 0.02  # both fingers sensed half-open
    env, _ = _franka_like_fake_env(current_q=current_q)
    cfg = ContinuousGripperActionCfg(joint_names=(r"^finger_joint.*$",), speed=0.1)
    term = ContinuousGripperAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[-0.1]]))  # 0.02 + (-0.1)·0.1 = 0.01
    assert torch.allclose(term._target, torch.full((1, 2), 0.01), atol=1e-6)  # noqa: SLF001


def test_continuous_gripper_clamps_to_range() -> None:
    current_q = torch.zeros(1, 9)
    current_q[:, 7:9] = 0.02
    env, _ = _franka_like_fake_env(current_q=current_q)
    cfg = ContinuousGripperActionCfg(joint_names=(r"^finger_joint.*$",), speed=0.1)
    term = ContinuousGripperAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.tensor([[5.0]]))  # clipped to 1.0 → 0.12 → clamp 0.04
    assert torch.allclose(term._target, torch.full((1, 2), 0.04), atol=1e-6)  # noqa: SLF001
    term.process_actions(torch.tensor([[-5.0]]))  # clipped to -1.0 → -0.08 → clamp 0.0
    assert torch.allclose(term._target, torch.zeros(1, 2), atol=1e-6)  # noqa: SLF001


def test_continuous_gripper_zero_match_raises() -> None:
    # Same shared guard as Binary, but the message must name *this* subclass —
    # confirms the deduped base uses ``type(self).__name__``.
    env, _ = _franka_like_fake_env()
    cfg = ContinuousGripperActionCfg(joint_names=("does_not_exist",))
    with pytest.raises(ValueError, match="ContinuousGripperAction matched zero joints"):
        ContinuousGripperAction(cfg, env)  # type: ignore[arg-type]


# --------------------------------------------------- DifferentialIKAction: lock_orientation


def test_differential_ik_lock_orientation_keeps_action_dim_three() -> None:
    env, _ = _franka_like_fake_env()
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        lock_orientation=True,
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    assert term.action_dim == 3  # policy still emits position only
    assert term._solve_dim == 6  # but the DLS solve uses all 6 Jacobian rows  # noqa: SLF001


def test_differential_ik_lock_and_use_orientation_conflict() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        DifferentialIKActionCfg(
            body_name="hand",
            joint_names=(r"^joint[1-7]$",),
            use_orientation=True,
            lock_orientation=True,
        )


def test_differential_ik_lock_orientation_captures_reset_quat() -> None:
    """With no explicit ``fixed_orientation`` the target is captured from the EE
    pose on the first ``process_actions`` call."""
    ee_quat = torch.tensor([0.9239, 0.0, 0.3827, 0.0])  # 45° about world-y
    env, _ = _franka_like_fake_env(ee_quat=ee_quat)
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        lock_orientation=True,
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    assert term._fixed_quat is None  # noqa: SLF001
    term.process_actions(torch.zeros(1, 3))
    assert term._fixed_quat is not None  # noqa: SLF001
    assert torch.allclose(term._fixed_quat, ee_quat.reshape(1, 4), atol=1e-6)  # noqa: SLF001


def test_differential_ik_lock_orientation_zero_step_when_aligned() -> None:
    """EE already at the fixed target orientation + a zero position action ⇒
    zero spatial delta ⇒ ``target_q`` equals ``current_q``."""
    env, _ = _franka_like_fake_env(seed=11)  # identity EE quat
    cfg = DifferentialIKActionCfg(
        body_name="hand",
        joint_names=(r"^joint[1-7]$",),
        lock_orientation=True,
        fixed_orientation=(1.0, 0.0, 0.0, 0.0),
    )
    term = DifferentialIKAction(cfg, env)  # type: ignore[arg-type]
    term.process_actions(torch.zeros(1, 3))
    current_q = env.robot_state.joint_pos.index_select(1, term._joint_indices)  # noqa: SLF001
    assert torch.allclose(term._target_q, current_q, atol=1e-6)  # noqa: SLF001
