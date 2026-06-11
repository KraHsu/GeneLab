"""Unit tests for locomotion gait-shaping rewards added in P1.

Tests follow the existing ``test_sensor.py`` pattern: a ``_FakeRewardEnv`` provides just
enough surface (link_names, robot_state, command_manager, sensors) for the reward
functions to read what they need. No Genesis runtime required.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

from genelab.managers.reward_manager import RewardTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.rewards import (
    alive_bonus,
    angular_momentum_penalty,
    applied_torque_l2,
    base_height_l2,
    body_angular_velocity_penalty,
    feet_air_time,
    feet_clearance,
    feet_slip,
    feet_swing_height,
    joint_acc_l2,
    joint_pos_limits,
    joint_vel_limits,
    lin_vel_z_l2,
    self_collision_cost,
    soft_landing,
    track_angular_velocity_z_exp,
    track_linear_velocity_xy_exp,
    upright_exp,
)
from genelab.sensor import ContactSensorCfg
from genelab.sensor.self_contact import SelfContactData, SelfContactSensor, SelfContactSensorCfg


def _foot_cfg(
    env,
    names=("left_foot", "right_foot"),
    offsets: tuple[tuple[float, float, float], ...] | None = None,
) -> SceneEntityCfg:
    """Helper: build a resolved asset_cfg pointing at the named foot links.

    When ``offsets`` is provided, the cfg carries per-link site offsets — the
    foot rewards then evaluate at ``link_pos + R · offset`` for position and
    ``v_link + ω × (R · offset)`` for velocity, matching the reference's site-frame
    reward signal (G1 ``left_foot``/``right_foot`` sites sit
    ``(0.04, 0, -0.037)`` below the ankle_roll_link origins).
    """
    cfg = SceneEntityCfg(name="robot", link_names=names, link_offsets=offsets)
    cfg.resolve(env)
    return cfg


def _link_cfg(env, link_name: str) -> SceneEntityCfg:
    cfg = SceneEntityCfg(name="robot", link_names=(link_name,))
    cfg.resolve(env)
    return cfg


# --------------------------------------------------------------------- fakes


@dataclass
class _FakeRobotState:
    """Minimal per-link state surface used by the locomotion rewards."""

    link_pos: torch.Tensor
    link_lin_vel_w: torch.Tensor
    link_ang_vel_w: torch.Tensor
    link_quat_w: torch.Tensor | None = None
    projected_gravity_b: torch.Tensor | None = None
    root_lin_vel_b: torch.Tensor | None = None
    root_ang_vel_b: torch.Tensor | None = None
    root_pos: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    joint_acc: torch.Tensor | None = None
    applied_torque: torch.Tensor | None = None


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor) -> None:
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:  # noqa: ARG002 - name is unused
        return self._command


class _FakeRewardRobot:
    def __init__(self, num_envs: int, num_links: int) -> None:
        self._force = torch.zeros(num_envs, num_links, 3)

    def get_links_net_contact_force(self) -> torch.Tensor:
        return self._force

    def set_contact_force(self, f: torch.Tensor) -> None:
        self._force = f


@dataclass
class _FakeRewardEnv:
    num_envs: int
    link_names: list[str]
    robot_state: _FakeRobotState
    command: torch.Tensor
    device: str = "cpu"
    sensors: dict[str, Any] = field(default_factory=dict)
    # ``feet_swing_height`` reads ``env.scene.terrain`` to measure swing relative to
    # the surface; ``None`` keeps these flat-ground tests on absolute world z.
    scene: Any = field(default_factory=lambda: SimpleNamespace(terrain=None))
    robot: _FakeRewardRobot | None = None
    # Per-actuated-joint (lower, upper) limits exposed by the real env via
    # ``Articulation.joint_pos_limits``. Populated lazily so tests that don't
    # touch ``joint_pos_limits`` don't need to wire it.
    joint_pos_limits: torch.Tensor | None = None
    # Per-actuated-joint velocity-limit magnitude (``Articulation.joint_vel_limits``).
    joint_vel_limits: torch.Tensor | None = None

    @property
    def command_manager(self) -> _FakeCommandManager:
        return _FakeCommandManager(self.command)

    def __post_init__(self) -> None:
        if self.robot is None:
            self.robot = _FakeRewardRobot(self.num_envs, len(self.link_names))


def _make_env(
    *,
    num_envs: int = 2,
    link_names: tuple[str, ...] = ("base", "left_foot", "right_foot"),
    foot_z: tuple[float, float] | None = None,
    foot_vel_xy: tuple[float, float] | None = None,
    base_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    command_xyz: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> _FakeRewardEnv:
    """Build a fake env with optional symmetric foot kinematics and a single command."""
    num_links = len(link_names)
    link_pos = torch.zeros(num_envs, num_links, 3)
    link_lin_vel_w = torch.zeros(num_envs, num_links, 3)
    link_ang_vel_w = torch.zeros(num_envs, num_links, 3)
    if foot_z is not None:
        # Apply to every link tagged "*foot*" — keep test setup terse.
        for i, n in enumerate(link_names):
            if "foot" in n:
                link_pos[:, i, 2] = float(
                    foot_z[0] if i == link_names.index("left_foot") else foot_z[1]
                )
    if foot_vel_xy is not None:
        for i, n in enumerate(link_names):
            if "foot" in n:
                link_lin_vel_w[:, i, 0] = float(foot_vel_xy[0])
                link_lin_vel_w[:, i, 1] = float(foot_vel_xy[1])
    link_ang_vel_w[:, 0, :] = torch.tensor(base_ang_vel)
    # Default link quaternions are identity (wxyz = [1, 0, 0, 0]).
    link_quat_w = torch.zeros(num_envs, num_links, 4)
    link_quat_w[..., 0] = 1.0
    # Root-frame gravity: world ``(0, 0, -1)`` with identity root quat — same vector.
    projected_gravity_b = torch.zeros(num_envs, 3)
    projected_gravity_b[:, 2] = -1.0
    # Root body-frame velocities + joint position scaffolding for tracking / limit tests.
    root_lin_vel_b = torch.zeros(num_envs, 3)
    root_ang_vel_b = torch.zeros(num_envs, 3)
    root_pos = torch.zeros(num_envs, 3)
    joint_pos = torch.zeros(num_envs, 1)
    state = _FakeRobotState(
        link_pos=link_pos,
        link_lin_vel_w=link_lin_vel_w,
        link_ang_vel_w=link_ang_vel_w,
        link_quat_w=link_quat_w,
        projected_gravity_b=projected_gravity_b,
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        root_pos=root_pos,
        joint_pos=joint_pos,
        joint_vel=torch.zeros(num_envs, 1),
        joint_acc=torch.zeros(num_envs, 1),
        applied_torque=torch.zeros(num_envs, 1),
    )
    command = torch.tensor(command_xyz, dtype=torch.float).unsqueeze(0).expand(num_envs, -1).clone()
    return _FakeRewardEnv(
        num_envs=num_envs, link_names=list(link_names), robot_state=state, command=command
    )


# --------------------------------------------------------------------- body_angular_velocity_penalty


# --------------------------------------------------------------------- upright_exp


def test_upright_exp_default_uses_root_projected_gravity() -> None:
    """Without ``asset_cfg``, the reward reads ``robot_state.projected_gravity_b``."""
    env = _make_env()
    assert env.robot_state.projected_gravity_b is not None
    # Force a non-zero tilt by setting the projected gravity xy.
    env.robot_state.projected_gravity_b[:, :2] = torch.tensor([0.3, -0.4])
    env.robot_state.projected_gravity_b[:, 2] = math.sqrt(1 - 0.25)  # keep unit-ish
    out = upright_exp(env, std=0.5)
    # xy_squared = 0.09 + 0.16 = 0.25; expected = exp(-0.25 / 0.25) = exp(-1).
    expected = math.exp(-1.0)
    assert torch.allclose(out, torch.full((env.num_envs,), expected), atol=1e-6)


def test_upright_exp_with_asset_cfg_projects_world_gravity_into_link_frame() -> None:
    """``asset_cfg`` selects a link; world gravity is projected into its frame.

    With an identity quaternion, the projection equals world gravity ``(0, 0, -1)``,
    xy=0, so the reward is 1.0 (perfectly upright).
    """
    env = _make_env()
    out = upright_exp(env, std=0.45, asset_cfg=_link_cfg(env, "base"))
    assert torch.allclose(out, torch.ones(env.num_envs), atol=1e-6)


def test_upright_exp_link_mode_picks_up_link_quat_tilt() -> None:
    """If a target link is rotated, its frame's projected gravity becomes non-zero."""
    env = _make_env()
    # Rotate the base link by 90° about +x: quat = (cos45, sin45, 0, 0) = (√2/2, √2/2, 0, 0).
    # World gravity (0, 0, -1) projected into a frame rotated 90° about +x becomes (0, 1, 0)
    # — xy magnitude = 1.
    s2 = math.sqrt(2.0) / 2.0
    assert env.robot_state.link_quat_w is not None
    env.robot_state.link_quat_w[:, 0] = torch.tensor([s2, s2, 0.0, 0.0])
    out = upright_exp(env, std=0.45, asset_cfg=_link_cfg(env, "base"))
    # xy_squared = 1.0; expected = exp(-1 / 0.45²) ≈ exp(-4.938).
    expected = math.exp(-1.0 / (0.45 * 0.45))
    assert torch.allclose(out, torch.full((env.num_envs,), expected), atol=1e-5)


def test_upright_exp_pelvis_vs_torso_diverge_under_waist_flex() -> None:
    """Regression for the reference parity gap: ``upright_exp`` with root mode and
    with ``asset_cfg=torso_link`` give DIFFERENT signals when the torso is
    rotated relative to the pelvis (waist joint flexed).

    Setup: pelvis stays upright (root projected_gravity_b = world up); torso link
    is rotated 45° forward (so its projected gravity has xy ≠ 0).
    """
    env = _make_env(link_names=("pelvis", "torso_link"))
    # Pelvis: perfect up. Already (0, 0, -1) by default.
    # Torso link: rotated 45° about +x. q = (cos22.5, sin22.5, 0, 0).
    angle = math.pi / 4
    assert env.robot_state.link_quat_w is not None
    env.robot_state.link_quat_w[:, 1] = torch.tensor(
        [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]
    )
    out_root = upright_exp(env, std=0.45)
    out_torso = upright_exp(env, std=0.45, asset_cfg=_link_cfg(env, "torso_link"))
    # Pelvis (root) sees no tilt → reward ~1.
    assert torch.allclose(out_root, torch.ones(env.num_envs), atol=1e-6)
    # Torso sees real tilt → reward < 1.
    assert (out_torso < 0.99).all().item()


# --------------------------------------------------------------------- body_angular_velocity_penalty


def test_body_angular_velocity_penalty_sums_xy_squared_only() -> None:
    env = _make_env(base_ang_vel=(0.3, -0.4, 5.0))  # z=5 must be ignored
    out = body_angular_velocity_penalty(env, _link_cfg(env, "base"))
    # 0.3^2 + (-0.4)^2 = 0.09 + 0.16 = 0.25
    assert torch.allclose(out, torch.full((env.num_envs,), 0.25), atol=1e-6)


def test_body_angular_velocity_penalty_zero_when_link_static() -> None:
    env = _make_env(base_ang_vel=(0.0, 0.0, 0.0))
    out = body_angular_velocity_penalty(env, _link_cfg(env, "base"))
    assert torch.all(out == 0.0)


# --------------------------------------------------------------------- feet_clearance


def test_feet_clearance_zero_when_command_below_threshold() -> None:
    """No locomotion command → no clearance penalty (otherwise standing envs accumulate)."""
    env = _make_env(
        foot_z=(0.5, 0.5),
        foot_vel_xy=(1.0, 0.0),
        command_xyz=(0.01, 0.0, 0.0),  # below default threshold 0.05
    )
    out = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    assert torch.all(out == 0.0)


def test_feet_clearance_gate_uses_l1_xy_plus_abs_yaw_not_l2_of_three() -> None:
    """Reference parity: gate is ``||cmd_xy|| + |cmd_z|``, not ``||cmd[:3]||``.

    For ``cmd=(0.03, 0, 0.03)`` and threshold ``0.05``:

    * the reference gate: ``0.03 + 0.03 = 0.06`` → active (penalty fires).
    * Pre-parity L2 gate: ``√(0.03² + 0.03²) ≈ 0.042`` → silent.

    Picking these numbers also keeps the threshold-boundary diagnostic simple:
    the GeneLab gate must now fire here.
    """
    env = _make_env(
        foot_z=(0.3, 0.3),
        foot_vel_xy=(1.0, 0.0),
        command_xyz=(0.03, 0.0, 0.03),
    )
    out = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    # 2 feet × |0.3 − 0.1| × 1.0 = 0.4 — non-zero confirms the gate fired.
    assert torch.all(out > 0.0)


def test_feet_clearance_gate_still_silent_below_combined_threshold() -> None:
    """The L1-xy + |ωz| gate stays silent when the combined total is sub-threshold.

    For ``cmd=(0.02, 0, 0.02)`` total = 0.04 < 0.05 ⇒ inactive. Sanity-check
    that the new formula isn't permissive everywhere; only the cases the L2
    rule missed.
    """
    env = _make_env(
        foot_z=(0.3, 0.3),
        foot_vel_xy=(1.0, 0.0),
        command_xyz=(0.02, 0.0, 0.02),
    )
    out = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    assert torch.all(out == 0.0)


def test_feet_clearance_penalty_uses_link_z_when_no_sensor() -> None:
    """Without ``height_sensor_name``, the reward falls back to ``link_pos.z``."""
    env = _make_env(
        foot_z=(0.3, 0.3),  # both feet 0.2 above target
        foot_vel_xy=(2.0, 0.0),  # vel_norm = 2.0
        command_xyz=(1.0, 0.0, 0.0),
    )
    out = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    # Σ_foot |h - target| * |v_xy| = 2 feet * 0.2 * 2.0 = 0.8
    assert torch.allclose(out, torch.full((env.num_envs,), 0.8), atol=1e-6)


def test_feet_clearance_link_offset_shifts_height_under_identity_rotation() -> None:
    """Reference parity: ``link_offsets`` moves the evaluation point to the foot site.

    With identity link quaternion, a site offset of ``(0.04, 0, -0.037)`` puts the
    evaluated height ``0.037 m`` *below* the link origin — important when the
    target swing height is ``0.1 m``. Setting ``link_z = 0.137`` then makes the
    site sit at exactly ``0.1 m`` and the reward goes to zero (perfect clearance).
    Without the offset the reward would be ``2 * |0.137 − 0.1| * 2 = 0.148`` per env.
    """
    env = _make_env(
        foot_z=(0.137, 0.137),
        foot_vel_xy=(2.0, 0.0),
        command_xyz=(1.0, 0.0, 0.0),
    )
    # With offset → site sits at 0.1 m → |h − target| = 0 → reward = 0.
    out_with = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env, offsets=((0.04, 0.0, -0.037), (0.04, 0.0, -0.037))),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    assert torch.allclose(out_with, torch.zeros(env.num_envs), atol=1e-6)
    # Without offset → reads link_z = 0.137 → reward = 2 * 0.037 * 2 = 0.148.
    out_without = feet_clearance(
        env,
        asset_cfg=_foot_cfg(env),
        target_height=0.1,
        command_name="twist",
        command_threshold=0.05,
    )
    assert torch.allclose(out_without, torch.full((env.num_envs,), 0.148), atol=1e-6)


def test_feet_slip_link_offset_adds_omega_cross_r_to_foot_velocity() -> None:
    """Reference parity: foot xy-velocity is ``v_link + ω × (R · offset)``.

    With offset ``r=(0.1, 0, 0)`` and link spinning about +z at ``ω_z=2`` rad/s,
    the cross-product contributes ``(0, 0.2, 0)`` to xy. Combined with the
    link-origin ``v_link = (3, 4, 0)``, the site sees ``(3, 4.2, 0)`` →
    ``|v_xy|² = 9 + 17.64 = 26.64`` per foot × 2 feet = ``53.28``. Without offset
    the reward gets the unshifted ``v_link`` and lands on ``50``.
    """
    env = _make_env(foot_vel_xy=(3.0, 4.0), command_xyz=(1.0, 0.0, 0.0))
    # Spin both feet about +z to surface the lever-arm term.
    assert env.robot_state.link_ang_vel_w is not None
    foot_ids = [env.link_names.index("left_foot"), env.link_names.index("right_foot")]
    for fid in foot_ids:
        env.robot_state.link_ang_vel_w[:, fid, 2] = 2.0
    # Push large contact forces so both feet register grounded.
    assert env.robot is not None
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 100.0], [0, 0, 100.0]]] * env.num_envs, dtype=torch.float)
    )
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=False
    ).build()
    contact.bind(env)
    contact.update(0.05)
    env.sensors["feet"] = contact

    out = feet_slip(
        env,
        sensor_name="feet",
        asset_cfg=_foot_cfg(env, offsets=((0.1, 0.0, 0.0), (0.1, 0.0, 0.0))),
        command_name="twist",
        command_threshold=0.05,
    )
    # Per foot: |(3, 4 + 0.2)|² = 9 + 17.64 = 26.64; × 2 feet = 53.28.
    assert torch.allclose(out, torch.full((env.num_envs,), 53.28), atol=1e-5)


# --------------------------------------------------------------------- feet_slip


def test_feet_slip_zero_when_feet_airborne() -> None:
    """No contact ⇒ no slip penalty regardless of foot velocity."""
    env = _make_env(foot_vel_xy=(3.0, 4.0), command_xyz=(1.0, 0.0, 0.0))
    # Wire a contact sensor saying "no contact" everywhere.
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=False
    ).build()
    contact.bind(env)
    contact.update(0.05)
    env.sensors["feet"] = contact
    out = feet_slip(
        env,
        sensor_name="feet",
        asset_cfg=_foot_cfg(env),
        command_name="twist",
        command_threshold=0.05,
    )
    assert torch.all(out == 0.0)


def test_feet_slip_penalises_grounded_xy_speed_squared() -> None:
    env = _make_env(foot_vel_xy=(3.0, 4.0), command_xyz=(1.0, 0.0, 0.0))
    # Push large contact forces so both feet register grounded.
    assert env.robot is not None
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 100.0], [0, 0, 100.0]]] * env.num_envs, dtype=torch.float)
    )
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=False
    ).build()
    contact.bind(env)
    contact.update(0.05)
    env.sensors["feet"] = contact
    out = feet_slip(
        env,
        sensor_name="feet",
        asset_cfg=_foot_cfg(env),
        command_name="twist",
        command_threshold=0.05,
    )
    # Σ_foot |v_xy|^2 = 2 feet * (3^2 + 4^2) = 2 * 25 = 50
    assert torch.allclose(out, torch.full((env.num_envs,), 50.0), atol=1e-5)


# --------------------------------------------------------------------- soft_landing


def test_soft_landing_fires_only_on_first_contact_step() -> None:
    env = _make_env(command_xyz=(1.0, 0.0, 0.0))
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    ).build()
    contact.bind(env)
    env.sensors["feet"] = contact

    # Two air ticks: no first_contact, no landing cost.
    assert env.robot is not None
    env.robot.set_contact_force(torch.zeros(env.num_envs, 3, 3))
    contact.update(0.02)
    contact._invalidate_cache()
    contact.update(0.02)
    out_air = soft_landing(env, sensor_name="feet", command_name="twist", command_threshold=0.05)
    assert torch.all(out_air == 0.0)

    # Landing tick: force jumps, first_contact fires, cost = Σ|F|.
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 50.0], [0, 0, 30.0]]] * env.num_envs, dtype=torch.float)
    )
    contact._invalidate_cache()
    contact.update(0.02)
    out_landing = soft_landing(
        env, sensor_name="feet", command_name="twist", command_threshold=0.05
    )
    assert torch.allclose(out_landing, torch.full((env.num_envs,), 80.0), atol=1e-5)

    # Continued contact: edge no longer fires.
    contact._invalidate_cache()
    contact.update(0.02)
    out_cont = soft_landing(env, sensor_name="feet", command_name="twist", command_threshold=0.05)
    assert torch.all(out_cont == 0.0)


# --------------------------------------------------------------------- feet_swing_height


def _swing_height_cfg(env, *, target: float, command_threshold: float = 0.05) -> RewardTermCfg:
    """Build the swing-height term cfg with a pre-resolved asset_cfg."""
    return RewardTermCfg(
        func=feet_swing_height,
        weight=1.0,
        params={
            "sensor_name": "feet",
            "asset_cfg": _foot_cfg(env),
            "target_height": target,
            "command_name": "twist",
            "command_threshold": command_threshold,
        },
    )


def test_feet_swing_height_charges_at_touchdown_proportional_to_apex_error() -> None:
    """Peak height reset at lift-off, accumulated in air, billed at landing."""
    env = _make_env(command_xyz=(1.0, 0.0, 0.0))
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    ).build()
    contact.bind(env)
    env.sensors["feet"] = contact

    # Build the stateful reward via the standard manager hook.
    cfg = _swing_height_cfg(env, target=0.1)
    reward = feet_swing_height(cfg=cfg, env=env)
    assert env.robot is not None

    # Step 1: contact (both feet grounded at z=0); no cost.
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 100.0], [0, 0, 100.0]]] * env.num_envs, dtype=torch.float)
    )
    contact.update(0.02)
    out = reward(env, **cfg.params)
    assert torch.all(out == 0.0)

    # Step 2: lift-off — feet move to z=0.05 in air.
    env.robot_state.link_pos[:, 1, 2] = 0.05
    env.robot_state.link_pos[:, 2, 2] = 0.05
    env.robot.set_contact_force(torch.zeros(env.num_envs, 3, 3))
    contact._invalidate_cache()
    contact.update(0.02)
    out = reward(env, **cfg.params)
    assert torch.all(out == 0.0)  # no landing yet

    # Step 3: airborne, feet reach apex z=0.12 (over target 0.10).
    env.robot_state.link_pos[:, 1, 2] = 0.12
    env.robot_state.link_pos[:, 2, 2] = 0.12
    contact._invalidate_cache()
    contact.update(0.02)
    _ = reward(env, **cfg.params)

    # Step 4: landing — feet back to z=0, contact forces fire, first_contact true.
    env.robot_state.link_pos[:, 1, 2] = 0.0
    env.robot_state.link_pos[:, 2, 2] = 0.0
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 100.0], [0, 0, 100.0]]] * env.num_envs, dtype=torch.float)
    )
    contact._invalidate_cache()
    contact.update(0.02)
    out = reward(env, **cfg.params)
    # Peak per foot = 0.12 → error = 0.12/0.10 - 1 = 0.2 → 0.04 squared per foot, 2 feet → 0.08.
    assert torch.allclose(out, torch.full((env.num_envs,), 0.08), atol=1e-5)


def test_feet_swing_height_silent_when_command_is_standing() -> None:
    env = _make_env(command_xyz=(0.01, 0.0, 0.0))  # below threshold
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    ).build()
    contact.bind(env)
    env.sensors["feet"] = contact
    cfg = _swing_height_cfg(env, target=0.1)
    reward = feet_swing_height(cfg=cfg, env=env)
    # Provoke first_contact and check the gate suppresses cost.
    assert env.robot is not None
    env.robot.set_contact_force(torch.zeros(env.num_envs, 3, 3))
    contact.update(0.02)
    env.robot.set_contact_force(
        torch.tensor([[[0, 0, 0], [0, 0, 100.0], [0, 0, 100.0]]] * env.num_envs, dtype=torch.float)
    )
    contact._invalidate_cache()
    contact.update(0.02)
    out = reward(env, **cfg.params)
    assert torch.all(out == 0.0)


# --------------------------------------------------------------------- angular_momentum_penalty


class _StubSensor:
    """Minimal sensor stand-in: ``data`` attribute is the only thing the rewards read."""

    def __init__(self, value) -> None:
        self.data = value


def test_angular_momentum_penalty_returns_squared_magnitude() -> None:
    """``L=(3, 4, 0)`` ⇒ ``||L||² = 25`` per env (reference parity: squared norm)."""
    env = _make_env()
    env.sensors["L"] = _StubSensor(
        torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    )
    out = angular_momentum_penalty(env, sensor_name="L")
    assert torch.allclose(out, torch.tensor([25.0, 1.0]), atol=1e-6)


# --------------------------------------------------------------------- track_lin/ang_vel


def test_track_linear_velocity_penalises_z_velocity() -> None:
    """Reference parity: ``exp(-((cmd-vel_xy)² + vel_z²) / std²)``.

    A perfectly tracked xy command with a non-zero z velocity should NOT score 1 —
    the z² term shows up in the error.
    """
    env = _make_env(command_xyz=(0.5, 0.0, 0.0))
    assert env.robot_state.root_lin_vel_b is not None
    env.robot_state.root_lin_vel_b[:, 0] = 0.5  # matches command exactly
    env.robot_state.root_lin_vel_b[:, 2] = 1.0  # but z is non-zero
    out = track_linear_velocity_xy_exp(env, command_name="twist", std=1.0)
    # xy_err = 0; z_err = 1; total = 1; reward = exp(-1) ≈ 0.368.
    expected = math.exp(-1.0)
    assert torch.allclose(out, torch.full((env.num_envs,), expected), atol=1e-6)


def test_track_angular_velocity_penalises_xy_angular_rates() -> None:
    """Reference parity: ``exp(-((cmd_z-vel_z)² + ||vel_xy||²) / std²)``."""
    env = _make_env(command_xyz=(0.0, 0.0, 0.3))
    assert env.robot_state.root_ang_vel_b is not None
    env.robot_state.root_ang_vel_b[:, 2] = 0.3  # tracks z command
    env.robot_state.root_ang_vel_b[:, 0] = 0.6  # but is pitching
    env.robot_state.root_ang_vel_b[:, 1] = -0.8
    out = track_angular_velocity_z_exp(env, command_name="twist", std=1.0)
    # z_err = 0; xy_err = 0.36 + 0.64 = 1.0; total = 1.0; reward = exp(-1) ≈ 0.368.
    expected = math.exp(-1.0)
    assert torch.allclose(out, torch.full((env.num_envs,), expected), atol=1e-6)


# --------------------------------------------------------------------- joint_pos_limits


def test_joint_pos_limits_zero_inside_real_per_joint_window() -> None:
    """Joints sitting within their (lower, upper) window contribute zero penalty."""
    env = _make_env()
    # Two joints: knee in [0, 2.0]; hip in [-1.5, 1.5]. joint_pos within both.
    env.joint_pos_limits = torch.tensor([[0.0, 2.0], [-1.5, 1.5]])
    assert env.robot_state.joint_pos is not None
    env.robot_state.joint_pos = torch.tensor([[1.0, 0.5], [1.8, -1.0]])
    out = joint_pos_limits(env)
    assert torch.allclose(out, torch.zeros(env.num_envs), atol=1e-6)


def test_joint_pos_limits_charges_absolute_excursion_per_joint() -> None:
    """Past-limit excursions accumulate as absolute distance, not squared."""
    env = _make_env()
    env.joint_pos_limits = torch.tensor([[0.0, 2.0], [-1.5, 1.5]])
    assert env.robot_state.joint_pos is not None
    # Env 0: knee at 2.3 (0.3 over upper); hip at 0 (in range).
    # Env 1: knee at -0.2 (0.2 under lower); hip at 1.8 (0.3 over upper).
    env.robot_state.joint_pos = torch.tensor([[2.3, 0.0], [-0.2, 1.8]])
    out = joint_pos_limits(env)
    # Env 0 → 0.3; env 1 → 0.2 + 0.3 = 0.5.
    expected = torch.tensor([0.3, 0.5])
    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------- feet_air_time


def test_feet_air_time_counts_feet_in_window() -> None:
    """Counts feet whose current_air_time is in (threshold_min, threshold_max)."""
    env = _make_env(command_xyz=(1.0, 0.0, 0.0))
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    ).build()
    contact.bind(env)
    env.sensors["feet"] = contact
    # Push the air_time state via the cached _AirTimeState so we don't depend on
    # update() timer arithmetic in this test.
    assert contact._air_state is not None
    contact._air_state.current_air_time = torch.tensor(
        [
            [0.2, 0.7],  # env 0: foot 0 in-range; foot 1 too long.
            [0.04, 0.1],  # env 1: foot 0 too short; foot 1 in-range.
        ]
    )
    contact._invalidate_cache()
    out = feet_air_time(
        env,
        sensor_name="feet",
        threshold_min=0.05,
        threshold_max=0.5,
        command_name="twist",
        command_threshold=0.5,
    )
    assert torch.allclose(out, torch.tensor([1.0, 1.0]), atol=1e-6)


def test_feet_air_time_gated_by_command_magnitude() -> None:
    """Below ``command_threshold`` the reward is zero on every env."""
    env = _make_env(command_xyz=(0.1, 0.0, 0.0))  # |cmd| = 0.1 below 0.5 threshold
    contact = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    ).build()
    contact.bind(env)
    env.sensors["feet"] = contact
    assert contact._air_state is not None
    contact._air_state.current_air_time = torch.tensor([[0.2, 0.3], [0.2, 0.3]])
    contact._invalidate_cache()
    out = feet_air_time(
        env,
        sensor_name="feet",
        threshold_min=0.05,
        threshold_max=0.5,
        command_name="twist",
        command_threshold=0.5,
    )
    assert torch.all(out == 0.0)


def test_velocity_tracking_error_l1_is_linear_and_axis_selectable() -> None:
    """``velocity_tracking_error_l1``: sum of |cmd[axis] − vel[axis]| over ``axes``.

    The exp tracking kernel saturates at large error (gradient ≈ 0 once the policy fully
    ignores an axis), so a policy that abandoned vy never feels a pull back. An L1 error
    term keeps a constant gradient at any distance; ``axes`` scopes it to the abandoned
    axis so well-tracked axes aren't double-penalized."""
    from genelab.mdp.rewards.tracking import velocity_tracking_error_l1

    env = _make_env(command_xyz=(0.0, 0.8, 0.0))
    assert env.robot_state.root_lin_vel_b is not None
    env.robot_state.root_lin_vel_b[:, 1] = 0.1  # vy far from the 0.8 command
    env.robot_state.root_lin_vel_b[:, 0] = 0.3  # vx error must NOT leak in via axes=(1,)
    out = velocity_tracking_error_l1(env, command_name="twist", axes=(1,))
    assert torch.allclose(out, torch.full((2,), 0.7), atol=1e-6)
    # Both axes: |0 - 0.3| + |0.8 - 0.1| = 1.0.
    out_xy = velocity_tracking_error_l1(env, command_name="twist", axes=(0, 1))
    assert torch.allclose(out_xy, torch.full((2,), 1.0), atol=1e-6)


def test_angular_velocity_tracking_error_l1_measures_yaw_command_gap() -> None:
    """``angular_velocity_tracking_error_l1``: |cmd_wz − ang_vel_z|, linear in the error.

    Same medicine as the linear L1: the exp yaw kernel saturates once the policy abandons
    pure-rotation commands (probed: in-place yaw collapsed to 2 % while mixed-command yaw
    still scored), so a constant-gradient pull is needed on the yaw axis too."""
    from genelab.mdp.rewards.tracking import angular_velocity_tracking_error_l1

    env = _make_env(command_xyz=(0.0, 0.0, 1.0))  # cmd: pure yaw 1.0 rad/s
    assert env.robot_state.root_ang_vel_b is not None
    env.robot_state.root_ang_vel_b[:, 2] = 0.2  # actual yaw far below command
    out = angular_velocity_tracking_error_l1(env, command_name="twist")
    assert torch.allclose(out, torch.full((2,), 0.8), atol=1e-6)


def test_feet_air_time_command_axes_gates_on_selected_components_only() -> None:
    """``command_axes`` restricts the gate to those command components.

    The wheeled-legged hybrid (Go2-W stage 2) rewards stepping only when *lateral* motion is
    demanded — rolling serves vx/yaw, so a full-magnitude gate would force stepping on pure
    forward commands where the wheels should stay grounded."""
    contact_cfg = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=True
    )
    air = torch.tensor([[0.2, 0.3], [0.2, 0.3]])

    # Pure forward command: axis-1 (vy) gate must stay closed even though |cmd| is large.
    env = _make_env(command_xyz=(1.0, 0.0, 0.0))
    contact = contact_cfg.build()
    contact.bind(env)
    env.sensors["feet"] = contact
    assert contact._air_state is not None
    contact._air_state.current_air_time = air.clone()
    contact._invalidate_cache()
    out = feet_air_time(
        env, sensor_name="feet", command_name="twist", command_threshold=0.1, command_axes=(1,)
    )
    assert torch.all(out == 0.0)

    # Lateral command: the same gate opens and both in-window feet count.
    env = _make_env(command_xyz=(0.0, 0.8, 0.0))
    contact = contact_cfg.build()
    contact.bind(env)
    env.sensors["feet"] = contact
    assert contact._air_state is not None
    contact._air_state.current_air_time = air.clone()
    contact._invalidate_cache()
    out = feet_air_time(
        env, sensor_name="feet", command_name="twist", command_threshold=0.1, command_axes=(1,)
    )
    assert torch.allclose(out, torch.tensor([2.0, 2.0]), atol=1e-6)


# --------------------------------------------------------------------- self_collision_cost


class _FakeSelfContactSensor(SelfContactSensor):
    """Real ``SelfContactSensor`` subclass with hard-coded ``data`` for the reward to read.

    Subclassing keeps the ``isinstance`` check in ``self_collision_cost`` happy without
    going through ``bind``/``update``/Genesis.
    """

    def __init__(
        self,
        *,
        force: torch.Tensor,
        force_history: torch.Tensor | None,
        any_above: torch.Tensor | None = None,
    ) -> None:
        super().__init__(SelfContactSensorCfg(name="self_stub"))
        self._latest_force = force
        # ``found`` is the per-step "any pair above threshold" bool. If the test
        # doesn't pass one explicitly, derive it from ``force > 1.0`` so the
        # historical no-history call sites keep their semantics.
        self._latest_any_above = any_above if any_above is not None else (force > 1.0)
        self._force_history = force_history

    @property
    def data(self) -> SelfContactData:  # type: ignore[override]
        assert self._latest_force is not None
        assert self._latest_any_above is not None
        return SelfContactData(
            force=self._latest_force,
            found=self._latest_any_above,
            force_history=self._force_history,
        )


def test_self_collision_cost_history_counts_any_above_substeps() -> None:
    """4-step bool history with two ``True`` substeps ⇒ cost=2 (reference parity)."""
    env = _make_env(num_envs=1)
    # History stores the per-step "any pair above threshold" bool (the sensor
    # does the thresholding before history accumulation, since Genesis pair
    # indices reshuffle each step).
    history = torch.tensor([[False, True, False, True]])
    env.sensors["self"] = _FakeSelfContactSensor(force=torch.zeros(1), force_history=history)
    out = self_collision_cost(env, sensor_name="self")
    assert torch.allclose(out, torch.tensor([2.0]), atol=1e-6)


def test_self_collision_cost_single_step_when_no_history() -> None:
    """Without ``force_history``, the cost reduces to the current single-step bool."""
    env = _make_env(num_envs=2)
    env.sensors["self"] = _FakeSelfContactSensor(
        force=torch.tensor([0.5, 12.0]),
        force_history=None,
        any_above=torch.tensor([False, True]),
    )
    out = self_collision_cost(env, sensor_name="self")
    assert torch.allclose(out, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_self_collision_cost_rejects_wrong_sensor_type() -> None:
    """Mistakenly wiring a generic ``ContactSensor`` should fail loudly at call time."""
    env = _make_env()
    env.sensors["wrong"] = _StubSensor(torch.zeros(1))  # not a SelfContactSensor
    try:
        self_collision_cost(env, sensor_name="wrong")
    except TypeError as exc:
        assert "SelfContactSensor" in str(exc)
    else:
        raise AssertionError("expected TypeError for non-SelfContactSensor wiring")


# --------------------------------------------------------------------- base hard-constraints


def test_lin_vel_z_l2_penalizes_vertical_velocity_only() -> None:
    env = _make_env(num_envs=2)
    assert env.robot_state.root_lin_vel_b is not None
    env.robot_state.root_lin_vel_b[:, 2] = torch.tensor([0.5, -2.0])
    torch.testing.assert_close(lin_vel_z_l2(env), torch.tensor([0.25, 4.0]))
    # x / y base velocity is ignored — only the vertical component is penalized.
    env.robot_state.root_lin_vel_b[:, :2] = 9.0
    torch.testing.assert_close(lin_vel_z_l2(env), torch.tensor([0.25, 4.0]))


def test_base_height_l2_squared_deviation_from_target() -> None:
    env = _make_env(num_envs=2)
    assert env.robot_state.root_pos is not None
    env.robot_state.root_pos[:, 2] = torch.tensor([0.9, 1.3])
    torch.testing.assert_close(base_height_l2(env, target_height=1.0), torch.tensor([0.01, 0.09]))


def test_alive_bonus_is_constant_ones_per_env() -> None:
    env = _make_env(num_envs=3)
    out = alive_bonus(env)
    assert out.shape == (3,)
    torch.testing.assert_close(out, torch.ones(3))


def test_applied_torque_l2_sums_squared_torque() -> None:
    env = _make_env(num_envs=2)
    env.robot_state.applied_torque = torch.tensor([[1.0, -2.0], [0.0, 3.0]])
    torch.testing.assert_close(applied_torque_l2(env), torch.tensor([5.0, 9.0]))


def test_joint_acc_l2_sums_squared_acceleration() -> None:
    env = _make_env(num_envs=2)
    env.robot_state.joint_acc = torch.tensor([[1.0, -2.0], [0.0, 3.0]])
    torch.testing.assert_close(joint_acc_l2(env), torch.tensor([5.0, 9.0]))


def test_joint_vel_limits_penalizes_excursion_past_limit() -> None:
    env = _make_env(num_envs=2)
    env.robot_state.joint_vel = torch.tensor([[1.0, -3.0], [0.5, 2.0]])
    env.joint_vel_limits = torch.tensor([2.0, 2.0])
    # |q̇| − limit, clamped at 0: env0 = 0 + (3−2) = 1; env1 = 0 + 0 = 0.
    torch.testing.assert_close(joint_vel_limits(env), torch.tensor([1.0, 0.0]))


def test_joint_vel_limits_inert_at_infinite_limit() -> None:
    env = _make_env(num_envs=1)
    env.robot_state.joint_vel = torch.tensor([[100.0, -50.0]])
    env.joint_vel_limits = torch.tensor([float("inf"), float("inf")])
    torch.testing.assert_close(joint_vel_limits(env), torch.tensor([0.0]))
