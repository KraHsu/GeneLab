"""Unit tests for locomotion gait-shaping rewards added in P1.

Tests follow the existing ``test_sensor.py`` pattern: a ``_FakeRewardEnv`` provides just
enough surface (link_names, robot_state, command_manager, sensors) for the reward
functions to read what they need. No Genesis runtime required.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.managers.reward_manager import RewardTermCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.rewards import (
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    feet_clearance,
    feet_slip,
    feet_swing_height,
    self_collision_cost,
    soft_landing,
)
from genelab.sensor import ContactSensorCfg
from genelab.sensor.self_contact import SelfContactData, SelfContactSensor, SelfContactSensorCfg


def _foot_cfg(env, names=("left_foot", "right_foot")) -> SceneEntityCfg:
    """Helper: build a resolved asset_cfg pointing at the named foot links."""
    cfg = SceneEntityCfg(name="robot", link_names=names)
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
    robot: _FakeRewardRobot | None = None

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
    state = _FakeRobotState(link_pos, link_lin_vel_w, link_ang_vel_w)
    command = torch.tensor(command_xyz, dtype=torch.float).unsqueeze(0).expand(num_envs, -1).clone()
    return _FakeRewardEnv(
        num_envs=num_envs, link_names=list(link_names), robot_state=state, command=command
    )


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


def test_angular_momentum_penalty_returns_vector_magnitude() -> None:
    """``L=(3, 4, 0)`` ⇒ ``|L|=5`` per env."""
    env = _make_env()
    env.sensors["L"] = _StubSensor(
        torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    )
    out = angular_momentum_penalty(env, sensor_name="L")
    assert torch.allclose(out, torch.tensor([5.0, 1.0]), atol=1e-6)


# --------------------------------------------------------------------- self_collision_cost


class _FakeSelfContactSensor(SelfContactSensor):
    """Real ``SelfContactSensor`` subclass with hard-coded ``data`` for the reward to read.

    Subclassing keeps the ``isinstance`` check in ``self_collision_cost`` happy without
    going through ``bind``/``update``/Genesis.
    """

    def __init__(self, *, force: torch.Tensor, force_history: torch.Tensor | None) -> None:
        super().__init__(SelfContactSensorCfg(name="self_stub"))
        self._latest_force = force
        self._force_history = force_history

    @property
    def data(self) -> SelfContactData:  # type: ignore[override]
        return SelfContactData(
            force=self._latest_force,  # type: ignore[arg-type]
            found=(self._latest_force > 1.0),  # type: ignore[operator]
            force_history=self._force_history,
        )


def test_self_collision_cost_history_counts_hits_above_threshold() -> None:
    """4-step history with two frames over threshold ⇒ cost=2."""
    env = _make_env(num_envs=1)
    history = torch.tensor([[0.5, 12.0, 3.0, 15.0]])  # (B=1, H=4)
    env.sensors["self"] = _FakeSelfContactSensor(force=torch.zeros(1), force_history=history)
    out = self_collision_cost(env, sensor_name="self", force_threshold=10.0)
    assert torch.allclose(out, torch.tensor([2.0]), atol=1e-6)


def test_self_collision_cost_single_step_when_no_history() -> None:
    """Without ``force_history``, the cost reduces to the current single-step bool."""
    env = _make_env(num_envs=2)
    env.sensors["self"] = _FakeSelfContactSensor(
        force=torch.tensor([0.5, 12.0]), force_history=None
    )
    out = self_collision_cost(env, sensor_name="self", force_threshold=10.0)
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
