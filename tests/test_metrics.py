"""Unit tests for ``MetricsManager`` + per-metric functions."""

from dataclasses import dataclass, field
from typing import Any

import torch

from genelab.managers import ActionTermCfg, MetricsManager, MetricsTermCfg
from genelab.managers.action_manager import ActionManager, ActionTerm
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp.metrics import (
    air_time_mean,
    angular_momentum_mean,
    landing_force_mean,
    mean_action_acc,
    peak_height_mean,
    slip_velocity_mean,
)


# --------------------------------------------------------------------- fakes


@dataclass
class _FakeMetricsEnv:
    num_envs: int = 4
    device: str = "cpu"


def _const(value: float):
    """Return a metric function that yields ``(B,)`` filled with ``value`` for every env."""

    def f(env: _FakeMetricsEnv) -> torch.Tensor:
        return torch.full((env.num_envs,), value, device=env.device)

    return f


def _per_env(values: torch.Tensor):
    """Return a metric function yielding a fixed per-env vector."""

    def f(env: _FakeMetricsEnv) -> torch.Tensor:  # noqa: ARG001 - fixed values
        return values

    return f


# --------------------------------------------------------------------- MetricsManager


def test_metrics_manager_accumulates_per_env_sums_across_compute_calls() -> None:
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"const_one": MetricsTermCfg(func=_const(1.0))}
    mgr = MetricsManager(cfg, env)
    for _ in range(5):
        mgr.compute()
    # Five compute() calls × value 1.0 → sum 5.0 per env.
    assert torch.allclose(mgr._episode_sums["const_one"], torch.full((2,), 5.0))
    assert torch.equal(mgr._step_count, torch.full((2,), 5, dtype=torch.long))


def test_metrics_manager_reset_returns_per_term_episode_mean() -> None:
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"x": MetricsTermCfg(func=_const(2.5))}
    mgr = MetricsManager(cfg, env)
    for _ in range(4):
        mgr.compute()
    extras = mgr.reset()
    # Mean = sum/count = (2.5*4)/4 = 2.5 per env; averaged across envs → 2.5.
    assert extras["Episode_Metrics/x"] == 2.5


def test_metrics_manager_reset_clears_only_specified_envs() -> None:
    env = _FakeMetricsEnv(num_envs=4)
    cfg = {"x": MetricsTermCfg(func=_const(1.0))}
    mgr = MetricsManager(cfg, env)
    for _ in range(3):
        mgr.compute()
    mgr.reset(torch.tensor([0, 1]))
    # Reset envs 0,1 → their accumulators zeroed; envs 2,3 untouched.
    assert torch.equal(mgr._episode_sums["x"], torch.tensor([0.0, 0.0, 3.0, 3.0]))
    assert torch.equal(mgr._step_count, torch.tensor([0, 0, 3, 3], dtype=torch.long))


def test_metrics_manager_reset_with_zero_step_count_returns_finite_mean() -> None:
    """An env reset before any compute() call should report 0, not NaN."""
    env = _FakeMetricsEnv(num_envs=2)
    cfg = {"x": MetricsTermCfg(func=_const(7.0))}
    mgr = MetricsManager(cfg, env)
    # Skip compute(); reset immediately — clamp(min=1) guards the divide.
    extras = mgr.reset()
    assert extras["Episode_Metrics/x"] == 0.0


def test_metrics_manager_empty_cfg_is_a_no_op() -> None:
    env = _FakeMetricsEnv()
    mgr = MetricsManager({}, env)
    mgr.compute()  # no-op
    assert mgr.reset() == {}


def test_metrics_manager_distinguishes_per_env_values_at_reset() -> None:
    """Per-env metric values must survive the mean reduction across the env subset."""
    env = _FakeMetricsEnv(num_envs=4)
    per_env_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cfg = {"x": MetricsTermCfg(func=_per_env(per_env_value))}
    mgr = MetricsManager(cfg, env)
    mgr.compute()  # single step → episode_sum equals per_env_value, count=1
    # Reset all → mean across all 4 envs = 2.5.
    extras = mgr.reset()
    assert abs(extras["Episode_Metrics/x"] - 2.5) < 1e-6


# --------------------------------------------------------------------- ActionManager history


class _NoopActionCfg(ActionTermCfg):
    pass


class _NoopActionTerm(ActionTerm):
    def __init__(self, cfg: ActionTermCfg, env) -> None:  # type: ignore[no-untyped-def]
        super().__init__(cfg, env)
        self._dim = 3

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, self._dim)

    def process_actions(self, actions: torch.Tensor) -> None:  # noqa: ARG002
        pass

    def apply_actions(self) -> None:
        pass


def test_action_manager_three_step_history_shifts_correctly() -> None:
    """Three ``process_action`` calls leave the buffers ordered ``a / a_prev / a_prev_prev``."""
    env = _FakeMetricsEnv(num_envs=1)
    cfg = {"noop": _NoopActionCfg(class_type=_NoopActionTerm)}
    mgr = ActionManager(cfg, env)
    v0 = torch.tensor([[0.1, 0.2, 0.3]])
    v1 = torch.tensor([[1.0, 1.0, 1.0]])
    v2 = torch.tensor([[-0.5, 0.0, 0.5]])
    mgr.process_action(v0)
    mgr.process_action(v1)
    mgr.process_action(v2)
    assert torch.allclose(mgr.action, v2)
    assert torch.allclose(mgr.prev_action, v1)
    assert torch.allclose(mgr.prev_prev_action, v0)


def test_action_manager_reset_clears_all_three_buffers() -> None:
    env = _FakeMetricsEnv(num_envs=1)
    cfg = {"noop": _NoopActionCfg(class_type=_NoopActionTerm)}
    mgr = ActionManager(cfg, env)
    mgr.process_action(torch.tensor([[1.0, 2.0, 3.0]]))
    mgr.process_action(torch.tensor([[4.0, 5.0, 6.0]]))
    mgr.process_action(torch.tensor([[7.0, 8.0, 9.0]]))
    mgr.reset()
    assert torch.all(mgr.action == 0.0)
    assert torch.all(mgr.prev_action == 0.0)
    assert torch.all(mgr.prev_prev_action == 0.0)


# --------------------------------------------------------------------- mean_action_acc


class _FakeActionManager:
    """Surface the three history slots without going through ActionManager."""

    def __init__(self, a, prev, prev_prev) -> None:
        self.action = a
        self.prev_action = prev
        self.prev_prev_action = prev_prev


@dataclass
class _FakeActionAccEnv:
    action_manager: _FakeActionManager
    num_envs: int = 1
    device: str = "cpu"


def test_mean_action_acc_zero_when_history_is_constant() -> None:
    """A perfectly constant action stream has zero second derivative ⇒ metric=0."""
    a = torch.ones(2, 4)
    env = _FakeActionAccEnv(action_manager=_FakeActionManager(a, a, a), num_envs=2)
    out = mean_action_acc(env)
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_mean_action_acc_known_finite_difference() -> None:
    """``a − 2·a_prev + a_prev_prev`` averaged over the action dim, per env."""
    # Env 0: action=[1,1,1], prev=[0,0,0], prev_prev=[0,0,0] → accel=[1,1,1] → mean=1.
    # Env 1: action=[0,0,2], prev=[0,1,1], prev_prev=[0,0,0] → accel=[0,-2,0] → mean=2/3.
    a = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 2.0]])
    p = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pp = torch.zeros(2, 3)
    env = _FakeActionAccEnv(action_manager=_FakeActionManager(a, p, pp), num_envs=2)
    out = mean_action_acc(env)
    expected = torch.tensor([1.0, 2.0 / 3.0])
    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------- gait metric fakes
#
# Lightweight sensor + robot_state stand-ins so the 5 mjlab-parity metrics
# (``angular_momentum_mean``, ``air_time_mean``, ``slip_velocity_mean``,
# ``landing_force_mean``, ``peak_height_mean``) can be exercised without
# spinning up Genesis. Each metric reads either a contact sensor's data,
# the robot_state link tensors, or the root-angmom sensor's data.


@dataclass
class _FakeContactData:
    current_air_time: torch.Tensor
    found: torch.Tensor
    first_contact: torch.Tensor
    force_norm: torch.Tensor


@dataclass
class _FakeRobotStateForMetrics:
    link_pos: torch.Tensor
    link_quat_w: torch.Tensor
    link_lin_vel_w: torch.Tensor
    link_ang_vel_w: torch.Tensor


@dataclass
class _FakeAngmomSensor:
    data: torch.Tensor  # (B, 3)


@dataclass
class _FakeMetricsEnvWithSensors:
    num_envs: int
    device: str = "cpu"
    sensors: dict[str, Any] = field(default_factory=dict)
    robot_state: Any = None
    link_names: list[str] = field(default_factory=lambda: ["base", "left_foot", "right_foot"])


def _make_contact_sensor(
    *,
    current_air_time: torch.Tensor,
    found: torch.Tensor,
    first_contact: torch.Tensor,
    force_norm: torch.Tensor,
) -> Any:
    """Build a real-ContactSensor-typed object whose ``data`` is our fake.

    The metric helpers use ``isinstance(env.sensors[name], ContactSensor)``, so the
    object must be the real class. We bypass ``__init__`` to skip Genesis setup
    and populate ``_cached_data`` directly — :class:`Sensor.data` reads from there
    when it's already populated (sensor.py:72-76).
    """
    from genelab.sensor.contact import ContactSensor

    sensor = ContactSensor.__new__(ContactSensor)
    sensor._cached_data = _FakeContactData(  # type: ignore[attr-defined]
        current_air_time=current_air_time,
        found=found,
        first_contact=first_contact,
        force_norm=force_norm,
    )
    sensor._cache_valid = True  # type: ignore[attr-defined]
    return sensor


# --------------------------------------------------------------------- angular_momentum_mean


def test_angular_momentum_mean_returns_norm_per_env() -> None:
    """Per-env ``||L||_2``. mjlab logs ``mean(magnitude)``; per-env value is the magnitude."""
    env = _FakeMetricsEnvWithSensors(num_envs=2)
    # Env 0: L = (3, 4, 0) → ||L|| = 5. Env 1: L = (0, 0, 12) → 12.
    env.sensors["angmom"] = _FakeAngmomSensor(
        data=torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    )
    out = angular_momentum_mean(env, sensor_name="angmom")
    assert torch.allclose(out, torch.tensor([5.0, 12.0]), atol=1e-6)


# --------------------------------------------------------------------- air_time_mean


def test_air_time_mean_averages_only_over_in_air_feet() -> None:
    """``sum(air_time × in_air) / num_in_air`` across (env, foot)."""
    env = _FakeMetricsEnvWithSensors(num_envs=2)
    # Env 0: foot air times (0.2, 0.0) — only first foot is in air.
    # Env 1: foot air times (0.4, 0.6) — both in air.
    # Total: (0.2 + 0.4 + 0.6) = 1.2, n_in_air = 3 → mean = 0.4.
    air = torch.tensor([[0.2, 0.0], [0.4, 0.6]])
    sensor = _make_contact_sensor(
        current_air_time=air,
        found=torch.zeros(2, 2, dtype=torch.bool),
        first_contact=torch.zeros(2, 2, dtype=torch.bool),
        force_norm=torch.zeros(2, 2),
    )
    env.sensors["feet"] = sensor
    out = air_time_mean(env, sensor_name="feet")
    assert out.shape == (2,)
    assert torch.allclose(out, torch.full((2,), 0.4), atol=1e-6)


def test_air_time_mean_zero_when_no_feet_in_air() -> None:
    """All grounded → ``num_in_air`` clamps to 1, mean is 0 (no contributors)."""
    env = _FakeMetricsEnvWithSensors(num_envs=1)
    sensor = _make_contact_sensor(
        current_air_time=torch.zeros(1, 2),
        found=torch.ones(1, 2, dtype=torch.bool),
        first_contact=torch.zeros(1, 2, dtype=torch.bool),
        force_norm=torch.zeros(1, 2),
    )
    env.sensors["feet"] = sensor
    out = air_time_mean(env, sensor_name="feet")
    assert torch.all(out == 0.0)


# --------------------------------------------------------------------- landing_force_mean


def test_landing_force_mean_averages_over_first_contact_feet() -> None:
    """``sum(force_norm × first_contact) / num_landings``."""
    env = _FakeMetricsEnvWithSensors(num_envs=2)
    # Env 0: foot 0 lands with |F|=20, foot 1 not landing (|F|=5 ignored).
    # Env 1: foot 0 not landing (|F|=10 ignored), foot 1 lands with |F|=30.
    # Total = 20 + 30 = 50, num_landings = 2 → mean = 25.
    sensor = _make_contact_sensor(
        current_air_time=torch.zeros(2, 2),
        found=torch.ones(2, 2, dtype=torch.bool),
        first_contact=torch.tensor([[True, False], [False, True]]),
        force_norm=torch.tensor([[20.0, 5.0], [10.0, 30.0]]),
    )
    env.sensors["feet"] = sensor
    out = landing_force_mean(env, sensor_name="feet")
    assert torch.allclose(out, torch.full((2,), 25.0), atol=1e-6)


# --------------------------------------------------------------------- slip_velocity_mean


def test_slip_velocity_mean_averages_grounded_xy_speed() -> None:
    """``sum(||v_xy|| × in_contact) / num_in_contact``."""
    env = _FakeMetricsEnvWithSensors(num_envs=2)
    # Build robot_state with foot links at index 1, 2 (matching default link_names).
    num_links = len(env.link_names)
    link_pos = torch.zeros(2, num_links, 3)
    link_quat_w = torch.zeros(2, num_links, 4)
    link_quat_w[..., 0] = 1.0  # identity quat
    # Env 0 foot velocities: foot 0 = (3, 4, 0) → ||xy||=5; foot 1 = (0, 0, 0) → 0.
    # Env 1 foot velocities: foot 0 = (1, 0, 0) → 1; foot 1 = (0, 2, 0) → 2.
    link_lin_vel_w = torch.zeros(2, num_links, 3)
    link_lin_vel_w[0, 1] = torch.tensor([3.0, 4.0, 0.0])
    link_lin_vel_w[1, 1] = torch.tensor([1.0, 0.0, 0.0])
    link_lin_vel_w[1, 2] = torch.tensor([0.0, 2.0, 0.0])
    link_ang_vel_w = torch.zeros(2, num_links, 3)
    env.robot_state = _FakeRobotStateForMetrics(
        link_pos=link_pos,
        link_quat_w=link_quat_w,
        link_lin_vel_w=link_lin_vel_w,
        link_ang_vel_w=link_ang_vel_w,
    )
    # Only contacting feet: env 0 foot 0; env 1 both feet.
    sensor = _make_contact_sensor(
        current_air_time=torch.zeros(2, 2),
        found=torch.tensor([[True, False], [True, True]]),
        first_contact=torch.zeros(2, 2, dtype=torch.bool),
        force_norm=torch.zeros(2, 2),
    )
    env.sensors["feet"] = sensor
    asset_cfg = SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot"))
    asset_cfg.resolve(env)  # type: ignore[arg-type]
    out = slip_velocity_mean(env, sensor_name="feet", asset_cfg=asset_cfg)
    # Sum = 5 + 1 + 2 = 8, num_in_contact = 3 → mean ≈ 2.667.
    assert torch.allclose(out, torch.full((2,), 8.0 / 3.0), atol=1e-6)


# --------------------------------------------------------------------- peak_height_mean


def test_peak_height_mean_samples_apex_on_landing_and_resets_buffer() -> None:
    """Accumulates per-foot peak while airborne; samples ``peak × first_contact`` at landing."""
    env = _FakeMetricsEnvWithSensors(num_envs=1)
    num_links = len(env.link_names)
    link_quat_w = torch.zeros(1, num_links, 4)
    link_quat_w[..., 0] = 1.0
    link_lin_vel_w = torch.zeros(1, num_links, 3)
    link_ang_vel_w = torch.zeros(1, num_links, 3)
    env.robot_state = _FakeRobotStateForMetrics(
        link_pos=torch.zeros(1, num_links, 3),
        link_quat_w=link_quat_w,
        link_lin_vel_w=link_lin_vel_w,
        link_ang_vel_w=link_ang_vel_w,
    )
    asset_cfg = SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot"))
    asset_cfg.resolve(env)  # type: ignore[arg-type]

    # Build the class-based metric via the manager's instantiate hook surface.
    cfg = MetricsTermCfg(
        func=peak_height_mean,
        params={"sensor_name": "feet", "asset_cfg": asset_cfg},
    )
    # Manual instantiation matching MetricsManager's pattern.
    metric = peak_height_mean(cfg, env)  # type: ignore[arg-type]

    # Frame 1: foot 0 mid-air at z=0.08; not landing. Peak buffer fills.
    env.robot_state.link_pos[0, 1, 2] = 0.08
    sensor = _make_contact_sensor(
        current_air_time=torch.tensor([[0.1, 0.0]]),
        found=torch.tensor([[False, True]]),
        first_contact=torch.tensor([[False, False]]),
        force_norm=torch.zeros(1, 2),
    )
    env.sensors["feet"] = sensor
    out1 = metric(env, sensor_name="feet", asset_cfg=asset_cfg)
    # No landings yet → numerator 0 / clamp(1) = 0.
    assert torch.all(out1 == 0.0)

    # Frame 2: foot 0 still airborne, climbs to z=0.12; foot 1 stays grounded.
    env.robot_state.link_pos[0, 1, 2] = 0.12
    sensor = _make_contact_sensor(
        current_air_time=torch.tensor([[0.15, 0.0]]),
        found=torch.tensor([[False, True]]),
        first_contact=torch.tensor([[False, False]]),
        force_norm=torch.zeros(1, 2),
    )
    env.sensors["feet"] = sensor
    out2 = metric(env, sensor_name="feet", asset_cfg=asset_cfg)
    assert torch.all(out2 == 0.0)

    # Frame 3: foot 0 lands. Peak should still equal 0.12 (it stops accumulating on contact).
    env.robot_state.link_pos[0, 1, 2] = 0.0
    sensor = _make_contact_sensor(
        current_air_time=torch.tensor([[0.0, 0.0]]),
        found=torch.tensor([[True, True]]),
        first_contact=torch.tensor([[True, False]]),
        force_norm=torch.zeros(1, 2),
    )
    env.sensors["feet"] = sensor
    out3 = metric(env, sensor_name="feet", asset_cfg=asset_cfg)
    # Landing peak 0.12 averaged over 1 landing → 0.12.
    assert torch.allclose(out3, torch.full((1,), 0.12), atol=1e-6)

    # Frame 4: post-landing — peak buffer was reset, no current landing → metric=0.
    sensor = _make_contact_sensor(
        current_air_time=torch.zeros(1, 2),
        found=torch.tensor([[True, True]]),
        first_contact=torch.tensor([[False, False]]),
        force_norm=torch.zeros(1, 2),
    )
    env.sensors["feet"] = sensor
    out4 = metric(env, sensor_name="feet", asset_cfg=asset_cfg)
    assert torch.all(out4 == 0.0)
