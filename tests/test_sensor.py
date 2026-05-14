"""Sensor abstraction + obs-noise pipeline. Uses a fake env so Genesis is not required."""

import math
from dataclasses import dataclass
from typing import Any

import torch

from genelab.managers import (
    ObservationGroupCfg,
    ObservationManager,
    ObservationTermCfg,
)
from genelab.mdp.noise import Gnoise, Unoise
from genelab.sensor import (
    BodyVelocitySensorCfg,
    ContactSensorCfg,
    GridPattern,
    HemispherePattern,
    IMUSensorCfg,
    RayCastSensorCfg,
    RingPattern,
    Sensor,
    SensorCfg,
    TerrainHeightSensorCfg,
)


@dataclass
class _ConstSensorCfg(SensorCfg):
    value: float = 1.0

    def build(self) -> "_ConstSensor":
        return _ConstSensor(self)


class _ConstSensor(Sensor[torch.Tensor]):
    def __init__(self, cfg: _ConstSensorCfg) -> None:
        super().__init__(cfg)
        self._cfg_typed = cfg
        self.compute_calls: int = 0

    def _compute_data(self) -> torch.Tensor:
        assert self._env is not None
        self.compute_calls += 1
        return torch.full((self._env.num_envs, 3), self._cfg_typed.value, device=self._env.device)


class _FakeEnv:
    """Just enough surface for sensors and the observation manager."""

    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        self.sensors: dict[str, Sensor[Any]] = {}


def test_sensor_caches_until_invalidated() -> None:
    env = _FakeEnv(num_envs=2)
    sensor = _ConstSensorCfg(name="c", value=1.5).build()
    sensor.bind(env)
    first = sensor.data
    assert first.shape == (2, 3)
    assert torch.allclose(first, torch.full((2, 3), 1.5))
    assert sensor.compute_calls == 1
    _ = sensor.data
    assert sensor.compute_calls == 1
    sensor.update(0.02)
    _ = sensor.data
    assert sensor.compute_calls == 2
    sensor.reset(torch.tensor([0, 1]))
    _ = sensor.data
    assert sensor.compute_calls == 3


def test_unoise_stays_within_bounds_and_is_additive() -> None:
    torch.manual_seed(0)
    data = torch.zeros(1024, 3)
    noisy = Unoise(n_min=-0.2, n_max=0.2).apply(data)
    delta = noisy - data
    assert (delta >= -0.2 - 1e-6).all()
    assert (delta <= 0.2 + 1e-6).all()
    assert delta.abs().mean() > 1e-3


def test_gnoise_has_expected_std() -> None:
    torch.manual_seed(0)
    data = torch.zeros(4096, 1)
    noisy = Gnoise(mean=0.0, std=0.5).apply(data)
    assert abs(noisy.std().item() - 0.5) < 0.05


def test_observation_pipeline_noise_only_when_corruption_enabled() -> None:
    env = _FakeEnv(num_envs=8)

    def const_two(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 3), 2.0, device=_env.device)

    torch.manual_seed(0)
    cfg = {
        "policy": ObservationGroupCfg(
            enable_corruption=True,
            terms={"x": ObservationTermCfg(func=const_two, noise=Unoise(-0.1, 0.1))},
        ),
        "critic": ObservationGroupCfg(
            enable_corruption=False,
            terms={"x": ObservationTermCfg(func=const_two, noise=Unoise(-0.1, 0.1))},
        ),
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.equal(obs["critic"], torch.full((8, 3), 2.0))
    assert not torch.equal(obs["policy"], obs["critic"])
    assert ((obs["policy"] - 2.0).abs() <= 0.1 + 1e-6).all()


def test_observation_pipeline_applies_noise_scale_clip_in_order() -> None:
    env = _FakeEnv(num_envs=1)

    def const_ten(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 1), 10.0, device=_env.device)

    # zero-magnitude noise so we can pin the math: 10 -> +0 noise -> *0.5 scale -> clip to [0, 3]
    cfg = {
        "policy": ObservationGroupCfg(
            enable_corruption=True,
            terms={
                "x": ObservationTermCfg(
                    func=const_ten,
                    noise=Unoise(0.0, 0.0),
                    scale=0.5,
                    clip=(0.0, 3.0),
                )
            },
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.allclose(obs["policy"], torch.tensor([[3.0]]))


def test_observation_manager_skips_corruption_when_noise_unset() -> None:
    env = _FakeEnv(num_envs=2)

    def const_one(_env: Any) -> torch.Tensor:
        return torch.full((_env.num_envs, 2), 1.0, device=_env.device)

    cfg = {
        "g": ObservationGroupCfg(
            enable_corruption=True,
            terms={"x": ObservationTermCfg(func=const_one)},
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    assert torch.equal(obs["g"], torch.full((2, 2), 1.0))


# --------------------------------------------------------------------- BodyVelocitySensor


class _FakeRobotState:
    def __init__(self, num_envs: int, num_links: int, device: str) -> None:
        self.link_quat_w = torch.zeros(num_envs, num_links, 4, device=device)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = torch.zeros(num_envs, num_links, 3, device=device)
        self.link_ang_vel_w = torch.zeros(num_envs, num_links, 3, device=device)


class _FakeRobotEnv:
    def __init__(self, num_envs: int = 2, link_names: tuple[str, ...] = ("pelvis",)) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot_state = _FakeRobotState(num_envs, len(link_names), self.device)


def test_body_velocity_sensor_gyro_rotates_to_body_frame() -> None:
    env = _FakeRobotEnv()
    # 90° rotation about +z: q = (cos45°, 0, 0, sin45°). World ω = +x → body ω = +y after inverse.
    s2 = 2.0**0.5 / 2
    env.robot_state.link_quat_w[:, 0] = torch.tensor([s2, 0.0, 0.0, s2])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    sensor = BodyVelocitySensorCfg(name="g", link_name="pelvis", measure="ang_vel").build()
    sensor.bind(env)
    out = sensor.data
    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)


def test_body_velocity_sensor_velocimeter_lever_arm_cancels() -> None:
    # Pure rotation ω = +z, site offset r = +x → v_site_world = ω × r = +y.
    # Then rotate into body frame (identity quat here) → still +y.
    env = _FakeRobotEnv()
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 0.0, 1.0])
    sensor = BodyVelocitySensorCfg(
        name="v", link_name="pelvis", offset=(1.0, 0.0, 0.0), measure="lin_vel"
    ).build()
    sensor.bind(env)
    out = sensor.data
    assert torch.allclose(out[0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_body_velocity_sensor_velocimeter_with_translation_only() -> None:
    # Pure translation, no rotation: world vel directly returned in body frame (identity quat).
    env = _FakeRobotEnv()
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([2.0, -1.0, 0.5])
    sensor = BodyVelocitySensorCfg(
        name="v", link_name="pelvis", offset=(0.05, 0.0, -0.08), measure="lin_vel"
    ).build()
    sensor.bind(env)
    out = sensor.data
    assert torch.allclose(out[0], torch.tensor([2.0, -1.0, 0.5]), atol=1e-6)


def test_body_velocity_sensor_bias_randomizes_on_reset() -> None:
    torch.manual_seed(0)
    env = _FakeRobotEnv(num_envs=64)
    sensor = BodyVelocitySensorCfg(
        name="g", link_name="pelvis", measure="ang_vel", bias_range=(-0.1, 0.1)
    ).build()
    sensor.bind(env)
    first = sensor.data.clone()
    assert (first.abs() <= 0.1 + 1e-6).all()
    assert first.std() > 0.0  # initial bias should already vary across envs
    sensor.reset(torch.arange(64))
    second = sensor.data
    assert not torch.equal(first, second)


def test_body_velocity_sensor_rejects_unknown_link() -> None:
    env = _FakeRobotEnv()
    sensor = BodyVelocitySensorCfg(name="x", link_name="nonexistent").build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link_name")


# --------------------------------------------------------------------- ContactSensor


class _FakeRobot:
    def __init__(self, contact_force: torch.Tensor) -> None:
        # contact_force shape: (num_envs, num_links, 3)
        self._contact_force = contact_force

    def get_links_net_contact_force(self) -> torch.Tensor:
        return self._contact_force

    def set_contact_force(self, contact_force: torch.Tensor) -> None:
        self._contact_force = contact_force


class _FakeContactEnv:
    def __init__(self, num_envs: int, link_names: tuple[str, ...]) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot = _FakeRobot(torch.zeros(num_envs, len(link_names), 3))


def test_contact_sensor_explicit_link_names_resolves_indices() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("base", "left_foot", "right_foot", "head"))
    sensor = ContactSensorCfg(
        name="feet", link_names=("left_foot", "right_foot"), track_air_time=False
    ).build()
    sensor.bind(env)
    assert sensor.link_names == ["left_foot", "right_foot"]
    env.robot.set_contact_force(
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 50.0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 30.0], [0, 0, 0]],
            ]
        )
    )
    sensor.update(0.02)
    data = sensor.data
    assert data.force_norm.shape == (2, 2)
    assert torch.allclose(data.force_norm[0], torch.tensor([50.0, 0.0]))
    assert torch.allclose(data.force_norm[1], torch.tensor([0.0, 30.0]))
    assert data.found.dtype == torch.bool
    assert torch.equal(data.found, torch.tensor([[True, False], [False, True]]))


def test_contact_sensor_regex_match() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("base", "left_foot", "right_foot"))
    sensor = ContactSensorCfg(name="f", link_names_expr=r"_foot$", track_air_time=False).build()
    sensor.bind(env)
    assert sensor.link_names == ["left_foot", "right_foot"]


def test_contact_sensor_air_time_state_machine() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)

    in_air = torch.zeros(1, 1, 3)
    in_contact = torch.tensor([[[0.0, 0.0, 100.0]]])
    dt = 0.05

    env.robot.set_contact_force(in_air)
    sensor.update(dt)  # tick 1 in air
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[dt]]))
    assert torch.allclose(sensor.data.current_contact_time, torch.tensor([[0.0]]))

    sensor._invalidate_cache()
    sensor.update(dt)  # tick 2 in air
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[2 * dt]]))

    env.robot.set_contact_force(in_contact)
    sensor._invalidate_cache()
    sensor.update(dt)  # landing tick
    d = sensor.data
    assert torch.allclose(d.current_air_time, torch.tensor([[0.0]]))
    assert torch.allclose(d.last_air_time, torch.tensor([[3 * dt]]))
    assert torch.allclose(d.current_contact_time, torch.tensor([[dt]]))

    sensor._invalidate_cache()
    sensor.update(dt)  # continued contact
    assert torch.allclose(sensor.data.current_contact_time, torch.tensor([[2 * dt]]))

    env.robot.set_contact_force(in_air)
    sensor._invalidate_cache()
    sensor.update(dt)  # lift-off
    d = sensor.data
    assert torch.allclose(d.current_contact_time, torch.tensor([[0.0]]))
    assert torch.allclose(d.last_contact_time, torch.tensor([[3 * dt]]))
    assert torch.allclose(d.current_air_time, torch.tensor([[dt]]))


def test_contact_sensor_reset_clears_state_for_env_ids_only() -> None:
    env = _FakeContactEnv(num_envs=2, link_names=("foot",))
    sensor = ContactSensorCfg(name="c", link_names=("foot",), track_air_time=True).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.zeros(2, 1, 3))
    for _ in range(3):
        sensor._invalidate_cache()
        sensor.update(0.1)
    assert torch.allclose(sensor.data.current_air_time, torch.tensor([[0.3], [0.3]]))
    sensor.reset(torch.tensor([0]))
    sensor._invalidate_cache()
    sensor.update(0.1)
    out = sensor.data.current_air_time
    assert torch.allclose(out[0], torch.tensor([0.1]))
    assert torch.allclose(out[1], torch.tensor([0.4]))


def test_contact_sensor_force_threshold_controls_found_bit() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("foot",))
    sensor = ContactSensorCfg(
        name="c", link_names=("foot",), force_threshold=10.0, track_air_time=False
    ).build()
    sensor.bind(env)
    env.robot.set_contact_force(torch.tensor([[[0.0, 0.0, 5.0]]]))
    sensor.update(0.02)
    assert sensor.data.found.item() is False
    env.robot.set_contact_force(torch.tensor([[[0.0, 0.0, 12.0]]]))
    sensor._invalidate_cache()
    sensor.update(0.02)
    assert sensor.data.found.item() is True


def test_contact_sensor_rejects_unresolved_links() -> None:
    env = _FakeContactEnv(num_envs=1, link_names=("a", "b"))
    try:
        ContactSensorCfg(name="x", link_names=("c",)).build().bind(env)
    except ValueError as exc:
        assert "'c'" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link")


# --------------------------------------------------------------------- RayCast / TerrainHeight


class _FakeTerrainState:
    def __init__(
        self,
        num_envs: int,
        num_links: int,
        device: str,
        link_pos: torch.Tensor,
    ) -> None:
        self.link_pos = link_pos
        self.link_quat_w = torch.zeros(num_envs, num_links, 4, device=device)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = torch.zeros(num_envs, num_links, 3, device=device)
        self.link_ang_vel_w = torch.zeros(num_envs, num_links, 3, device=device)


class _FakeTerrainEnv:
    def __init__(
        self,
        num_envs: int,
        link_names: tuple[str, ...],
        link_pos: torch.Tensor,
    ) -> None:
        self.num_envs = num_envs
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot_state = _FakeTerrainState(num_envs, len(link_names), self.device, link_pos)


def test_grid_pattern_generates_expected_count_and_shape() -> None:
    pattern = GridPattern(resolution=0.5, size=(2.0, 1.0), direction=(0.0, 0.0, -1.0))
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 5 * 3
    assert starts.shape == (15, 3)
    assert dirs.shape == (15, 3)
    assert torch.allclose(dirs[0], torch.tensor([0.0, 0.0, -1.0]))
    # Outer corners are at the configured extent.
    assert torch.allclose(starts[..., 0].max(), torch.tensor(1.0))
    assert torch.allclose(starts[..., 0].min(), torch.tensor(-1.0))


def test_raycast_sensor_returns_link_z_distance_on_flat_plane() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 1.2]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
        max_distance=5.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    # 9 rays in a 3x3 grid; each starts at link_pos.z=1.2 and hits z=0 → distance 1.2.
    assert data.distances.shape == (1, 9)
    assert torch.allclose(data.distances, torch.full((1, 9), 1.2))
    assert torch.allclose(data.hit_pos_w[..., 2], torch.zeros(1, 9))
    assert torch.allclose(data.normals_w[..., 2], torch.ones(1, 9))


def test_raycast_sensor_clamps_at_max_distance_when_no_intersection() -> None:
    # Link is below ground → upward-only hit; default backend treats as miss → max_distance.
    link_pos = torch.tensor([[[0.0, 0.0, -1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
        max_distance=3.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert torch.allclose(data.distances, torch.full_like(data.distances, 3.0))


def test_terrain_height_sensor_returns_link_height_on_flat_plane() -> None:
    # Two envs at different heights — height_scan output is constant across rays.
    link_pos = torch.tensor([[[0.0, 0.0, 0.8]], [[1.0, 1.0, 1.5]]])
    env = _FakeTerrainEnv(2, ("torso",), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="h",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
    ).build()
    sensor.bind(env)
    heights = sensor.data
    assert heights.shape == (2, 9)
    assert torch.allclose(heights[0], torch.full((9,), 0.8))
    assert torch.allclose(heights[1], torch.full((9,), 1.5))


def test_terrain_height_sensor_lifecycle_propagates_to_inner_sensor() -> None:
    link_pos = torch.tensor([[[0.0, 0.0, 1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = TerrainHeightSensorCfg(
        name="h",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
    ).build()
    sensor.bind(env)
    _ = sensor.data
    assert sensor._cache_valid is True
    sensor.update(0.02)
    assert sensor._cache_valid is False
    _ = sensor.data
    sensor.reset(torch.tensor([0]))
    assert sensor._cache_valid is False


# --------------------------------------------------------------------- Pattern: Ring / Hemisphere


def test_ring_pattern_full_sweep_does_not_duplicate_endpoint() -> None:
    pattern = RingPattern(
        num_horizontal=8,
        num_vertical=1,
        horizontal_fov_deg=(-180.0, 180.0),
        vertical_fov_deg=(0.0, 0.0),
    )
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 8
    assert starts.shape == (8, 3) and dirs.shape == (8, 3)
    assert torch.allclose(starts, torch.zeros(8, 3))
    # All rays sit in the horizontal plane.
    assert torch.allclose(dirs[..., 2], torch.zeros(8), atol=1e-6)
    # Unit norm.
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(8), atol=1e-6)
    # 360° sweep with N=8 means azimuth step = 45°; first ray points along -x.
    # The closing endpoint (+180°) must not equal the opening (-180°): both map to (-1, 0, 0)
    # in cartesian. Confirm no duplicate by checking the last ray is at +135°, not +180°.
    last_dir = dirs[-1]
    expected_last = torch.tensor(
        [math.cos(math.radians(135.0)), math.sin(math.radians(135.0)), 0.0]
    )
    assert torch.allclose(last_dir, expected_last, atol=1e-6)


def test_ring_pattern_multi_line_lidar_layout() -> None:
    pattern = RingPattern(
        num_horizontal=4,
        num_vertical=3,
        horizontal_fov_deg=(0.0, 90.0),
        vertical_fov_deg=(-10.0, 10.0),
    )
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 12
    assert dirs.shape == (12, 3)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(12), atol=1e-6)
    # Vertical varies slowest (outer index): first 4 rays share the same elevation.
    assert torch.allclose(dirs[0, 2], dirs[3, 2], atol=1e-6)
    assert not torch.isclose(dirs[0, 2], dirs[4, 2], atol=1e-6)


def test_hemisphere_pattern_covers_pole_axis_half_space() -> None:
    pattern = HemispherePattern(num_rays_target=128, pole_axis=(0.0, 0.0, 1.0), polar_fov_deg=90.0)
    starts, dirs = pattern.generate("cpu")
    assert pattern.num_rays() == 128
    assert dirs.shape == (128, 3)
    # All directions are unit length.
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(128), atol=1e-5)
    # On a hemisphere around +z all rays have dot(d, +z) >= 0.
    pole = torch.tensor([0.0, 0.0, 1.0])
    assert (dirs @ pole >= -1e-6).all()
    # Mean direction tracks the pole (Fibonacci lattice is well-distributed).
    assert torch.allclose(dirs.mean(dim=0)[2], torch.tensor(0.5), atol=0.1)


def test_hemisphere_pattern_rotates_to_arbitrary_pole_axis() -> None:
    pattern = HemispherePattern(num_rays_target=64, pole_axis=(1.0, 0.0, 0.0), polar_fov_deg=90.0)
    _, dirs = pattern.generate("cpu")
    pole = torch.tensor([1.0, 0.0, 0.0])
    assert (dirs @ pole >= -1e-6).all()
    # No ray points opposite to the pole.
    assert (dirs @ pole > -0.01).all()


def test_hemisphere_pattern_antipodal_pole_axis_flips_correctly() -> None:
    pattern = HemispherePattern(num_rays_target=32, pole_axis=(0.0, 0.0, -1.0), polar_fov_deg=90.0)
    _, dirs = pattern.generate("cpu")
    # All rays point downward (z <= 0) since the pole is -z.
    assert (dirs[..., 2] <= 1e-6).all()
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_raycast_sensor_accepts_ring_pattern_against_flat_plane() -> None:
    # Sensor at z=1.0 above the flat plane, all rays tilted 30° below horizon.
    link_pos = torch.tensor([[[0.0, 0.0, 1.0]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="lidar",
        link_name="torso",
        pattern=RingPattern(
            num_horizontal=4,
            num_vertical=1,
            horizontal_fov_deg=(-180.0, 180.0),
            vertical_fov_deg=(-30.0, -30.0),
        ),
        max_distance=10.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert data.distances.shape == (1, 4)
    # Distance along a 30°-down ray to hit z=0 from z=1.0 is 1.0 / sin(30°) = 2.0.
    assert torch.allclose(data.distances, torch.full((1, 4), 2.0), atol=1e-5)
    # All hit points sit on the plane z=0.
    assert torch.allclose(data.hit_pos_w[..., 2], torch.zeros(1, 4), atol=1e-5)


def test_raycast_sensor_with_upward_hemisphere_clamps_to_max_distance() -> None:
    # Hemisphere pointing +z: no ray hits the flat plane below.
    link_pos = torch.tensor([[[0.0, 0.0, 0.5]]])
    env = _FakeTerrainEnv(1, ("torso",), link_pos)
    sensor = RayCastSensorCfg(
        name="dome",
        link_name="torso",
        pattern=HemispherePattern(
            num_rays_target=16, pole_axis=(0.0, 0.0, 1.0), polar_fov_deg=80.0
        ),
        max_distance=4.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert data.distances.shape == (1, 16)
    assert torch.allclose(data.distances, torch.full((1, 16), 4.0))


# --------------------------------------------------------------------- IMUSensor


def test_imu_sensor_static_link_reads_negative_body_gravity() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=True).build()
    sensor.bind(env)
    sensor.update(0.02)
    data = sensor.data
    assert data.orientation.shape == (2, 4)
    assert torch.allclose(data.orientation[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # Identity quat → projected gravity is (0, 0, -1).
    assert torch.allclose(data.projected_gravity_b[0], torch.tensor([0.0, 0.0, -1.0]))
    # First step zero-acceleration + gravity bias → lin_acc_b ≈ -g_b = (0, 0, +9.81).
    assert torch.allclose(data.lin_acc_b[0], torch.tensor([0.0, 0.0, 9.81]), atol=1e-5)
    assert torch.allclose(data.ang_acc_b[0], torch.zeros(3), atol=1e-6)


def test_imu_sensor_first_step_after_reset_returns_zero_acc() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    # Prime the sensor with a non-trivial velocity to seed _prev.
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 1.0, 0.0])
    sensor.update(0.02)
    # Now reset env 0 → first-step flag re-armed.
    sensor.reset(torch.tensor([0]))
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([5.0, 0.0, 0.0])
    sensor.update(0.02)
    data = sensor.data
    assert torch.allclose(data.lin_acc_b[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(data.ang_acc_b[0], torch.zeros(3), atol=1e-6)


def test_imu_sensor_finite_diff_linear_acceleration() -> None:
    # Two ticks with constant world-frame velocity ramp v = a * t → lin_acc_b ≈ a.
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    dt = 0.02
    a = torch.tensor([2.0, -1.5, 0.5])
    env.robot_state.link_lin_vel_w[:, 0] = 0.0 * a
    sensor.update(dt)  # primes prev
    env.robot_state.link_lin_vel_w[:, 0] = 1.0 * dt * a
    sensor.update(dt)
    data = sensor.data
    expected = a
    assert torch.allclose(data.lin_acc_b[0], expected, atol=1e-5)


def test_imu_sensor_finite_diff_angular_acceleration() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    sensor.bind(env)
    dt = 0.02
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    sensor.update(dt)  # primes prev
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([1.1, 0.0, 0.0])
    sensor.update(dt)
    data = sensor.data
    assert torch.allclose(data.ang_acc_b[0], torch.tensor([0.1 / dt, 0.0, 0.0]), atol=1e-5)


def test_imu_sensor_lin_acc_bias_randomizes_on_reset() -> None:
    torch.manual_seed(0)
    env = _FakeRobotEnv(num_envs=64)
    sensor = IMUSensorCfg(
        name="imu",
        link_name="pelvis",
        gravity_bias=False,
        bias_range_lin_acc=(-0.5, 0.5),
    ).build()
    sensor.bind(env)
    sensor.update(0.02)
    first = sensor.data.lin_acc_b.clone()
    # First-step acceleration is zero across envs; bias is the sole non-zero contribution.
    assert (first.abs() <= 0.5 + 1e-6).all()
    assert first.std() > 0.0
    sensor.reset(torch.arange(64))
    sensor.update(0.02)
    second = sensor.data.lin_acc_b
    assert not torch.equal(first, second)


def test_imu_sensor_coexists_with_body_velocity_sensor() -> None:
    env = _FakeRobotEnv()
    env.robot_state.link_lin_vel_w[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    env.robot_state.link_ang_vel_w[:, 0] = torch.tensor([0.0, 0.0, 0.5])
    imu = IMUSensorCfg(name="imu", link_name="pelvis", gravity_bias=False).build()
    bv = BodyVelocitySensorCfg(name="bv", link_name="pelvis", measure="lin_vel").build()
    imu.bind(env)
    bv.bind(env)
    imu.update(0.02)
    bv.update(0.02)
    # IMU at first step returns zero accel; BodyVelocity returns body-frame velocity.
    imu_data = imu.data
    bv_data = bv.data
    assert torch.allclose(imu_data.lin_acc_b[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(bv_data[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_imu_sensor_rejects_unknown_link() -> None:
    env = _FakeRobotEnv()
    sensor = IMUSensorCfg(name="x", link_name="nonexistent").build()
    try:
        sensor.bind(env)
    except ValueError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown link_name")
