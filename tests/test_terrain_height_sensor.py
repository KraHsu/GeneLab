"""Ray-cast sensor on top of a real height field.

Two layers of coverage:

* **Bilinear math** — synthetic ``TerrainImporter`` with a hand-built height field,
  driven through a fake env. No Genesis runtime needed; runs everywhere.
* **End-to-end on PyramidStairs** — spawns ``gs.morphs.Terrain`` via :class:`TerrainImporter`,
  feeds the resulting heightfield back to the sensor through a fake env, asserts the
  sensor output matches the direct ``terrain.heightfield`` lookup at the same ``(x, y)``
  within 1 cm. Gated on the ``genesis_runtime`` fixture (skipped on headless CI).
"""

from typing import Any

import numpy as np
import torch

from genelab.sensor import (
    GridPattern,
    RayCastSensorCfg,
    TerrainHeightSensorCfg,
)
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    TerrainGeneratorCfg,
    TerrainImporter,
)


class _FakeScene:
    def __init__(self, terrain: TerrainImporter | None) -> None:
        self.terrain = terrain


class _FakeRobotState:
    def __init__(self, link_pos: torch.Tensor) -> None:
        num_envs, num_links = link_pos.shape[0], link_pos.shape[1]
        self.link_pos = link_pos
        self.link_quat_w = torch.zeros(num_envs, num_links, 4)
        self.link_quat_w[..., 0] = 1.0
        self.link_lin_vel_w = torch.zeros(num_envs, num_links, 3)
        self.link_ang_vel_w = torch.zeros(num_envs, num_links, 3)


class _FakeTerrainEnv:
    def __init__(
        self,
        link_pos: torch.Tensor,
        terrain: TerrainImporter | None,
        link_names: tuple[str, ...] = ("torso",),
    ) -> None:
        self.num_envs = link_pos.shape[0]
        self.device = "cpu"
        self.link_names = list(link_names)
        self.robot_state = _FakeRobotState(link_pos)
        self.scene = _FakeScene(terrain)


class _MockTerrainHandle:
    """Mimics the bits of ``gs.RigidEntity`` that :class:`TerrainImporter` reads post-spawn."""

    def __init__(self, hf: np.ndarray, scale: tuple[float, float]) -> None:
        self.terrain_hf = hf
        self.terrain_scale = np.array(scale, dtype=np.float32)


def _build_mocked_importer(
    hf: np.ndarray, scale: tuple[float, float], pos: tuple[float, float, float]
) -> TerrainImporter:
    """Build a :class:`TerrainImporter` bypassing Genesis spawn (mocked handle)."""
    cfg = TerrainGeneratorCfg(
        num_rows=1, num_cols=1, sub_terrains={"flat": FlatPatchCfg()}, pos=pos
    )
    importer = TerrainImporter(cfg)
    importer._gs_handle = _MockTerrainHandle(hf, scale)  # pyright: ignore[reportPrivateUsage]
    return importer


def test_raycast_falls_back_to_flat_plane_when_scene_has_no_terrain() -> None:
    # Reproduce test_sensor.test_raycast_sensor_returns_link_z_distance_on_flat_plane,
    # but with the new code path (scene exists, terrain is None).
    env = _FakeTerrainEnv(torch.tensor([[[0.0, 0.0, 1.2]]]), terrain=None)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
        max_distance=5.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    assert torch.allclose(data.distances, torch.full((1, 9), 1.2))
    assert torch.allclose(data.hit_pos_w[..., 2], torch.zeros(1, 9))


def test_raycast_samples_constant_height_field() -> None:
    # Constant hf of value 200 with vertical_scale=0.005 -> terrain z = 1.0 everywhere.
    hf = np.full((50, 50), 200.0, dtype=np.float32)
    importer = _build_mocked_importer(hf, scale=(0.1, 0.005), pos=(0.0, 0.0, 0.0))
    env = _FakeTerrainEnv(torch.tensor([[[2.5, 2.5, 2.0]]]), terrain=importer)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=0.5, size=(1.0, 1.0)),
        max_distance=5.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    # Link at z=2.0, terrain at z=1.0 → distance 1.0 everywhere.
    assert torch.allclose(data.distances, torch.full((1, 9), 1.0), atol=1e-5)
    assert torch.allclose(data.hit_pos_w[..., 2], torch.full((1, 9), 1.0), atol=1e-5)


def test_raycast_bilinear_interpolates_linear_ramp() -> None:
    # Linear ramp along x (in grid coords): hf[i, j] = i * 100. With h_scale=0.1 and
    # v_scale=0.01, world height at (x, y) = (x / 0.1) * 100 * 0.01 = 10 * x.
    n = 20
    hf = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        hf[i, :] = i * 100.0
    importer = _build_mocked_importer(hf, scale=(0.1, 0.01), pos=(0.0, 0.0, 0.0))
    # Single ray straight down from a known (x, y, z).
    env = _FakeTerrainEnv(torch.tensor([[[0.5, 0.5, 5.0]]]), terrain=importer)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),  # single ray
        max_distance=10.0,
    ).build()
    sensor.bind(env)
    # At x=0.5 world: terrain z = 10 * 0.5 = 5.0 → ray hits at the link position.
    # Distance from link_z=5.0 to terrain_z=5.0 is 0; current implementation requires
    # link strictly above terrain for a valid hit, so this corner case returns max_dist.
    # Use a position where link_z > terrain_z by a clear margin.
    env = _FakeTerrainEnv(torch.tensor([[[0.5, 0.5, 6.0]]]), terrain=importer)
    sensor.bind(env)
    data = sensor.data
    assert torch.allclose(data.distances, torch.tensor([[1.0]]), atol=1e-5)
    assert torch.allclose(data.hit_pos_w[..., 2], torch.tensor([[5.0]]), atol=1e-5)


def test_raycast_clamps_out_of_bounds_xy_to_border() -> None:
    # Build a height field that has 99 at the (0, 0) corner and 0 elsewhere; verify a
    # ray placed off the negative side of the grid still samples the corner.
    hf = np.zeros((10, 10), dtype=np.float32)
    hf[0, 0] = 99.0
    importer = _build_mocked_importer(hf, scale=(0.1, 0.01), pos=(0.0, 0.0, 0.0))
    env = _FakeTerrainEnv(torch.tensor([[[-5.0, -5.0, 3.0]]]), terrain=importer)
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
        max_distance=5.0,
    ).build()
    sensor.bind(env)
    data = sensor.data
    # Clamped to (0, 0) corner: terrain z = 99 * 0.01 = 0.99. Distance = 3.0 - 0.99.
    assert torch.allclose(data.distances, torch.tensor([[3.0 - 0.99]]), atol=1e-5)


def test_terrain_height_sensor_yields_world_z_offset() -> None:
    # Step height field: half is 0, half is 100 (with v_scale=0.01 → 1.0m step).
    hf = np.zeros((10, 10), dtype=np.float32)
    hf[5:, :] = 100.0
    importer = _build_mocked_importer(hf, scale=(0.1, 0.01), pos=(0.0, 0.0, 0.0))
    # Single ray over the high half (x ≈ 0.6 → grid row 6 → terrain z = 1.0).
    env = _FakeTerrainEnv(torch.tensor([[[0.6, 0.5, 2.5]]]), terrain=importer)
    sensor = TerrainHeightSensorCfg(
        name="h",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
    ).build()
    sensor.bind(env)
    out = sensor.data
    # link_z - hit_z = 2.5 - 1.0 = 1.5
    assert torch.allclose(out, torch.tensor([[1.5]]), atol=1e-5)


def test_raycast_on_real_pyramid_stairs_matches_ground_truth(genesis_runtime: Any) -> None:
    """4×4 PyramidStairs smoke: sensor output vs raw ``terrain_hf`` lookup < 1 cm.

    Probes are placed on exact integer grid cells so the sensor's bilinear interp
    degenerates to a nearest-neighbour pick, making the heightfield array itself the
    independent ground truth.
    """
    gs = genesis_runtime
    h_scale = 0.1
    v_scale = 0.005
    cfg = TerrainGeneratorCfg(
        num_rows=4,
        num_cols=4,
        subterrain_size=(4.0, 4.0),
        horizontal_scale=h_scale,
        vertical_scale=v_scale,
        sub_terrains={
            "stairs": PyramidStairsCfg(step_width=0.5, step_height=-0.1),
        },
        layout=tuple(tuple("stairs" for _ in range(4)) for _ in range(4)),
    )
    importer = TerrainImporter(cfg)
    scene = gs.Scene(show_viewer=False)
    importer.spawn(scene)
    scene.build(n_envs=1)

    hf = importer.heightfield  # ndarray (H, W) of integer steps
    n_rows, n_cols = hf.shape
    # Pick 4 grid indices distributed across the terrain.
    grid_pts = [
        (i, j) for i in (10, 40, 80, 120) for j in (10, 40, 80, 120) if i < n_rows and j < n_cols
    ]
    assert grid_pts, "expected non-empty probe set inside the heightfield"
    # World coords of each grid pick + the raw-truth elevation.
    probe_world = torch.tensor([[i * h_scale, j * h_scale] for i, j in grid_pts])
    truth_z = torch.tensor([float(hf[i, j]) * v_scale for i, j in grid_pts])

    link_z = 5.0
    # Use the sensor's GridPattern facility to fire one ray per probe by placing one
    # ray per pattern cell. Easiest: synthesize a 1D pattern at resolution = step. The
    # pattern centres rays at the link pos and shifts them by xs / ys; build a flat
    # grid of probes around an anchor that maps the pattern points to ``probe_world``.
    # Simpler: drive the sensor's _intersect_world_rays directly with explicit starts.
    env = _FakeTerrainEnv(
        torch.tensor([[[0.0, 0.0, link_z]]]),
        terrain=importer,
        link_names=("torso",),
    )
    sensor = RayCastSensorCfg(
        name="r",
        link_name="torso",
        pattern=GridPattern(resolution=1.0, size=(0.0, 0.0)),
        max_distance=20.0,
    ).build()
    sensor.bind(env)

    starts_w = torch.cat([probe_world, torch.full((len(grid_pts), 1), link_z)], dim=-1).unsqueeze(
        0
    )  # (1, P, 3)
    dirs_w = torch.zeros_like(starts_w)
    dirs_w[..., 2] = -1.0
    distances, hit_pos_w, _ = sensor._intersect_world_rays(starts_w, dirs_w)  # pyright: ignore[reportPrivateUsage]

    # Sensor output is in world-z; truth_z is relative to terrain origin (pos[2] = 0).
    measured_z = hit_pos_w[..., 2].squeeze(0)
    err = (measured_z - truth_z).abs()
    assert err.max().item() < 0.01, f"max error {err.max().item():.4f}m exceeds 1cm tolerance"
    # Distance == link_z - terrain_z for vertical rays.
    expected_dist = (torch.full_like(truth_z, link_z) - truth_z).clamp(max=20.0)
    assert torch.allclose(distances.squeeze(0), expected_dist, atol=1e-3)
