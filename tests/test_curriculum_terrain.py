"""Curriculum: ``terrain_levels_vel`` promotes / demotes envs by walked distance.

No Genesis runtime — uses a synthetic fake env where ``articulation.data.root_pos`` is
hand-set per case and the spawned terrain importer is mocked (skips spawn but still
runs ``init_per_env_state``).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from genelab.mdp import terrain_levels_vel
from genelab.terrains import FlatPatchCfg, TerrainGeneratorCfg, TerrainImporter


class _MockTerrainHandle:
    def __init__(self, hf: np.ndarray, scale: tuple[float, float]) -> None:
        self.terrain_hf = hf
        self.terrain_scale = np.array(scale, dtype=np.float32)


def _build_terrain(num_envs: int, num_rows: int = 4, num_cols: int = 4) -> TerrainImporter:
    cfg = TerrainGeneratorCfg(
        num_rows=num_rows,
        num_cols=num_cols,
        subterrain_size=(8.0, 8.0),
        sub_terrains={"flat": FlatPatchCfg()},
    )
    importer = TerrainImporter(cfg)
    importer._gs_handle = _MockTerrainHandle(  # pyright: ignore[reportPrivateUsage]
        np.zeros((10, 10), dtype=np.float32), (0.1, 0.005)
    )
    importer.init_per_env_state(num_envs, device="cpu")
    return importer


@dataclass
class _FakeRobotState:
    root_pos: torch.Tensor


_STAND_HEIGHT = 0.8


class _FakeArticulation:
    def __init__(self, root_pos: torch.Tensor) -> None:
        self.data = _FakeRobotState(root_pos=root_pos)
        # ``terrain_levels_vel`` reads ``cfg.init_pos[2]`` to reseat the base at
        # standing height above the sampled surface.
        self.cfg = SimpleNamespace(init_pos=(0.0, 0.0, _STAND_HEIGHT))
        self.written: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def write_root_state(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.written.append(
            (root_pos.clone(), root_quat.clone(), lin_vel.clone(), ang_vel.clone(), env_ids.clone())
        )


class _FakeScene:
    def __init__(self, terrain: TerrainImporter | None) -> None:
        self.terrain = terrain


class _FakeEnv:
    def __init__(self, terrain: TerrainImporter | None, root_pos: torch.Tensor) -> None:
        self.scene = _FakeScene(terrain)
        self.articulation = _FakeArticulation(root_pos)
        self.device = "cpu"


def test_terrain_levels_vel_noop_when_terrain_absent() -> None:
    env = _FakeEnv(terrain=None, root_pos=torch.zeros(2, 3))
    out = terrain_levels_vel(
        env,  # type: ignore[arg-type]
        torch.tensor([0, 1]),
        distance_threshold=2.0,
    )
    assert out.dim() == 0
    assert float(out) == 0.0
    assert env.articulation.written == []


def test_terrain_levels_vel_promotes_when_walked_exceeds_threshold() -> None:
    terrain = _build_terrain(num_envs=3, num_rows=4)
    # Walk env 0 a long way, env 1 a short way, env 2 not at all (still at spawn).
    spawn = terrain.spawn_pos.clone()
    root_pos = spawn.clone()
    root_pos[0, 0] += 5.0  # walked 5m in x → promote (threshold 2m)
    root_pos[1, 0] += 0.2  # walked 0.2m → demote (threshold*0.5 = 1m)
    # env 2: no displacement → walked 0, < 1m → demote (but stays at 0 floor)
    env = _FakeEnv(terrain=terrain, root_pos=root_pos)

    mean_level = terrain_levels_vel(
        env,  # type: ignore[arg-type]
        torch.tensor([0, 1, 2]),
        distance_threshold=2.0,
    )

    assert int(terrain.terrain_levels[0]) == 1  # promoted
    assert int(terrain.terrain_levels[1]) == 0  # demote attempt clamped at 0
    assert int(terrain.terrain_levels[2]) == 0  # already at 0
    # Mean level = 1/3.
    assert abs(float(mean_level) - 1.0 / 3.0) < 1e-6
    # Articulation got one write_root_state call covering all three env_ids.
    assert len(env.articulation.written) == 1
    wrote_pos, _, _, _, wrote_ids = env.articulation.written[0]
    assert torch.equal(wrote_ids, torch.tensor([0, 1, 2]))
    # Env 0's new spawn moved to a row-1 cell; envs 1 and 2 keep their planar cell.
    assert wrote_pos[0, 0] != spawn[0, 0] or wrote_pos[0, 1] != spawn[0, 1]
    assert torch.allclose(wrote_pos[1, :2], spawn[1, :2])
    assert torch.allclose(wrote_pos[2, :2], spawn[2, :2])
    # Base z is reseated at standing height above the (flat, zero) surface, not the
    # bare cell origin (z=0) which would bury the robot. Default spawn_clearance=0.1.
    assert torch.allclose(wrote_pos[:, 2], torch.full((3,), _STAND_HEIGHT + 0.1))


def test_terrain_levels_vel_demotes_partially_walked_envs() -> None:
    terrain = _build_terrain(num_envs=2, num_rows=4)
    # Start both envs already at level 2 so demotion is observable.
    terrain.terrain_levels[:] = 2
    terrain.update_env_origins(torch.tensor([0, 1]))
    spawn = terrain.spawn_pos.clone()
    root_pos = spawn.clone()
    root_pos[0, 0] += 0.3  # below 0.5 * threshold → demote
    root_pos[1, 0] += 4.0  # above threshold → promote
    env = _FakeEnv(terrain=terrain, root_pos=root_pos)

    terrain_levels_vel(
        env,  # type: ignore[arg-type]
        torch.tensor([0, 1]),
        distance_threshold=2.0,
    )

    assert int(terrain.terrain_levels[0]) == 1  # demoted from 2
    assert int(terrain.terrain_levels[1]) == 3  # promoted from 2


def test_terrain_levels_vel_clamps_at_top_row() -> None:
    terrain = _build_terrain(num_envs=1, num_rows=4)
    terrain.terrain_levels[0] = 3  # already at top
    terrain.update_env_origins(torch.tensor([0]))
    spawn = terrain.spawn_pos.clone()
    root_pos = spawn.clone()
    root_pos[0, 0] += 10.0
    env = _FakeEnv(terrain=terrain, root_pos=root_pos)

    terrain_levels_vel(
        env,  # type: ignore[arg-type]
        torch.tensor([0]),
        distance_threshold=2.0,
    )
    assert int(terrain.terrain_levels[0]) == 3  # stays clamped at num_rows - 1


def test_terrain_levels_vel_demote_ratio_overrides() -> None:
    # ratio = 0.9 → walked 1.5m on threshold 2.0 (ratio threshold 1.8) → demote.
    terrain = _build_terrain(num_envs=1, num_rows=4)
    terrain.terrain_levels[0] = 2
    terrain.update_env_origins(torch.tensor([0]))
    spawn = terrain.spawn_pos.clone()
    root_pos = spawn.clone()
    root_pos[0, 0] += 1.5
    env = _FakeEnv(terrain=terrain, root_pos=root_pos)
    terrain_levels_vel(
        env,  # type: ignore[arg-type]
        torch.tensor([0]),
        distance_threshold=2.0,
        demote_ratio=0.9,
    )
    assert int(terrain.terrain_levels[0]) == 1  # demoted because 1.5 < 1.8


def test_terrain_importer_state_blocks_before_init() -> None:
    cfg = TerrainGeneratorCfg(num_rows=2, num_cols=2, sub_terrains={"flat": FlatPatchCfg()})
    importer = TerrainImporter(cfg)
    for attr in ("terrain_levels", "terrain_cols", "spawn_pos"):
        try:
            _ = getattr(importer, attr)
        except RuntimeError as e:
            assert "init_per_env_state" in str(e)
        else:
            raise AssertionError(f"expected RuntimeError before init for {attr}")


def test_init_per_env_state_is_idempotent() -> None:
    terrain = _build_terrain(num_envs=4)
    levels_before = terrain.terrain_levels.clone()
    terrain.init_per_env_state(num_envs=99, device="cpu")  # second call no-op
    assert torch.equal(terrain.terrain_levels, levels_before)
