"""Terrain-showcase runner: prints pelvis height and heightfield samples."""

from typing import TYPE_CHECKING, cast

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import RayCastData


class TerrainShowcaseRunner(ShowcaseRunner):
    """G1 is dropped at the cell origin and let go.

    The runner dumps the root height and the min/mean/max of the downward ray-cast
    grid so the user can verify each sub-terrain produces the expected elevation
    profile (flat ≈ 0, stairs steps, rough noise, slope linear drift, wave sinusoid).
    """

    task_slug = "terrain"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=20)

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        root = self.log_root()
        ray = cast("RayCastData", env.sensors["pelvis_height"].data)
        rs = env.articulations["robot"].data
        root_z = float(rs.root_pos[0, 2])
        dists = ray.distances[0]
        with (root / "terrain.log").open("a", encoding="utf-8") as fh:
            fh.write(
                f"step={step:04d}  root_z={root_z:.3f}  "
                f"raycast: rays={dists.numel()} min={float(dists.min()):.3f} "
                f"mean={float(dists.mean()):.3f} max={float(dists.max()):.3f}\n"
            )
