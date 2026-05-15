"""RayCast-showcase runner: prints distance histograms for all three patterns."""

from typing import TYPE_CHECKING, cast

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import RayCastData


_SENSOR_NAMES: tuple[str, ...] = ("raycast_grid", "raycast_ring", "raycast_hemi")


class RayCastShowcaseRunner(ShowcaseRunner):
    """Drives the Franka at default pose; prints per-pattern distance summary.

    Output (per dump frame) is appended to ``logs/showcase/raycast/distances.log``:

    ::

        step=NNNN  raycast_grid: rays=81 min=… mean=… max=…
                   raycast_ring: rays=32 min=… mean=… max=…
                   raycast_hemi: rays=128 min=… mean=… max=…
    """

    task_slug = "raycast"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, log_interval=20)

    def _dump(self, env: "ManagerBasedRlEnv", step: int) -> None:
        root = self.log_root()
        lines: list[str] = [f"step={step:04d}"]
        for sensor_name in _SENSOR_NAMES:
            data = cast("RayCastData", env.sensors[sensor_name].data)
            d = data.distances[0]  # first env
            lines.append(
                f"  {sensor_name}: rays={d.numel()} "
                f"min={float(d.min()):.3f} "
                f"mean={float(d.mean()):.3f} "
                f"max={float(d.max()):.3f}"
            )
        with (root / "distances.log").open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
