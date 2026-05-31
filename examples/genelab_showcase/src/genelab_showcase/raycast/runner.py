"""RayCast-showcase runner: Franka joint-1 sweep + a live point-cloud window.

This runner drives the sweep and opens a separate window that visualises **only** the ray-cast
hit point cloud as a GPU-accelerated 3D scatter (pyqtgraph ``GLScatterPlotItem``, one colour
per sensor). The main scene shows no ray-cast debug; nothing is written to disk.
"""

import math
from typing import TYPE_CHECKING, Any

import torch

from genelab_showcase.runner import ShowcaseRunner

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# (sensor name, RGBA colour in 0-1) for the point-cloud scatter.
_SENSORS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("mesh_grid", (1.0, 0.5, 0.06, 1.0)),
    ("mesh_cone", (0.9, 0.13, 0.6, 1.0)),
)


class RayCastShowcaseRunner(ShowcaseRunner):
    """Sweep Franka joint 1 so the gripper-forward ray-casts scan the front props; render the
    hit point cloud live in a dedicated pyqtgraph window."""

    task_slug = "raycast"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg)
        self._joint1_idx: int | None = None
        self._cloud_interval = 3  # refresh the point cloud every N steps
        self._inited = False
        self._app: Any = None
        self._view: Any = None
        self._scatter: Any = None
        self._np: Any = None

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        if self._joint1_idx is None:
            self._joint1_idx = env.articulations["robot"].joint_names.index("joint1")
        # Slow sweep (~6 s period) so the point cloud drifts across the props.
        phase = 2.0 * math.pi * step / 300.0
        action[:, self._joint1_idx] = math.sin(phase) * 0.5
        return action

    def _ensure_cloud_window(self) -> bool:
        """Lazily open the GPU 3D-scatter window. Returns False if pyqtgraph/GL is unavailable."""
        if self._inited:
            return self._scatter is not None
        self._inited = True
        try:
            import numpy as np
            import pyqtgraph as pg
            import pyqtgraph.opengl as gl

            self._np = np
            self._app = pg.mkQApp("ray-cast point cloud")
            view = gl.GLViewWidget()
            view.setWindowTitle("ray-cast hit point cloud")
            view.opts["center"] = pg.Vector(0.45, 0.0, 0.1)
            view.setCameraPosition(distance=1.1, elevation=22, azimuth=-75)
            grid = gl.GLGridItem()
            grid.setSize(1.0, 1.0, 1.0)
            grid.setSpacing(0.1, 0.1, 0.1)
            view.addItem(grid)
            self._scatter = gl.GLScatterPlotItem(size=8.0, pxMode=True)
            view.addItem(self._scatter)
            view.show()
            self._view = view
            return True
        except Exception as exc:  # noqa: BLE001 - viz is best-effort, never break the sim
            print(f"raycast showcase: point-cloud window unavailable ({exc})")
            self._cloud_interval = 1_000_000  # stop retrying
            return False

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if step % self._cloud_interval or not self._ensure_cloud_window():
            return
        np = self._np
        positions = []
        colors = []
        for name, color in _SENSORS:
            pts = env.sensors[name].data.hit_pos_w[0].detach().cpu().numpy()
            positions.append(pts)
            colors.append(np.tile(color, (len(pts), 1)))
        self._scatter.setData(
            pos=np.concatenate(positions), color=np.concatenate(colors).astype(np.float32)
        )
        self._app.processEvents()
