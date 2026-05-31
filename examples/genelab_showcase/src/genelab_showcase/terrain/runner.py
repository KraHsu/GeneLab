"""Terrain-showcase runner: a spring-suspended G1 toured across the sub-terrains; live scan viz.

A virtual pelvis spring holds the passive G1 upright while this runner walks it from cell to
cell (flat → stairs → rough → slope → wave). The downward pelvis ray-cast scans the surface;
the runner draws the rays in the main scene and renders the hit point cloud + a height
cross-section in two live pyqtgraph windows. Nothing is written to disk.
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from genelab_showcase._viz import LazyQtWindows
from genelab_showcase.runner import ShowcaseRunner, SpringCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from genelab.sensor import RayCastData

_TERRAINS: tuple[str, ...] = ("flat", "stairs", "rough", "slope", "wave")
_DWELL = 150  # control steps spent on each cell (~3 s at 50 Hz)
_VIZ_INTERVAL = 3  # refresh the scan viz every N steps
_MAX_RAY_LINES = 36  # in-scene ray segments (subsampled from the full grid)
_VIEWER_SMOOTHING = 0.9  # camera-follow lag; 1.0 = locked-to-prev, 0.0 = instant snap


class TerrainShowcaseRunner(ShowcaseRunner):
    """Tours a spring-held G1 across the sub-terrains, drawing the downward scan live."""

    task_slug = "terrain"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg, spring=SpringCfg(link_name="pelvis", gravity_comp=1.0))
        self._nodes: list[Any] = []
        self._follow_set = False
        self._qt = LazyQtWindows()
        self._view: Any = None
        self._scatter: Any = None
        self._profile: Any = None

    def _teleport_to(self, env: "ManagerBasedRlEnv", col: int) -> None:
        """Place the robot at the centre of sub-terrain column ``col`` and re-anchor the spring."""
        terrain = env.scene.terrain
        if terrain is None:
            return
        robot = env.articulations["robot"]
        pos = torch.zeros(1, 3, device=env.device)
        pos[0, 0] = terrain.cfg.pos[0] + 0.5 * terrain.cfg.subterrain_size[0]
        pos[0, 1] = terrain.cfg.pos[1] + (col + 0.5) * terrain.cfg.subterrain_size[1]
        stand = float(robot.cfg.init_pos[2])
        pos[0, 2] = terrain.surface_height_at(pos) + stand
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)  # identity orientation
        zero = torch.zeros(1, 3, device=env.device)
        robot.write_root_state(pos, quat, zero, zero, torch.arange(1, device=env.device))
        # Re-anchor the spring so it holds the robot at the new cell instead of dragging it back.
        self._spring_anchor_xy = pos[:, :2].clone()
        self._spring_target_z = float(pos[0, 2])

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if not self._follow_set:
            # Make the main viewer track the robot as it tours the terrains (Genesis updates
            # the follow each render once the entity is set).
            viewer = getattr(env.scene.gs_scene, "viewer", None)
            if viewer is not None:
                viewer.follow_entity(
                    env.articulations["robot"].gs_handle, smoothing=_VIEWER_SMOOTHING
                )
            self._follow_set = True
        if step % _DWELL == 0:
            self._teleport_to(env, (step // _DWELL) % len(_TERRAINS))
        if step % _VIZ_INTERVAL:
            return
        data = cast("RayCastData", env.sensors["pelvis_height"].data)
        self._draw_rays(env, data)
        if self._qt.ensure(self._build_scan_windows, label="terrain scan"):
            self._update_windows(data)

    def _draw_rays(self, env: "ManagerBasedRlEnv", data: "RayCastData") -> None:
        gs_scene = env.scene.gs_scene
        for node in self._nodes:
            gs_scene.clear_debug_object(node)
        self._nodes = []
        starts = data.ray_starts_w[0].detach().cpu().numpy()
        hits = data.hit_pos_w[0].detach().cpu().numpy()
        stride = max(1, len(starts) // _MAX_RAY_LINES)
        for i in range(0, len(starts), stride):
            self._nodes.append(
                gs_scene.draw_debug_line(
                    starts[i], hits[i], radius=0.004, color=(0.3, 0.85, 1.0, 0.5)
                )
            )

    def _build_scan_windows(self, app: Any) -> None:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl

        view = gl.GLViewWidget()
        view.setWindowTitle("terrain hit point cloud")
        view.setCameraPosition(distance=1.8, elevation=28, azimuth=-60)
        grid = gl.GLGridItem()
        grid.setSize(1.4, 1.4, 1.0)
        grid.setSpacing(0.1, 0.1, 0.1)
        view.addItem(grid)
        self._scatter = gl.GLScatterPlotItem(size=8.0)
        view.addItem(self._scatter)
        view.show()
        self._view = view
        plot = pg.plot(title="terrain height cross-section")
        plot.setLabel("bottom", "x under robot (m)")
        plot.setLabel("left", "relative height (m)")
        plot.setYRange(-0.25, 0.25)
        self._profile = plot.plot(pen=pg.mkPen((255, 160, 30), width=2), symbol="o", symbolSize=4)

    def _update_windows(self, data: "RayCastData") -> None:
        hits = data.hit_pos_w[0].detach().cpu().numpy()  # (M, 3)
        # Point cloud, coloured low→high by height and centred on its own footprint.
        z = hits[:, 2]
        span = max(float(z.max() - z.min()), 1e-6)
        t = ((z - z.min()) / span).astype(np.float32)
        colors = np.stack([t, np.full_like(t, 0.5), 1.0 - t, np.ones_like(t)], axis=1)
        centred = hits.copy()
        centred[:, :2] -= centred[:, :2].mean(axis=0)
        self._scatter.setData(pos=centred, color=colors)
        # Height cross-section: the middle row of the square scan grid, height vs local x.
        n = int(round(len(hits) ** 0.5))
        row = hits.reshape(n, n, 3)[n // 2]
        self._profile.setData(row[:, 0] - row[:, 0].mean(), row[:, 2] - row[:, 2].mean())
        self._qt.process()
