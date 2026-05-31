"""Curriculum-showcase runner: a spring-held 5×5 G1 grid + a live terrain-level histogram.

The 25 envs already tile the 5×5 difficulty grid one robot per cell (env ``e`` renders at cell
``(e // 5, e % 5)`` via the multi-env render offset), so no manual placement is needed. The
spring holds each passive G1 upright and a kinematic pin keeps it on its cell. A controlled
curriculum then promotes / demotes each robot's logical level over time; the runner colours
each robot's head marker by its level and draws the live level distribution as a histogram.
Nothing is written to disk.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from genelab_showcase._viz import LazyQtWindows
from genelab_showcase.runner import ShowcaseRunner, SpringCfg

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

_NUM_ROWS = 5  # terrain difficulty levels (0 = smoothest, 4 = roughest)
_NUM_COLS = 5  # robots per difficulty level (one grid column each)
_CYCLE = 60  # control steps between curriculum promote/demote rounds (~1.2 s at 50 Hz)
_VIZ_INTERVAL = 3  # refresh markers + histogram every N steps
_MARKER_HEIGHT = 0.55  # m above the pelvis to float the level marker (over the head)
_MARKER_RADIUS = 0.11
# Level → RGBA, low (blue) to high (red).
_LEVEL_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (0.25, 0.55, 1.0, 0.95),
    (0.20, 0.90, 0.90, 0.95),
    (0.30, 0.90, 0.30, 0.95),
    (1.00, 0.70, 0.10, 0.95),
    (1.00, 0.25, 0.20, 0.95),
)


class CurriculumShowcaseRunner(ShowcaseRunner):
    """A spring-held 5×5 G1 grid whose per-robot curriculum level drives colour + histogram."""

    task_slug = "curriculum"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        # Vertical + uprighting spring only; the planar position is pinned kinematically
        # because a horizontal force spring is unstable for 25 passive G1s on rough terrain.
        super().__init__(
            env_cfg,
            spring=SpringCfg(
                link_name="pelvis",
                gravity_comp=0.95,
                stiffness=8000.0,
                damping=1000.0,
                horizontal=False,
                upright_stiffness=1500.0,
                upright_damping=150.0,
            ),
        )
        self._cell_xy: Any = None  # captured local spawn x/y per env (the cell centre)
        self._levels: Any = None  # torch (num_envs,) logical curriculum level per env
        self._render_offset: Any = None  # (num_envs, 3) per-env render offset → world markers
        self._nodes: list[Any] = []
        self._qt = LazyQtWindows()
        self._bar: Any = None

    def _capture_layout(self, env: "ManagerBasedRlEnv") -> None:
        """Record the per-env cell, starting level (= grid row) and render offset once."""
        n = env.num_envs
        # Local root x/y is the cell centre in each env's own frame (~0); pin it there.
        self._cell_xy = env.articulations["robot"].data.root_pos[:, :2].clone()
        # env e tiles cell (e // _NUM_COLS, e % _NUM_COLS); its difficulty level is the row index.
        self._levels = torch.tensor(
            [e // _NUM_COLS for e in range(n)], dtype=torch.long, device=env.device
        )
        # Markers are debug-drawn in world space, so add the multi-env render offset.
        self._render_offset = np.asarray(env.scene.gs_scene.envs_offset)

    def _pin_xy(self, env: "ManagerBasedRlEnv") -> None:
        """Hold each robot's planar position at its cell (keeps the spring's z / orientation)."""
        robot = env.articulations["robot"]
        rs = robot.data
        pos = rs.root_pos.clone()
        pos[:, :2] = self._cell_xy
        robot.write_root_state(
            pos,
            rs.root_quat.clone(),
            rs.root_lin_vel_w.clone(),
            rs.root_ang_vel_w.clone(),
            torch.arange(env.num_envs, device=env.device),
        )

    def _evolve_levels(self, env: "ManagerBasedRlEnv") -> None:
        """Controlled curriculum: each robot randomly promotes / demotes one level."""
        roll = torch.rand(env.num_envs, device=env.device)
        delta = torch.zeros_like(self._levels)
        delta[roll < 0.3] = 1
        delta[roll > 0.7] = -1
        self._levels = (self._levels + delta).clamp(0, _NUM_ROWS - 1)

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        if self._cell_xy is None:
            self._capture_layout(env)
        self._pin_xy(env)
        if step > 0 and step % _CYCLE == 0:
            self._evolve_levels(env)
        if step % _VIZ_INTERVAL:
            return
        self._draw_markers(env)
        if self._qt.ensure(self._build_hist_window, label="curriculum levels"):
            self._update_hist()

    def _draw_markers(self, env: "ManagerBasedRlEnv") -> None:
        gs_scene = env.scene.gs_scene
        for node in self._nodes:
            gs_scene.clear_debug_object(node)
        self._nodes = []
        # Local root pos + the env's render offset → the robot's position in the viewer.
        pos = env.articulations["robot"].data.root_pos.detach().cpu().numpy() + self._render_offset
        levels = self._levels.detach().cpu().numpy()
        for p, level in zip(pos, levels, strict=True):
            head = p.copy()
            head[2] += _MARKER_HEIGHT
            color = _LEVEL_COLORS[int(level)]
            self._nodes.append(gs_scene.draw_debug_sphere(head, radius=_MARKER_RADIUS, color=color))

    def _build_hist_window(self, app: Any) -> None:
        import pyqtgraph as pg

        plot = pg.plot(title="terrain-level distribution")
        plot.setLabel("bottom", "difficulty level (0 = smoothest)")
        plot.setLabel("left", "robots")
        plot.setYRange(0, 12)
        brushes = [
            pg.mkBrush(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in _LEVEL_COLORS
        ]
        self._bar = pg.BarGraphItem(
            x=list(range(_NUM_ROWS)), height=[0] * _NUM_ROWS, width=0.7, brushes=brushes
        )
        plot.addItem(self._bar)

    def _update_hist(self) -> None:
        counts = torch.bincount(self._levels, minlength=_NUM_ROWS).detach().cpu().numpy()
        self._bar.setOpts(height=counts)
        self._qt.process()
