"""Tactile-showcase runner: a flat plate presses + drags dynamic shapes; dense array → heatmap.

The plate eases down onto three dynamic shapes (two balls, a cube), holds them under a pulsing
press, and oscillates horizontally — the balls roll and the cube slides under it. Each step the
dense ``KinematicDepthSensor`` grid on the plate underside is reshaped to ``(GRID_N, GRID_N)``,
bilinearly upsampled, and shown as a live pyqtgraph heatmap: the imprints brighten with press
force and track sideways as the shapes move. Nothing is written to disk.
"""

import math
from typing import TYPE_CHECKING, Any

import torch

from genelab_showcase._viz import LazyQtWindows
from genelab_showcase.runner import ShowcaseRunner
from genelab_showcase.tactile.env_cfg import DRAG_JOINT, GRID_N, PRESS_JOINT

if TYPE_CHECKING:
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# Vertical press: a steady ``_PRESS_BASE`` contact plus a slow ``_PRESS_AMP`` pulse so the
# heatmap intensity ramps with force without ever fully releasing the shapes (keeps them pinned
# and stable). Horizontal drag: ``_DRAG_AMP`` sideways sweep that rolls / slides the shapes.
# Values are action units (× action_scale 0.18 ≈ metres of joint travel).
_PRESS_BASE = 0.30
_PRESS_AMP = 0.28
_PRESS_PERIOD = 320
_DRAG_AMP = 0.34  # ≈ ±0.06 m of horizontal travel
_DRAG_PERIOD = 240
_EASE_IN = 140  # steps to ramp the plate down from clear into the shapes (gradual emergence)
_DEPTH_FULL = 0.024  # m — penetration that saturates the heatmap colour
_UPSCALE = 128  # bilinear-upsample the GRID_N×GRID_N probe grid to this many px for a smooth map


class TactileShowcaseRunner(ShowcaseRunner):
    """Press + drag a flat dense-tactile plate over dynamic shapes; stream the pressure heatmap."""

    task_slug = "tactile"

    def __init__(self, env_cfg: "ManagerBasedRlEnvCfg") -> None:
        super().__init__(env_cfg)
        self._press_idx: int | None = None
        self._drag_idx: int | None = None
        self._qt = LazyQtWindows()
        self._win: Any = None
        self._img: Any = None

    def _scripted_action(self, env: "ManagerBasedRlEnv", step: int) -> torch.Tensor:
        action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        if self._press_idx is None:
            names = env.articulations["robot"].joint_names
            self._press_idx = names.index(PRESS_JOINT)
            self._drag_idx = names.index(DRAG_JOINT)
        assert self._drag_idx is not None
        ease = min(1.0, step / _EASE_IN)  # gradual descent + emergence at the start
        press_pulse = _PRESS_BASE + _PRESS_AMP * (
            0.5 - 0.5 * math.cos(2.0 * math.pi * step / _PRESS_PERIOD)
        )
        action[:, self._press_idx] = -press_pulse * ease
        action[:, self._drag_idx] = _DRAG_AMP * math.sin(2.0 * math.pi * step / _DRAG_PERIOD) * ease
        return action

    def _build_heatmap_window(self, app: Any) -> None:
        import pyqtgraph as pg

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle("tactile pressure map (penetration depth)")
        win.resize(480, 480)
        vb = win.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)
        self._img = pg.ImageItem(axisOrder="row-major")
        vb.addItem(self._img)
        cmap = pg.colormap.get("inferno")
        if cmap is not None:
            self._img.setLookupTable(cmap.getLookupTable())
        self._img.setLevels((0.0, _DEPTH_FULL))
        win.show()
        self._win = win

    def _post_step(self, env: "ManagerBasedRlEnv", step: int) -> None:
        del step
        depth = env.sensors["pad"].data.depth[0]  # (GRID_N*GRID_N,)
        # Bilinear-upsample the probe grid so the heatmap reads as a smooth pressure field
        # rather than GRID_N×GRID_N hard cells.
        grid = torch.nn.functional.interpolate(
            depth.reshape(1, 1, GRID_N, GRID_N),
            size=(_UPSCALE, _UPSCALE),
            mode="bilinear",
            align_corners=True,
        )[0, 0]
        if not self._qt.ensure(self._build_heatmap_window, label="tactile heatmap"):
            return
        self._img.setImage(grid.detach().cpu().numpy(), autoLevels=False, levels=(0.0, _DEPTH_FULL))
        self._qt.process()
