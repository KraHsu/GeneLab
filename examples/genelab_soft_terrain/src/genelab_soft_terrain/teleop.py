"""ImGui teleop for the soft-terrain Go1: twist sliders (vx, vy, ωz) + a Reset button.

Extends the stock :class:`genelab.bridges.imgui.ImGuiTwistBridge` (which gives the three
velocity sliders and a Zero button) with a Reset button so you can re-spawn the robot from
the viewer — handy on the analytic soft terrain where ``auto_reset`` is off during play.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from genelab.bridges.imgui import ImGuiTwistBridge, ImGuiTwistBridgeCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


class ImGuiTwistResetBridge(ImGuiTwistBridge):
    """Twist sliders + Zero + a Reset button that re-spawns the robot."""

    def __init__(self, cfg: "ImGuiTwistResetBridgeCfg") -> None:
        super().__init__(cfg)
        self._reset_requested = False

    def pre_step(self, env: "EnvContext") -> None:
        # Act on a Reset click (set on the viewer thread) here on the main thread.
        if self._reset_requested:
            self._reset_requested = False
            env.reset()  # type: ignore[attr-defined]
        super().pre_step(env)

    def _draw_panel(self, imgui: Any) -> None:
        super()._draw_panel(imgui)
        if imgui.button("Reset"):
            self._reset_requested = True


@dataclass(kw_only=True)
class ImGuiTwistResetBridgeCfg(ImGuiTwistBridgeCfg):
    """Config for :class:`ImGuiTwistResetBridge` (twist sliders + Reset)."""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ImGuiTwistResetBridge
