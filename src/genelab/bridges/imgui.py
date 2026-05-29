"""ImGui twist bridge: three native sliders for ``(vx, vy, ωz)`` in Genesis's
built-in ImGui overlay.

Drives a 3-component velocity-twist command term (``vx``, ``vy``, ``ωz``). Same
output shape as the keyboard + DearPyGui twist bridges, but uses Genesis 1.0's
built-in ``ImGuiOverlayPlugin`` (enabled via ``SimulationCfg.viewer_imgui=True``
since S1) so no extra GUI dependency is needed.

The bridge:

* In :meth:`on_build` locates the ``ImGuiOverlayPlugin`` on the active viewer and
  registers a ``register_panel(callback)`` hook that draws the three sliders. When
  the scene was constructed without ``viewer_imgui=True`` (or runs headless), the
  bridge logs a warning and becomes a no-op.
* In :meth:`pre_step` writes the slider values into
  ``env.command_manager.get_term(command_name).command`` so the next manager
  ``compute`` pass picks them up.

Caveats:

* The slider state lives on the bridge; the panel callback mutates it and
  ``pre_step`` reads it. ImGui callbacks run on the viewer thread but Python's
  GIL keeps the simple int / float / list assignments atomic without an explicit
  lock.
* Closing the ImGui panel does **not** terminate the play loop — close the
  Genesis viewer or hit Ctrl-C.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from genelab.bridges.base import BridgeCfg

if TYPE_CHECKING:
    from genelab.contracts import EnvContext

_logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ImGuiTwistBridgeCfg(BridgeCfg):
    """Configuration for :class:`ImGuiTwistBridge`.

    Range fields are required (no defaults): an in-viewer slider GUI cannot
    guess sensible limits for an unknown robot, so the caller must set them to
    the twist command term's actual range.
    """

    vx_range: tuple[float, float]
    """Slider min / max for ``vx`` (m/s)."""

    vy_range: tuple[float, float]
    """Slider min / max for ``vy`` (m/s)."""

    wz_range: tuple[float, float]
    """Slider min / max for ``ωz`` (rad/s)."""

    command_name: str = "twist"
    """Name of the command term to drive."""

    panel_section: str = "side"
    """``ImGuiOverlayPlugin.register_panel`` section — ``"side"`` (default) docks the
    sliders into the main control panel; ``"overlay"`` makes a floating window."""

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ImGuiTwistBridge


class ImGuiTwistBridge:
    """ImGui-native slider bridge for a 3-component twist command."""

    def __init__(self, cfg: ImGuiTwistBridgeCfg) -> None:
        self.cfg = cfg
        # Slider state — read by :meth:`pre_step`, mutated by the ImGui panel callback.
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._wz: float = 0.0
        self._enabled: bool = False

    def on_build(self, env: "EnvContext") -> None:
        viewer = getattr(getattr(env.scene, "gs_scene", None), "viewer", None)
        if viewer is None:
            _logger.warning("ImGuiTwistBridge: scene has no viewer; bridge disabled.")
            return

        plugin = _find_imgui_plugin(viewer)
        if plugin is None:
            _logger.warning(
                "ImGuiTwistBridge: no ImGuiOverlayPlugin on the viewer "
                "(set SimulationCfg.viewer_imgui=True to enable it); bridge disabled."
            )
            return

        term = env.command_manager.get_term(self.cfg.command_name)
        pin = getattr(term, "set_external_source", None)
        if callable(pin):
            pin(enabled=True)

        plugin.register_panel(self._draw_panel, section=self.cfg.panel_section)
        self._enabled = True

    def pre_step(self, env: "EnvContext") -> None:
        if not self._enabled:
            return
        term = env.command_manager.get_term(self.cfg.command_name)
        buf = term.command
        desired = torch.tensor([self._vx, self._vy, self._wz], device=buf.device, dtype=buf.dtype)
        buf[:] = desired

    def post_step(self, env: "EnvContext") -> None:
        del env

    def on_close(self, env: "EnvContext") -> None:
        # Genesis tears the ImGui plugin down with the viewer; nothing to release.
        del env

    # ------------------------------------------------------------------ ImGui panel

    def _draw_panel(self, imgui: Any) -> None:
        """Panel callback registered with ``ImGuiOverlayPlugin.register_panel``.

        Called once per ImGui frame with the live ``imgui`` module reference. Each
        ``slider_float`` returns ``(changed, new_value)`` — we accept the value
        unconditionally so dragging the slider drives the twist immediately.
        """
        imgui.text("Twist teleop")
        _, self._vx = imgui.slider_float(
            "vx", float(self._vx), float(self.cfg.vx_range[0]), float(self.cfg.vx_range[1])
        )
        _, self._vy = imgui.slider_float(
            "vy", float(self._vy), float(self.cfg.vy_range[0]), float(self.cfg.vy_range[1])
        )
        _, self._wz = imgui.slider_float(
            "wz", float(self._wz), float(self.cfg.wz_range[0]), float(self.cfg.wz_range[1])
        )
        if imgui.button("Zero"):
            self._vx = 0.0
            self._vy = 0.0
            self._wz = 0.0


def _find_imgui_plugin(viewer: Any) -> Any | None:
    """Find the ``ImGuiOverlayPlugin`` attached to ``viewer``, or return ``None``.

    Duck-types against Genesis 1.0's private ``viewer._viewer_plugins`` list: any
    plugin that exposes ``register_panel`` (the panel-registration API the bridge
    relies on) is acceptable. Returning ``None`` lets
    :meth:`ImGuiTwistBridge.on_build` log a useful warning instead of crashing.
    """
    plugins = getattr(viewer, "_viewer_plugins", None) or []
    for p in plugins:
        if callable(getattr(p, "register_panel", None)):
            return p
    return None
