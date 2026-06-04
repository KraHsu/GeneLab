"""Play-time bridges: ``Bridge`` protocol + reference implementations.

Two twist-teleop bridges ship here: :class:`KeyboardTwistBridge` (Genesis's built-in
keybinds, zero extra deps) and :class:`ImGuiTwistBridge` (sliders in the Genesis viewer's
ImGui overlay, needs the ``imgui`` extra). For a **display-only** viewer panel that does not
drive the policy every step, prefer :attr:`SimulationCfg.panels` over a bridge — see
:func:`genelab.scene.register_viewer_panels`.
"""

from genelab.bridges.base import Bridge, BridgeCfg
from genelab.bridges.imgui import ImGuiTwistBridge, ImGuiTwistBridgeCfg
from genelab.bridges.keyboard import KeyboardTwistBridge, KeyboardTwistBridgeCfg

__all__ = [
    "Bridge",
    "BridgeCfg",
    "ImGuiTwistBridge",
    "ImGuiTwistBridgeCfg",
    "KeyboardTwistBridge",
    "KeyboardTwistBridgeCfg",
]
