"""Play-time bridges: ``Bridge`` protocol + reference implementations.

``DearPyGuiCommandBridge`` is not re-exported here because its module performs a
lazy ``import dearpygui`` at first use; importing the symbol unconditionally
would still go through the module body (no ``dearpygui`` import there), but we
keep the surface minimal and ask GUI users to take the longer path:

    from genelab.bridges.dearpygui import DearPyGuiCommandBridgeCfg
"""

from genelab.bridges.base import Bridge, BridgeCfg
from genelab.bridges.keyboard import KeyboardCommandBridge, KeyboardCommandBridgeCfg

__all__ = [
    "Bridge",
    "BridgeCfg",
    "KeyboardCommandBridge",
    "KeyboardCommandBridgeCfg",
]
