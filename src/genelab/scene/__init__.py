"""Scene assembly: groups entities, terrain, and viewer plugins into a runnable scene."""

from genelab.scene.interactive_scene import (
    InteractiveScene,
    find_imgui_panel_host,
    register_viewer_panels,
)

__all__ = ["InteractiveScene", "find_imgui_panel_host", "register_viewer_panels"]
