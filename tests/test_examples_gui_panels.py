"""Tests for the GUI-panels example and the ``SimulationCfg.panels`` plumbing.

Covers the registry wiring of ``GeneLab-GUI-Panels-Demo-v0`` and the core
``register_viewer_panels`` helper that forwards panel callbacks to Genesis's ImGui overlay,
using a fake viewer/plugin so no display or ``imgui_bundle`` import is needed.
"""

import logging
from typing import Any

from genelab.registry import ENVS, TASKS, load_extension_module
from genelab.scene import find_imgui_panel_host, register_viewer_panels
from genelab_examples.gui_panels.config import GuiPanelsEnvCfg
from genelab_examples.gui_panels.widgets import PanelState, cookbook_panels


class _FakeOverlay:
    """Stand-in for Genesis's ``ImGuiOverlayPlugin`` — records ``register_panel`` calls."""

    def __init__(self) -> None:
        self.registered: list[tuple[Any, str]] = []

    def register_panel(self, callback: Any, section: str = "side") -> None:
        self.registered.append((callback, section))


class _FakeViewer:
    def __init__(self, plugins: list[Any]) -> None:
        self._viewer_plugins = plugins


def test_gui_panels_task_and_env_are_registered() -> None:
    load_extension_module("genelab_examples.tasks")
    assert "GeneLab-GUI-Panels-Demo-v0" in TASKS
    assert "gui-panels-demo" in ENVS


def test_cookbook_exposes_callables_and_cfg_preloads_them() -> None:
    panels = cookbook_panels(PanelState())
    assert len(panels) == 8
    assert all(callable(p) for p in panels)

    cfg = GuiPanelsEnvCfg()
    assert len(cfg.simulation.panels) == 8
    # A non-empty panel list is what implies the overlay; the demo does not need to also
    # toggle ``viewer_imgui`` by hand.
    assert cfg.simulation.viewer_imgui is False


def test_register_viewer_panels_forwards_to_overlay() -> None:
    overlay = _FakeOverlay()
    viewer = _FakeViewer([object(), overlay])
    panels = cookbook_panels(PanelState())

    assert find_imgui_panel_host(viewer) is overlay
    n = register_viewer_panels(viewer, panels)
    assert n == 8
    assert [section for _, section in overlay.registered] == ["side"] * 8


def test_register_viewer_panels_no_overlay_is_graceful(caplog: Any) -> None:
    viewer = _FakeViewer([object()])  # no plugin exposes register_panel
    assert find_imgui_panel_host(viewer) is None
    with caplog.at_level(logging.WARNING):
        n = register_viewer_panels(
            viewer, cookbook_panels(PanelState()), logger=logging.getLogger()
        )
    assert n == 0
    assert any("no ImGui overlay" in rec.message for rec in caplog.records)


def test_register_viewer_panels_empty_is_noop() -> None:
    # Empty list never inspects the viewer, so ``None`` is a fine argument.
    assert register_viewer_panels(None, []) == 0
