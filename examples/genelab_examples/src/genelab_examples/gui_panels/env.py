"""Runnable standalone scene that renders the ImGui panel cookbook.

The scene itself is intentionally trivial — a box dropped on a plane — so the focus stays on
the GUI. The only GeneLab-specific line is :func:`genelab.scene.register_viewer_panels`, which
forwards ``simulation.panels`` to Genesis's ImGui overlay; everything else is plain Genesis.
"""

from typing import Any, cast

from genelab_examples.gui_panels.config import GuiPanelsEnvCfg


def _genesis() -> Any:
    # Genesis is optional at import time and ships no complete type stubs.
    import genesis as gs  # pyright: ignore[reportMissingImports, reportMissingTypeStubs]

    return cast(Any, gs)


class GuiPanelsDemoEnv:
    """Standalone Genesis scene that demonstrates adding custom ImGui panels."""

    def __init__(self, cfg: GuiPanelsEnvCfg | None = None) -> None:
        self.cfg = cfg or GuiPanelsEnvCfg()

    def play(self) -> None:
        from genelab.cache import ensure_project_cache
        from genelab.scene import register_viewer_panels

        ensure_project_cache()

        sim = self.cfg.simulation
        if not sim.vis:
            raise SystemExit(
                "the GUI panels demo needs the viewer — pass --vis "
                "(or set env.simulation.vis=true)."
            )

        gs = _genesis()
        gs.init(backend=gs.gpu if sim.gpu else gs.cpu, precision="32")
        try:
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim.dt, substeps=sim.substeps),
                viewer_options=gs.options.ViewerOptions(
                    camera_pos=(2.6, 0.0, 1.6),
                    camera_lookat=(0.0, 0.0, 0.4),
                    camera_fov=40,
                    refresh_rate=sim.render_fps or 60,
                    enable_gui=True,  # the ImGui overlay that hosts our panels
                ),
                show_viewer=True,
            )
            scene.add_entity(gs.morphs.Plane())
            scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.8)))

            # The whole of GeneLab's GUI wrapping: forward the panel callbacks to the overlay.
            n = register_viewer_panels(scene.viewer, sim.panels)
            scene.build()
            print(f"GUI panels demo: registered {n} ImGui panel(s). Close the viewer to exit.")

            viewer_closed = gs.GenesisException
            for _ in range(max(1, sim.steps)):
                try:
                    scene.step()
                except viewer_closed as exc:
                    if str(exc) == "Viewer closed.":
                        break
                    raise
        finally:
            gs.destroy()


def create_gui_panels_env(cfg: GuiPanelsEnvCfg | None = None) -> GuiPanelsDemoEnv:
    return GuiPanelsDemoEnv(cfg)
