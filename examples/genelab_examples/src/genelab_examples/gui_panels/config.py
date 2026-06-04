"""Config for the ImGui panel cookbook demo."""

from dataclasses import dataclass, field

from genelab.configs import SimulationCfg

from genelab_examples.gui_panels.widgets import PanelState, cookbook_panels


def _default_simulation() -> SimulationCfg:
    # ``panels`` is the only GeneLab-specific step: hand the overlay a list of
    # ``callback(imgui)`` functions. A non-empty list also implies the ImGui overlay, so
    # ``viewer_imgui`` does not need to be set separately. The shared ``PanelState`` keeps
    # every panel's widgets editing the same values.
    return SimulationCfg(
        vis=True,
        steps=100_000,  # large: a viewer run ends when the window is closed, not at a step cap
        dt=0.01,
        substeps=2,
        panels=cookbook_panels(PanelState()),
    )


@dataclass
class GuiPanelsEnvCfg:
    """Environment cfg whose ``simulation.panels`` is preloaded with the widget cookbook."""

    simulation: SimulationCfg = field(default_factory=_default_simulation)
