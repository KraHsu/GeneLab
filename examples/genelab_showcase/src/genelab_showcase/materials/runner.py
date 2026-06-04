"""Play runner for the material demos, with an in-viewer ImGui slider.

Builds an :class:`~genelab.scene.InteractiveScene` from a
:class:`~genelab_showcase.materials.scenes.MaterialsDemoCfg` and steps it. When the
viewer is shown, a small ImGui panel adds one material slider (jelly stiffness or
floating-object density) plus an **Apply** button.

Material stiffness / density is baked into the entity at build time — the implicit FEM
solver caches its system matrices, so it cannot be changed mid-step — so **Apply**
rebuilds the scene with the new value and re-drops. Plain re-dropping at the current
value is left to the Genesis overlay's own Play / Pause / **Reset** controls, so this
panel deliberately adds no second "Reset" (which would collide on the ImGui label).

Playback length follows the standard play gate: ``max_steps`` (the ``--max-steps``
flag) is a hard cap; otherwise a shown viewer runs until the window closes and a
headless run stops at ``simulation.steps``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from genelab.cache import ensure_project_cache
from genelab.scene import InteractiveScene

if TYPE_CHECKING:
    from genelab_showcase.materials.scenes import MaterialsDemoCfg


@dataclass
class _Controls:
    """State shared between the ImGui panel (viewer thread) and the step loop."""

    value: float
    apply: bool = False


def _slider_panel(controls: _Controls, label: str, lo: float, hi: float) -> Callable[[Any], None]:
    """Return a ``callback(imgui)`` drawing the material slider and an Apply button."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Material")
        _, controls.value = imgui.slider_float(label, controls.value, lo, hi)
        # Unique label — the Genesis overlay already owns a "Reset" button, so a second
        # widget literally labelled "Reset" would collide on the ImGui ID.
        if imgui.button(f"Apply {label}"):
            controls.apply = True

    return draw


class MaterialsDemoRunner:
    """Steps a material demo, rebuilding it when the Apply button changes the slider."""

    def __init__(
        self,
        cfg: "MaterialsDemoCfg",
        *,
        slider_label: str,
        slider_range: tuple[float, float],
        slider_value: float,
        apply_value: Callable[["MaterialsDemoCfg", float], None],
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.slider_label = slider_label
        self.slider_range = slider_range
        self.apply_value = apply_value
        self._controls = _Controls(value=slider_value)
        self.device = device

    def play(self, *, max_steps: int | None = None) -> None:
        ensure_project_cache()
        sim = self.cfg.simulation
        lo, hi = self.slider_range
        if sim.vis:
            sim.panels = [*sim.panels, _slider_panel(self._controls, self.slider_label, lo, hi)]

        total = 0
        while True:
            self.apply_value(self.cfg, self._controls.value)
            scene = InteractiveScene(sim, self.cfg.scene, device_hint=self.device)
            scene.build()
            self._controls.apply = False

            if max_steps is not None:
                bound: int | None = int(max_steps)
            elif sim.vis:
                bound = None
            else:
                bound = int(sim.steps)

            closed = False
            try:
                while bound is None or total < bound:
                    scene.step()
                    if scene.viewer_closed:
                        closed = True
                        break
                    if self._controls.apply:  # slider changed → rebuild with new value
                        break
                    total += 1
            finally:
                scene.close()

            if closed or (bound is not None and total >= bound):
                break
