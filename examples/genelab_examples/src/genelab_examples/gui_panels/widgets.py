"""A cookbook of ImGui panels, one tiny function per common widget family.

Each ``*_panel`` function returns a ``callback(imgui)`` — the exact shape
:attr:`SimulationCfg.panels` expects and what Genesis's ``ImGuiOverlayPlugin.register_panel``
calls once per frame. The body of each callback is the copy-paste recipe for that widget:
to add a slider, you write ``imgui.slider_float(...)``; to add a checkbox,
``imgui.checkbox(...)``; nothing GeneLab-specific in between.

``imgui`` is the live ``imgui_bundle.imgui`` module (installed via ``uv sync --extra imgui``).
The immediate-mode convention: widgets that edit a value return ``(changed, new_value)``, so
the idiom is ``_, state.x = imgui.slider_float("x", state.x, lo, hi)`` — keep the returned
value whether or not it changed. Buttons return a plain ``bool`` (clicked this frame).

State lives on a shared :class:`PanelState` so every panel reads/writes the same values; the
callbacks run on the viewer thread, and plain int/float/str/bool assignments are atomic under
the GIL, so no lock is needed (mirrors ``ImGuiTwistBridge``).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A panel callback: receives the live imgui module, draws widgets, returns nothing.
type Panel = Callable[[Any], None]


@dataclass
class PanelState:
    """Mutable values shared by every panel in the cookbook."""

    # sliders / drags
    gain: float = 1.0
    speed: float = 0.5
    count: int = 3
    offset: float = 0.0
    # inputs
    label: str = "hello"
    threshold: float = 0.25
    iterations: int = 10
    # toggles
    enabled: bool = True
    wireframe: bool = False
    mode_index: int = 0  # radio group
    # dropdown
    shape_index: int = 0
    # color (RGB, 0..1)
    color: list[float] = field(default_factory=lambda: [0.2, 0.6, 0.9])
    # layout / output
    progress: float = 0.0
    clicks: int = 0


_MODES = ["idle", "walk", "run"]
_SHAPES = ["box", "sphere", "capsule", "cylinder"]


def text_panel(state: PanelState) -> Panel:
    """Static text: plain, colored, bulleted, and a section heading."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Text")
        imgui.text("Plain label")
        imgui.text_colored(imgui.ImVec4(0.4, 0.9, 0.5, 1.0), "Colored label")
        imgui.bullet_text("Bulleted item")
        imgui.text_wrapped(
            "text_wrapped wraps long strings to the panel width instead of clipping them."
        )

    return draw


def button_panel(state: PanelState) -> Panel:
    """Buttons laid out on one row, plus a click counter."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Buttons")
        if imgui.button("Click me"):
            state.clicks += 1
        imgui.same_line()
        if imgui.small_button("Reset"):
            state.clicks = 0
        imgui.text(f"clicks: {state.clicks}")

    return draw


def slider_panel(state: PanelState) -> Panel:
    """Float / int sliders and a drag (click-drag to scrub a value)."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Sliders")
        _, state.gain = imgui.slider_float("gain", state.gain, 0.0, 2.0)
        _, state.count = imgui.slider_int("count", state.count, 0, 10)
        _, state.offset = imgui.drag_float("offset", state.offset, 0.01, -1.0, 1.0)

    return draw


def input_panel(state: PanelState) -> Panel:
    """Text / float / int input boxes (type a value directly)."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Inputs")
        _, state.label = imgui.input_text("label", state.label)
        _, state.threshold = imgui.input_float("threshold", state.threshold)
        _, state.iterations = imgui.input_int("iterations", state.iterations)

    return draw


def toggle_panel(state: PanelState) -> Panel:
    """Checkboxes and a radio-button group (single choice)."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Toggles")
        _, state.enabled = imgui.checkbox("enabled", state.enabled)
        _, state.wireframe = imgui.checkbox("wireframe", state.wireframe)
        imgui.text("mode:")
        for i, name in enumerate(_MODES):
            imgui.same_line()
            if imgui.radio_button(name, state.mode_index == i):
                state.mode_index = i

    return draw


def select_panel(state: PanelState) -> Panel:
    """A combo (dropdown) selecting from a fixed list."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Dropdown")
        _, state.shape_index = imgui.combo("shape", state.shape_index, _SHAPES)
        imgui.text(f"selected: {_SHAPES[state.shape_index]}")

    return draw


def color_panel(state: PanelState) -> Panel:
    """An RGB color editor with a swatch."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Color")
        _, state.color = imgui.color_edit3("tint", state.color)

    return draw


def layout_panel(state: PanelState) -> Panel:
    """Layout helpers: collapsing header, tree node, progress bar, tab bar."""

    def draw(imgui: Any) -> None:
        imgui.separator_text("Layout")
        # advance a progress bar so it visibly animates
        state.progress = (state.progress + 0.005) % 1.0
        imgui.progress_bar(state.progress, imgui.ImVec2(-1.0, 0.0), f"{state.progress * 100:.0f}%")

        if imgui.collapsing_header("Collapsing section"):
            imgui.text("Hidden until the header is expanded.")

        if imgui.tree_node("Tree node"):
            imgui.bullet_text("child A")
            imgui.bullet_text("child B")
            imgui.tree_pop()

        if imgui.begin_tab_bar("cookbook_tabs"):
            if imgui.begin_tab_item("Tab 1")[0]:
                imgui.text("First tab body")
                imgui.end_tab_item()
            if imgui.begin_tab_item("Tab 2")[0]:
                imgui.text("Second tab body")
                imgui.end_tab_item()
            imgui.end_tab_bar()

    return draw


def _scoped(panel_id: str, panel: Panel) -> Panel:
    """Wrap a panel so all its widget IDs live under a unique ``push_id`` scope.

    GeneLab forwards ``"side"`` panels into Genesis's own control-panel *window*, so an
    unscoped label like ``"Reset"`` would share an ImGui ID with Genesis's built-in buttons —
    and ImGui asserts when two widgets with the same ID both become active. Pushing a
    per-panel id namespaces every widget inside, so labels never clash with the host window or
    with sibling panels. Always do this for a panel you register into a shared overlay.
    """

    def draw(imgui: Any) -> None:
        imgui.push_id(panel_id)
        try:
            panel(imgui)
        finally:
            imgui.pop_id()

    return draw


def cookbook_panels(state: PanelState) -> list[Panel]:
    """Every widget-family panel, ready to drop into ``SimulationCfg.panels``."""
    return [
        _scoped("text", text_panel(state)),
        _scoped("buttons", button_panel(state)),
        _scoped("sliders", slider_panel(state)),
        _scoped("inputs", input_panel(state)),
        _scoped("toggles", toggle_panel(state)),
        _scoped("select", select_panel(state)),
        _scoped("color", color_panel(state)),
        _scoped("layout", layout_panel(state)),
    ]
