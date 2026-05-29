"""Unit tests for :class:`genelab.bridges.ImGuiTwistBridge`.

The bridge writes into the named twist command term via the live ``env``; the
ImGui plugin is opaque to it apart from ``register_panel``. Tests use a fake
viewer / plugin / env that record every call so we can assert (a) the bridge
becomes a no-op when the plugin is missing, (b) registered panel callbacks
mutate the slider state, and (c) ``pre_step`` writes the latest state into the
command buffer.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.bridges.imgui import ImGuiTwistBridge, ImGuiTwistBridgeCfg  # noqa: E402


class _FakeImGuiPlugin:
    """Stand-in for ``ImGuiOverlayPlugin``.

    The bridge calls ``register_panel(callback, section=...)``; the test stores
    the callback so it can drive it manually as ImGui would each frame.
    """

    def __init__(self) -> None:
        self.panels: list[tuple[Any, str]] = []

    def register_panel(self, callback: Any, section: str = "side") -> None:
        self.panels.append((callback, section))


class _FakeViewer:
    def __init__(self, plugin: Any | None) -> None:
        self._viewer_plugins = [plugin] if plugin is not None else []


class _FakeGsScene:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer


class _FakeScene:
    def __init__(self, viewer: Any) -> None:
        self.gs_scene = _FakeGsScene(viewer)


class _FakeCommandTerm:
    def __init__(self, num_envs: int = 2) -> None:
        self.command = torch.zeros(num_envs, 3)
        self.external_source_calls: list[bool] = []

    def set_external_source(self, *, enabled: bool) -> None:
        self.external_source_calls.append(enabled)


class _FakeCommandManager:
    def __init__(self, terms: dict[str, _FakeCommandTerm]) -> None:
        self._terms = terms

    def get_term(self, name: str) -> _FakeCommandTerm:
        return self._terms[name]


class _FakeEnv:
    def __init__(self, plugin: Any | None, term: _FakeCommandTerm) -> None:
        self.scene = _FakeScene(_FakeViewer(plugin))
        self.command_manager = _FakeCommandManager({"twist": term})


class _FakeImguiApi:
    """Stand-in for the live ``imgui`` module the panel callback uses."""

    def __init__(self, vx: float, vy: float, wz: float, *, zero_clicked: bool = False) -> None:
        self._vx = vx
        self._vy = vy
        self._wz = wz
        self._zero_clicked = zero_clicked
        self.slider_calls: list[tuple[str, float, float, float]] = []

    def text(self, msg: str) -> None:
        del msg

    def slider_float(self, label: str, current: float, lo: float, hi: float) -> tuple[bool, float]:
        self.slider_calls.append((label, current, lo, hi))
        if label == "vx":
            return True, self._vx
        if label == "vy":
            return True, self._vy
        if label == "wz":
            return True, self._wz
        return False, current

    def button(self, label: str) -> bool:
        del label
        return self._zero_clicked


def _build(
    *, with_plugin: bool = True
) -> tuple[ImGuiTwistBridge, _FakeEnv, _FakeImGuiPlugin | None, _FakeCommandTerm]:
    cfg = ImGuiTwistBridgeCfg(
        vx_range=(-1.0, 1.0),
        vy_range=(-1.0, 1.0),
        wz_range=(-0.5, 0.5),
    )
    bridge = cfg.class_type(cfg)  # type: ignore[misc]
    plugin = _FakeImGuiPlugin() if with_plugin else None
    term = _FakeCommandTerm()
    env = _FakeEnv(plugin, term)
    return bridge, env, plugin, term


def test_on_build_registers_panel_when_plugin_present() -> None:
    bridge, env, plugin, term = _build(with_plugin=True)
    bridge.on_build(env)  # type: ignore[arg-type]
    assert plugin is not None and len(plugin.panels) == 1
    assert plugin.panels[0][1] == "side"
    # The bridge pinned ``command.set_external_source`` so the manager skips
    # random resampling on the term.
    assert term.external_source_calls == [True]
    assert bridge._enabled is True  # pyright: ignore[reportPrivateUsage]


def test_on_build_is_no_op_when_plugin_missing() -> None:
    bridge, env, _plugin, term = _build(with_plugin=False)
    bridge.on_build(env)  # type: ignore[arg-type]
    assert bridge._enabled is False  # pyright: ignore[reportPrivateUsage]
    assert term.external_source_calls == []


def test_panel_callback_writes_slider_state_and_zero_button_resets() -> None:
    bridge, env, plugin, _term = _build(with_plugin=True)
    bridge.on_build(env)  # type: ignore[arg-type]
    callback, _section = plugin.panels[0]  # type: ignore[union-attr]

    # First frame: sliders return non-zero values; bridge state should follow.
    callback(_FakeImguiApi(vx=0.5, vy=-0.25, wz=0.1))
    assert bridge._vx == pytest.approx(0.5)  # pyright: ignore[reportPrivateUsage]
    assert bridge._vy == pytest.approx(-0.25)  # pyright: ignore[reportPrivateUsage]
    assert bridge._wz == pytest.approx(0.1)  # pyright: ignore[reportPrivateUsage]

    # Second frame: zero button clicked — bridge state resets even if sliders held.
    callback(_FakeImguiApi(vx=0.5, vy=-0.25, wz=0.1, zero_clicked=True))
    assert bridge._vx == 0.0  # pyright: ignore[reportPrivateUsage]
    assert bridge._vy == 0.0  # pyright: ignore[reportPrivateUsage]
    assert bridge._wz == 0.0  # pyright: ignore[reportPrivateUsage]


def test_pre_step_writes_slider_state_into_command_buffer() -> None:
    bridge, env, plugin, term = _build(with_plugin=True)
    bridge.on_build(env)  # type: ignore[arg-type]
    callback, _section = plugin.panels[0]  # type: ignore[union-attr]
    callback(_FakeImguiApi(vx=0.7, vy=-0.3, wz=0.2))

    bridge.pre_step(env)  # type: ignore[arg-type]
    expected = torch.tensor([[0.7, -0.3, 0.2], [0.7, -0.3, 0.2]])
    assert torch.allclose(term.command, expected)


def test_pre_step_is_no_op_when_disabled() -> None:
    bridge, env, _plugin, term = _build(with_plugin=False)
    bridge.on_build(env)  # type: ignore[arg-type]
    # Pre-populate the buffer so we can verify nothing overwrites it.
    term.command[:] = torch.tensor([[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]])
    bridge.pre_step(env)  # type: ignore[arg-type]
    assert torch.allclose(term.command, torch.full((2, 3), 9.0))
