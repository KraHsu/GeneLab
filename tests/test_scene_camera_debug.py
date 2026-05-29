"""Unit tests for ``InteractiveScene.draw_camera_frustums`` / ``draw_camera_trajectory``.

The scene helpers wrap Genesis 1.0's ``scene.draw_debug_frustum(camera, color=...)``
and ``scene.draw_debug_trajectory(poss, ...)``. The tests use a fake gs_scene that
records every debug-draw call so we can assert (a) the right camera handle is
forwarded for each selection and (b) selection errors are raised with useful
messages — no real Genesis viewer needed.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")


class _FakeCameraHandle:
    """Stand-in for the gs.Camera handle attached by ``CameraSensor._allocate_camera``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCameraSensor:
    """Minimum subset of CameraSensor that ``_select_camera_sensors`` reads.

    Subclasses the real ``CameraSensor`` class symbol so the ``isinstance`` check
    inside the helper still passes — the helper only ever touches ``gs_camera``.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._cam = _FakeCameraHandle(name)

    @property
    def gs_camera(self) -> Any:
        return self._cam


class _FakeGsScene:
    """Records every debug-draw call so tests can assert on them."""

    def __init__(self) -> None:
        self.frustum_calls: list[tuple[Any, tuple[float, float, float, float]]] = []
        self.trajectory_calls: list[tuple[Any, float, tuple[float, float, float, float]]] = []

    def draw_debug_frustum(self, camera: Any, color: tuple[float, float, float, float]) -> None:
        self.frustum_calls.append((camera, color))

    def draw_debug_trajectory(
        self, poss: Any, radius: float, color: tuple[float, float, float, float]
    ) -> None:
        self.trajectory_calls.append((poss, radius, color))


def _build_scene(sensors: dict[str, Any], *, built: bool = True) -> Any:
    """Construct just enough ``InteractiveScene`` surface for the debug helpers."""
    from genelab.scene.interactive_scene import InteractiveScene

    scene = InteractiveScene.__new__(InteractiveScene)
    scene._gs_scene = _FakeGsScene()
    scene._sensors = sensors
    scene._built = built
    return scene


def _patch_camera_sensor_isinstance(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_select_camera_sensors`` uses ``isinstance(sensor, CameraSensor)``; route
    the import to a class the fakes inherit from so the test fakes pass the check
    without needing a real Genesis env to construct a real ``CameraSensor``."""
    import genelab.sensor.camera as camera_mod

    monkeypatch.setattr(camera_mod, "CameraSensor", _FakeCameraSensor)


def test_draw_camera_frustums_default_draws_every_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_sensor_isinstance(monkeypatch)
    sensors = {"front": _FakeCameraSensor("front"), "side": _FakeCameraSensor("side")}
    scene = _build_scene(sensors)

    n = scene.draw_camera_frustums()

    assert n == 2
    drawn = [call[0].name for call in scene._gs_scene.frustum_calls]
    assert sorted(drawn) == ["front", "side"]


def test_draw_camera_frustums_named_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_sensor_isinstance(monkeypatch)
    sensors = {"front": _FakeCameraSensor("front"), "side": _FakeCameraSensor("side")}
    scene = _build_scene(sensors)

    n = scene.draw_camera_frustums(camera_names=("side",), color=(0.1, 0.2, 0.3, 0.4))

    assert n == 1
    cam, color = scene._gs_scene.frustum_calls[0]
    assert cam.name == "side"
    assert color == (0.1, 0.2, 0.3, 0.4)


def test_draw_camera_frustums_unknown_name_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_sensor_isinstance(monkeypatch)
    sensors = {"front": _FakeCameraSensor("front")}
    scene = _build_scene(sensors)

    with pytest.raises(KeyError, match="no sensor named 'ghost'"):
        scene.draw_camera_frustums(camera_names=("ghost",))


def test_draw_camera_frustums_non_camera_sensor_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_camera_sensor_isinstance(monkeypatch)
    sensors = {"front": _FakeCameraSensor("front"), "imu": object()}
    scene = _build_scene(sensors)

    with pytest.raises(TypeError, match="not a CameraSensor"):
        scene.draw_camera_frustums(camera_names=("imu",))


def test_draw_camera_frustums_pre_build_raises() -> None:
    scene = _build_scene({}, built=False)
    with pytest.raises(RuntimeError, match="before InteractiveScene.build"):
        scene.draw_camera_frustums()


def test_draw_camera_trajectory_forwards_args() -> None:
    scene = _build_scene({})
    poss = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    scene.draw_camera_trajectory(poss, radius=0.005, color=(0.0, 1.0, 0.0, 1.0))

    assert len(scene._gs_scene.trajectory_calls) == 1
    forwarded, radius, color = scene._gs_scene.trajectory_calls[0]
    assert torch.equal(forwarded, poss)
    assert radius == pytest.approx(0.005)
    assert color == (0.0, 1.0, 0.0, 1.0)


def test_draw_camera_trajectory_pre_build_raises() -> None:
    scene = _build_scene({}, built=False)
    with pytest.raises(RuntimeError, match="before InteractiveScene.build"):
        scene.draw_camera_trajectory([[0.0, 0.0, 0.0]])
