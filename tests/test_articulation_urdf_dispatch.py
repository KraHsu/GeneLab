"""Unit tests for ``Articulation`` dispatching between MJCF and URDF/xacro morphs.

The morph classes validate file existence at construction, so the dispatch tests
intercept ``gs.morphs.MJCF`` / ``gs.morphs.URDF`` with synthetic recorders that
capture the kwargs. Validation tests cover the (no real file needed) error paths
on ``Articulation.__init__``.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.entity.articulation import Articulation, ArticulationCfg  # noqa: E402


class _RecordingMorph:
    """Stand-in for a Genesis morph that records the kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeGsScene:
    """Records ``add_entity`` calls so tests can inspect the morph type + kwargs."""

    def __init__(self) -> None:
        self.morphs: list[Any] = []

    def add_entity(self, morph: Any) -> Any:
        self.morphs.append(morph)
        return morph


@pytest.fixture
def fake_morphs(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[_RecordingMorph]]:
    """Replace ``gs.morphs.MJCF`` / ``gs.morphs.URDF`` with recorders for dispatch tests."""

    class _FakeMJCF(_RecordingMorph):
        pass

    class _FakeURDF(_RecordingMorph):
        pass

    import genesis as gs

    monkeypatch.setattr(gs.morphs, "MJCF", _FakeMJCF)
    monkeypatch.setattr(gs.morphs, "URDF", _FakeURDF)
    return {"MJCF": _FakeMJCF, "URDF": _FakeURDF}


def test_mjcf_path_dispatches_to_mjcf_morph(
    fake_morphs: dict[str, type[_RecordingMorph]],
) -> None:
    cfg = ArticulationCfg(mjcf_path="/tmp/robot.xml", init_pos=(0.5, 0.0, 1.0))
    art = Articulation(cfg, name="robot")
    scene = _FakeGsScene()
    art.spawn(scene)

    morph = scene.morphs[0]
    assert isinstance(morph, fake_morphs["MJCF"])
    assert not isinstance(morph, fake_morphs["URDF"])
    assert morph.kwargs["file"] == "/tmp/robot.xml"
    assert morph.kwargs["pos"] == (0.5, 0.0, 1.0)
    assert "xacro_args" not in morph.kwargs


def test_urdf_path_dispatches_to_urdf_morph(
    fake_morphs: dict[str, type[_RecordingMorph]],
) -> None:
    cfg = ArticulationCfg(urdf_path="/tmp/robot.urdf", init_pos=(0.0, 0.5, 1.0))
    art = Articulation(cfg, name="robot")
    scene = _FakeGsScene()
    art.spawn(scene)

    morph = scene.morphs[0]
    assert isinstance(morph, fake_morphs["URDF"])
    assert morph.kwargs["file"] == "/tmp/robot.urdf"
    assert morph.kwargs["pos"] == (0.0, 0.5, 1.0)
    assert "xacro_args" not in morph.kwargs


def test_xacro_args_forwarded_to_urdf_morph(
    fake_morphs: dict[str, type[_RecordingMorph]],
) -> None:
    cfg = ArticulationCfg(
        urdf_path="/tmp/robot.urdf.xacro",
        xacro_args={"use_sim": "true", "arm_length": "0.5"},
    )
    art = Articulation(cfg, name="robot")
    scene = _FakeGsScene()
    art.spawn(scene)

    morph = scene.morphs[0]
    assert isinstance(morph, fake_morphs["URDF"])
    assert morph.kwargs["file"] == "/tmp/robot.urdf.xacro"
    assert morph.kwargs["xacro_args"] == {"use_sim": "true", "arm_length": "0.5"}


def test_requires_jac_and_ik_forwarded_to_urdf(
    fake_morphs: dict[str, type[_RecordingMorph]],
) -> None:
    cfg = ArticulationCfg(urdf_path="/tmp/robot.urdf", requires_jac_and_ik=True)
    art = Articulation(cfg, name="robot")
    scene = _FakeGsScene()
    art.spawn(scene)

    morph = scene.morphs[0]
    assert morph.kwargs.get("requires_jac_and_IK") is True


def test_requires_jac_and_ik_default_false_omits_kwarg(
    fake_morphs: dict[str, type[_RecordingMorph]],
) -> None:
    """The cfg default of ``requires_jac_and_ik=False`` does not override Genesis defaults."""
    cfg = ArticulationCfg(urdf_path="/tmp/robot.urdf")
    art = Articulation(cfg, name="robot")
    scene = _FakeGsScene()
    art.spawn(scene)

    morph = scene.morphs[0]
    assert "requires_jac_and_IK" not in morph.kwargs


def test_no_paths_raises() -> None:
    with pytest.raises(ValueError, match="exactly one of mjcf_path / urdf_path"):
        Articulation(ArticulationCfg(), name="robot")


def test_both_paths_raises() -> None:
    cfg = ArticulationCfg(mjcf_path="/a.xml", urdf_path="/b.urdf")
    with pytest.raises(ValueError, match="exactly one of mjcf_path / urdf_path"):
        Articulation(cfg, name="robot")


def test_xacro_args_without_urdf_raises() -> None:
    cfg = ArticulationCfg(mjcf_path="/a.xml", xacro_args={"k": "v"})
    with pytest.raises(ValueError, match="xacro_args is only valid with urdf_path"):
        Articulation(cfg, name="robot")
