"""Unit tests for :class:`genelab.entity.Avatar`."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.entity.avatar import Avatar, AvatarCfg  # noqa: E402


class _RecordingMorph:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeGsHandle:
    def __init__(self) -> None:
        self.set_pos_calls: list[tuple[Any, Any]] = []
        self.set_quat_calls: list[tuple[Any, Any]] = []

    def set_pos(self, pos: Any, *, envs_idx: Any = None) -> None:
        self.set_pos_calls.append((pos, envs_idx))

    def set_quat(self, quat: Any, *, envs_idx: Any = None) -> None:
        self.set_quat_calls.append((quat, envs_idx))


class _FakeGsScene:
    def __init__(self) -> None:
        self.entities: list[tuple[Any, Any]] = []

    def add_entity(self, morph: Any, material: Any) -> _FakeGsHandle:
        self.entities.append((morph, material))
        return _FakeGsHandle()


@pytest.fixture
def patched_genesis(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub ``gs.morphs.*`` + ``gs.materials.Kinematic`` so ``spawn`` runs offline."""

    class _FakeBox(_RecordingMorph):
        pass

    class _FakeSphere(_RecordingMorph):
        pass

    class _FakeMesh(_RecordingMorph):
        pass

    class _FakeMJCF(_RecordingMorph):
        pass

    class _FakeKinematic:
        pass

    import genesis as gs

    monkeypatch.setattr(gs.morphs, "Box", _FakeBox)
    monkeypatch.setattr(gs.morphs, "Sphere", _FakeSphere)
    monkeypatch.setattr(gs.morphs, "Mesh", _FakeMesh)
    monkeypatch.setattr(gs.morphs, "MJCF", _FakeMJCF)
    monkeypatch.setattr(gs.materials, "Kinematic", _FakeKinematic)
    return {
        "Box": _FakeBox,
        "Sphere": _FakeSphere,
        "Mesh": _FakeMesh,
        "MJCF": _FakeMJCF,
        "Kinematic": _FakeKinematic,
    }


def test_avatar_box_spawn_records_kinematic_material(patched_genesis: dict[str, Any]) -> None:
    cfg = AvatarCfg(morph="box", size=(0.2, 0.3, 0.4), init_pos=(1.0, 2.0, 3.0))
    avatar = Avatar(cfg, name="ghost")
    scene = _FakeGsScene()
    avatar.spawn(scene)

    morph, material = scene.entities[0]
    assert isinstance(morph, patched_genesis["Box"])
    assert isinstance(material, patched_genesis["Kinematic"])
    assert morph.kwargs["size"] == (0.2, 0.3, 0.4)
    assert morph.kwargs["pos"] == (1.0, 2.0, 3.0)


def test_avatar_sphere_spawn(patched_genesis: dict[str, Any]) -> None:
    cfg = AvatarCfg(morph="sphere", size=(0.05,))
    avatar = Avatar(cfg, name="ball")
    scene = _FakeGsScene()
    avatar.spawn(scene)
    morph, _ = scene.entities[0]
    assert isinstance(morph, patched_genesis["Sphere"])
    assert morph.kwargs["radius"] == 0.05


def test_avatar_mesh_requires_file(patched_genesis: dict[str, Any]) -> None:
    avatar = Avatar(AvatarCfg(morph="mesh", file=None), name="m")
    with pytest.raises(ValueError, match="requires cfg.file"):
        avatar.spawn(_FakeGsScene())


def test_avatar_box_rejects_wrong_size(patched_genesis: dict[str, Any]) -> None:
    avatar = Avatar(AvatarCfg(morph="box", size=(0.1, 0.2)), name="b")
    with pytest.raises(ValueError, match="size=\\(x, y, z\\)"):
        avatar.spawn(_FakeGsScene())


def test_avatar_unknown_morph_raises(patched_genesis: dict[str, Any]) -> None:
    avatar = Avatar(AvatarCfg(morph="capsule"), name="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown morph"):
        avatar.spawn(_FakeGsScene())


def test_set_pose_before_spawn_raises() -> None:
    avatar = Avatar(AvatarCfg(), name="x")
    with pytest.raises(RuntimeError, match="set_pose called before spawn"):
        avatar.set_pose(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_set_pose_writes_pos_and_quat(patched_genesis: dict[str, Any]) -> None:
    avatar = Avatar(AvatarCfg(morph="box", size=(0.1, 0.1, 0.1)), name="g")
    scene = _FakeGsScene()
    avatar.spawn(scene)

    pos = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    avatar.set_pose(pos, quat)

    handle = avatar.gs_handle
    assert len(handle.set_pos_calls) == 1
    assert torch.equal(handle.set_pos_calls[0][0], pos)
    assert len(handle.set_quat_calls) == 1
    assert torch.equal(handle.set_quat_calls[0][0], quat)


def test_avatar_gs_handle_none_before_spawn() -> None:
    avatar = Avatar(AvatarCfg(), name="x")
    assert avatar.gs_handle is None
