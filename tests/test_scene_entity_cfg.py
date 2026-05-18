"""Unit tests for ``SceneEntityCfg`` and the manager-side auto-resolve hook (P5.2)."""

from dataclasses import dataclass

import pytest

from genelab.managers._base import instantiate_class_term
from genelab.managers.manager_term_cfg import ManagerTermBaseCfg
from genelab.managers.scene_entity_cfg import SceneEntityCfg


@dataclass
class _FakeEnv:
    link_names: list[str]
    joint_names: list[str]


# --------------------------------------------------------------------- resolve()


def test_resolve_link_names_populates_link_ids() -> None:
    env = _FakeEnv(link_names=["base", "left_foot", "right_foot"], joint_names=[])
    cfg = SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot"))
    cfg.resolve(env)
    assert cfg.link_ids == (1, 2)


def test_resolve_joint_names_populates_joint_ids() -> None:
    env = _FakeEnv(link_names=[], joint_names=["hip", "knee", "ankle"])
    cfg = SceneEntityCfg(name="robot", joint_names=("knee",))
    cfg.resolve(env)
    assert cfg.joint_ids == (1,)


def test_resolve_is_idempotent_safe_on_re_run() -> None:
    """Resolving twice must not duplicate or corrupt the index tuples."""
    env = _FakeEnv(link_names=["a", "b"], joint_names=[])
    cfg = SceneEntityCfg(name="robot", link_names=("b",))
    cfg.resolve(env)
    first = cfg.link_ids
    cfg.resolve(env)
    assert cfg.link_ids == first  # unchanged
    assert cfg.link_ids == (1,)


def test_resolve_raises_on_unknown_link_name() -> None:
    env = _FakeEnv(link_names=["base", "torso"], joint_names=[])
    cfg = SceneEntityCfg(name="robot", link_names=("torso", "missing"))
    with pytest.raises(ValueError, match="'missing'"):
        cfg.resolve(env)


def test_resolve_raises_on_unknown_joint_name() -> None:
    env = _FakeEnv(link_names=[], joint_names=["hip", "knee"])
    cfg = SceneEntityCfg(name="robot", joint_names=("elbow",))
    with pytest.raises(ValueError, match="'elbow'"):
        cfg.resolve(env)


def test_resolve_no_op_when_no_names_set() -> None:
    """Default SceneEntityCfg with no names is a "select everything" sentinel —
    resolve should leave link_ids/joint_ids None for DR consumers to interpret."""
    env = _FakeEnv(link_names=["a", "b"], joint_names=["x"])
    cfg = SceneEntityCfg(name="robot")
    cfg.resolve(env)
    assert cfg.link_ids is None
    assert cfg.joint_ids is None


# --------------------------------------------------------------------- auto-resolve hook


def _noop_func(env, **kwargs) -> None:  # noqa: ARG001
    return None


def test_instantiate_class_term_resolves_scene_entity_cfg_in_params() -> None:
    """A SceneEntityCfg sitting in term_cfg.params gets resolved at construction."""
    env = _FakeEnv(link_names=["base", "foot"], joint_names=[])
    asset_cfg = SceneEntityCfg(name="robot", link_names=("foot",))
    term_cfg = ManagerTermBaseCfg(func=_noop_func, params={"asset_cfg": asset_cfg})
    instantiate_class_term(term_cfg, env)  # type: ignore[arg-type]
    assert asset_cfg.link_ids == (1,)


def test_instantiate_class_term_ignores_non_scene_entity_params() -> None:
    """Other param types must not be touched by the resolve pass."""
    env = _FakeEnv(link_names=["a"], joint_names=[])
    term_cfg = ManagerTermBaseCfg(
        func=_noop_func,
        params={"threshold": 0.5, "names": ("a",), "ranges": (-1.0, 1.0)},
    )
    instantiate_class_term(term_cfg, env)  # type: ignore[arg-type]
    # Sanity: params untouched.
    assert term_cfg.params == {"threshold": 0.5, "names": ("a",), "ranges": (-1.0, 1.0)}


def test_instantiate_class_term_resolves_asset_cfg_before_class_init() -> None:
    """Regression: stateful class terms like ``feet_swing_height`` cache
    ``asset_cfg.link_ids`` in their ``__init__``. The resolve pass MUST run before
    the class is instantiated — otherwise the init reads ``None`` and crashes
    (caught in production by an 8-GPU G1 training run that hit
    ``ValueError: SceneEntityCfg(name='robot') has no link_ids``).
    """
    captured: dict[str, tuple[int, ...] | None] = {}

    class _RecordingTerm:
        def __init__(self, cfg, env) -> None:  # type: ignore[no-untyped-def]
            # Mirrors what feet_swing_height.__init__ does — read the cfg at init time.
            captured["link_ids"] = cfg.params["asset_cfg"].link_ids

        def __call__(self, env) -> None:  # noqa: ARG002 - placeholder
            return None

    env = _FakeEnv(link_names=["base", "left_foot", "right_foot"], joint_names=[])
    asset_cfg = SceneEntityCfg(name="robot", link_names=("left_foot", "right_foot"))
    term_cfg = ManagerTermBaseCfg(func=_RecordingTerm, params={"asset_cfg": asset_cfg})
    instantiate_class_term(term_cfg, env)  # type: ignore[arg-type]
    # The recording term's init saw fully resolved link_ids.
    assert captured["link_ids"] == (1, 2)
