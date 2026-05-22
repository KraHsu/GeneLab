"""Multi-robot foundation (ROADMAP M3.6 / ADR-0012 slice S1).

Genesis-free: covers the entity-cfg resolution + primary selection + the entity-aware
``SceneEntityCfg.resolve`` seam. The actual multi-robot env *build* is exercised by the
``genesis_runtime``-gated scene test (``test_interactive_scene.py``).
"""

from __future__ import annotations

import pytest

from genelab.entity import ArticulationCfg
from genelab.envs.manager_based_rl_env import (
    ManagerBasedRlEnvCfg,
    _primary_entity_name,
    _resolve_entity_cfgs,
)
from genelab.managers.scene_entity_cfg import SceneEntityCfg


def test_resolve_entity_cfgs_falls_back_to_single_robot() -> None:
    cfg = ManagerBasedRlEnvCfg()  # robots empty → {"robot": robot}
    resolved = _resolve_entity_cfgs(cfg)
    assert list(resolved.keys()) == ["robot"]
    assert resolved["robot"] is cfg.robot


def test_resolve_entity_cfgs_uses_robots_dict_when_set() -> None:
    a, b = ArticulationCfg(), ArticulationCfg()
    cfg = ManagerBasedRlEnvCfg(robots={"robot_a": a, "robot_b": b})
    resolved = _resolve_entity_cfgs(cfg)
    assert resolved == {"robot_a": a, "robot_b": b}


def test_primary_entity_name_prefers_robot_else_first() -> None:
    assert _primary_entity_name({"robot": ArticulationCfg(), "b": ArticulationCfg()}) == "robot"
    assert _primary_entity_name({"alpha": ArticulationCfg(), "beta": ArticulationCfg()}) == "alpha"


class _FakeArticulation:
    def __init__(self, joint_names: list[str], link_names: list[str]) -> None:
        self.joint_names = joint_names
        self.link_names = link_names


class _MultiEntityEnv:
    """Fake env exposing the M3.6 ``articulations`` accessor."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.articulations = {
            "robot": _FakeArticulation(["ja0", "ja1"], ["la0"]),
            "robot_b": _FakeArticulation(["jb0"], ["lb0", "lb1"]),
        }


class _LegacyEnv:
    """Fake env *without* ``articulations`` — exercises the backward-compat fallback."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.joint_names = ["jx0", "jx1"]
        self.link_names = ["lx0"]


def test_scene_entity_cfg_resolves_against_named_entity() -> None:
    env = _MultiEntityEnv()
    cfg = SceneEntityCfg(name="robot_b", joint_names=("jb0",), link_names=("lb1",))
    cfg.resolve(env)  # type: ignore[arg-type]
    assert cfg.joint_ids == (0,)  # jb0 is index 0 in robot_b's joints
    assert cfg.link_ids == (1,)  # lb1 is index 1 in robot_b's links


def test_scene_entity_cfg_rejects_name_from_wrong_entity() -> None:
    env = _MultiEntityEnv()
    # "ja0" belongs to "robot", not "robot_b".
    cfg = SceneEntityCfg(name="robot_b", joint_names=("ja0",))
    with pytest.raises(ValueError, match="ja0"):
        cfg.resolve(env)  # type: ignore[arg-type]


def test_scene_entity_cfg_fallback_without_articulations() -> None:
    env = _LegacyEnv()
    cfg = SceneEntityCfg(joint_names=("jx1",), link_names=("lx0",))
    cfg.resolve(env)  # type: ignore[arg-type]
    assert cfg.joint_ids == (1,)
    assert cfg.link_ids == (0,)
