"""Multi-robot foundation (ROADMAP M3.6 / ADR-0012 slice S1).

Genesis-free: covers the entity-cfg resolution + primary selection + the entity-aware
``SceneEntityCfg.resolve`` seam. The actual multi-robot env *build* is exercised by the
``genesis_runtime``-gated scene test (``test_interactive_scene.py``).
"""

from __future__ import annotations

import types

import pytest
import torch

from genelab.entity import ArticulationCfg
from genelab.envs.manager_based_rl_env import (
    ManagerBasedRlEnvCfg,
    _primary_entity_name,
    _resolve_entity_cfgs,
)
from genelab.managers.scene_entity_cfg import SceneEntityCfg
from genelab.mdp._helpers import resolve_articulation
from genelab.mdp.actions.joint_position import JointPositionAction, JointPositionActionCfg


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


# ---------------------------------------------------------------- S2: asset_name routing


class _PrimaryOnlyEnv:
    """Fake env exposing only the singular ``articulation`` (no ``articulations``)."""

    def __init__(self) -> None:
        self.articulation = object()


def test_resolve_articulation_picks_named_entity_else_primary() -> None:
    a, b = object(), object()
    multi = types.SimpleNamespace(articulations={"robot": a, "robot_b": b}, articulation=a)
    assert resolve_articulation(multi, "robot_b") is b  # type: ignore[arg-type]
    assert resolve_articulation(multi, "robot") is a  # type: ignore[arg-type]
    # Unknown name → primary fallback.
    assert resolve_articulation(multi, "nope") is a  # type: ignore[arg-type]
    # Legacy env without .articulations → primary.
    legacy = _PrimaryOnlyEnv()
    assert resolve_articulation(legacy, "robot") is legacy.articulation  # type: ignore[arg-type]


class _RoutingArticulation:
    def __init__(self, joint_names: list[str], num_envs: int) -> None:
        self.joint_names = joint_names
        n = len(joint_names)
        self.default_joint_pos = torch.zeros(n)
        self.action_scale_tensor = torch.ones(n)
        self.data = types.SimpleNamespace(encoder_bias=torch.zeros(num_envs, n))
        self.written: tuple[torch.Tensor, torch.Tensor] | None = None

    def write_joint_targets_partial(self, idx: torch.Tensor, target: torch.Tensor) -> None:
        self.written = (idx, target)


class _RoutingEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self._a = _RoutingArticulation(["ja0"], self.num_envs)
        self._b = _RoutingArticulation(["jb0", "jb1"], self.num_envs)
        self.articulations = {"robot": self._a, "robot_b": self._b}
        self.articulation = self._a


def test_joint_position_action_routes_to_named_entity() -> None:
    env = _RoutingEnv()
    cfg = JointPositionActionCfg(asset_name="robot_b", joint_names=(".*",), scale=1.0)
    term = JointPositionAction(cfg, env)  # type: ignore[arg-type]
    # Matched robot_b's two joints, not robot's one.
    assert term.action_dim == 2
    term.process_actions(torch.zeros(2, 2))
    term.apply_actions()
    assert env._b.written is not None  # wrote to robot_b
    assert env._a.written is None  # primary left untouched


# ---------------------------------------------------------------- S4: sensor entity routing


def test_sensor_entity_resolvers_pick_named_else_primary() -> None:
    from genelab.sensor._entity import entity_articulation, entity_handle, entity_state

    a = types.SimpleNamespace(gs_handle="h_a", data="state_a")
    b = types.SimpleNamespace(gs_handle="h_b", data="state_b")
    multi = types.SimpleNamespace(
        articulations={"robot": a, "robot_b": b}, articulation=a, robot="h_a", robot_state="state_a"
    )
    assert entity_articulation(multi, "robot_b") is b  # type: ignore[arg-type]
    assert entity_handle(multi, "robot_b") == "h_b"  # type: ignore[arg-type]
    assert entity_state(multi, "robot_b") == "state_b"  # type: ignore[arg-type]
    # Unknown name / legacy env → primary fallback.
    assert entity_articulation(multi, "nope") is a  # type: ignore[arg-type]
    legacy = types.SimpleNamespace(articulation=a, robot="h_a", robot_state="state_a")
    assert entity_handle(legacy, "robot") == "h_a"  # type: ignore[arg-type]
    assert entity_state(legacy, "robot") == "state_a"  # type: ignore[arg-type]
    # entity_articulation falls back to env itself for minimal fakes (env-level link_names).
    bare = types.SimpleNamespace(link_names=["l0"])
    assert entity_articulation(bare, "robot").link_names == ["l0"]  # type: ignore[arg-type]


def test_force_torque_sensor_routes_to_named_entity() -> None:
    from genelab.sensor import ForceTorqueSensorCfg

    force_b = torch.tensor([[10.0, 11, 12], [20, 21, 22]])  # (2 envs, 3 dofs)
    handle_b = types.SimpleNamespace(get_dofs_force=lambda: force_b)
    art_a = types.SimpleNamespace(joint_names=["ja"], actuated_dof_ids=torch.tensor([0]))
    art_b = types.SimpleNamespace(
        joint_names=["jb0", "jb1"],
        actuated_dof_ids=torch.tensor([1, 2]),  # robot_b's joints sit at global dofs 1, 2
        gs_handle=handle_b,
    )
    env = types.SimpleNamespace(
        num_envs=2,
        device="cpu",
        articulations={"robot": art_a, "robot_b": art_b},
        articulation=art_a,
        robot=None,
        joint_names=["ja"],
    )
    sensor = ForceTorqueSensorCfg(name="ft", entity_name="robot_b").build()
    sensor.bind(env)  # type: ignore[arg-type]
    assert sensor.joint_names == ["jb0", "jb1"]  # resolved against robot_b
    assert torch.allclose(sensor.data.force, force_b[:, [1, 2]])  # robot_b's dofs from its handle
