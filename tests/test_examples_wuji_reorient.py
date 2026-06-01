"""Tests for the Wuji-hand SO(3) in-hand reorientation task.

Cfg-structure / registration / math tests run without a simulator (the MJCF resolver is
monkeypatched so no asset download happens). The env build + step is ``genesis_runtime``-
gated so it skips cleanly on headless CI.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.registry import TASKS, load_extension_module  # noqa: E402
from genelab_wuji.reorient.asset import _MESHES, _MJCF_TEMPLATE  # noqa: E402
from genelab_wuji.reorient.mdp._math import (  # noqa: E402
    matrix_to_rotation_6d,
    quat_to_rotation_6d,
    random_quat,
)


def test_random_quat_is_unit_and_canonical() -> None:
    q = random_quat(64, "cpu")
    assert q.shape == (64, 4)
    assert torch.allclose(q.norm(dim=-1), torch.ones(64), atol=1e-5)
    assert bool((q[:, 0] >= 0).all())  # canonical w >= 0


def test_rotation_6d_shapes() -> None:
    q = random_quat(8, "cpu")
    assert quat_to_rotation_6d(q).shape == (8, 6)
    eye = torch.eye(3).expand(8, 3, 3)
    assert matrix_to_rotation_6d(eye).shape == (8, 6)


def test_reorient_mesh_asset_spec() -> None:
    spec = _MESHES
    assert spec.url.startswith("https://raw.githubusercontent.com/KraHsu/genelab-assets/")
    assert len(spec.md5) == 32
    assert spec.archive_member.endswith("meshes/right/right_palm_link.STL")
    assert _MJCF_TEMPLATE.exists()  # MJCF ships in-tree


def test_reorient_task_registered() -> None:
    load_extension_module("genelab_wuji.tasks")
    assert "Genelab-Reorient-Wuji-Hand-v0" in TASKS.names()
    assert TASKS.entry("Genelab-Reorient-Wuji-Hand-v0").description.startswith("RSL-RL PPO")


def _patched_cfg(monkeypatch: pytest.MonkeyPatch, play: bool):
    # Avoid the mesh download: the cfg only stores mjcf_path as a string until env build.
    monkeypatch.setattr("genelab_wuji.reorient.asset.resolve_reorient_mjcf", lambda: _MJCF_TEMPLATE)
    from genelab_wuji.reorient.env_cfg import wuji_hand_reorient_env_cfg

    return wuji_hand_reorient_env_cfg(play=play, num_envs=16)


def test_reorient_env_cfg_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _patched_cfg(monkeypatch, play=False)
    assert set(cfg.observations_cfg) == {"policy", "critic"}
    assert "joint_pos" in cfg.actions_cfg
    assert "reorient_command" in cfg.commands_cfg
    # task + contact rewards both present
    for key in (
        "orientation_alignment",
        "hold_escalation",
        "tip_slide",
        "palm_detach",
        "finger_collision",
    ):
        assert key in cfg.rewards_cfg
    assert {"time_out", "cage_drop"} <= set(cfg.terminations_cfg)
    sensor_names = {s.name for s in cfg.scene.sensors}
    assert {"hand_object_contact", "finger_collision"} <= sensor_names
    # training DR present
    for key in ("robot_friction", "pd_gains", "encoder_bias", "object_disturbance"):
        assert key in cfg.events_cfg
    # training curriculum present; success threshold starts loose
    assert {"success_curriculum", "adaptive_episode"} <= set(cfg.curriculum_cfg)
    assert cfg.commands_cfg["reorient_command"].success_threshold == 0.8


def test_reorient_play_cfg_strips_domain_randomization(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _patched_cfg(monkeypatch, play=True)
    # evaluation runs nominal physics: DR + disturbance removed, resets kept.
    for key in ("robot_friction", "pd_gains", "encoder_bias", "object_disturbance"):
        assert key not in cfg.events_cfg
    assert "reset_object" in cfg.events_cfg
    # no curriculum in eval; tight target success threshold
    assert cfg.curriculum_cfg == {}
    assert cfg.commands_cfg["reorient_command"].success_threshold == 0.2


def test_reorient_env_builds_and_steps(genesis_runtime: Any) -> None:
    del genesis_runtime  # fixture only guards EGL/runtime availability
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from genelab_wuji.reorient.env_cfg import wuji_hand_reorient_env_cfg

    cfg = wuji_hand_reorient_env_cfg(play=False, num_envs=4)
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.device = "cpu"
    env = ManagerBasedRlEnv(cfg)
    assert env.num_actions == 20
    obs = env.observation_manager.compute()
    assert obs["policy"].shape == (4, 69)
    assert obs["critic"].shape == (4, 74)
    act = torch.zeros(env.num_envs, env.num_actions)
    for _ in range(8):
        env.step(act)
    rew = env.reward_manager.compute(cfg.decimation * cfg.simulation.dt)
    assert bool(torch.isfinite(rew).all())
    assert env.sensors["hand_object_contact"].data.tip_found.shape == (4, 5)
