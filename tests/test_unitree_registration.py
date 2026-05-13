"""Verify the unitree extension registers task / robot / env entries correctly."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from genelab.registry import (  # noqa: E402
    ENVS,
    ROBOTS,
    TASKS,
    load_extension_module,
)
from genelab.rl import RslRlOnPolicyRunnerCfg  # noqa: E402


def test_genelab_unitree_registers_all_entries() -> None:
    load_extension_module("genelab_unitree.tasks")
    assert "unitree-g1" in ROBOTS.names()
    assert "g1-velocity-flat-env" in ENVS.names()
    assert "Genelab-Velocity-Flat-Unitree-G1-v0" in TASKS.names()


def test_g1_velocity_task_cfg_is_trainable_with_ppo_agent() -> None:
    load_extension_module("genelab_unitree.tasks")
    task = TASKS.get("Genelab-Velocity-Flat-Unitree-G1-v0")
    assert task.cfg.trainable is True
    assert task.cfg.env is not None
    assert task.cfg.play_env is not None
    assert isinstance(task.cfg.agent, RslRlOnPolicyRunnerCfg)
    # Play mode should drop the push event and shrink num_envs.
    assert "push_robot" not in task.cfg.play_env.events_cfg
    assert "push_robot" in task.cfg.env.events_cfg
    assert task.cfg.play_env.scene.num_envs <= task.cfg.env.scene.num_envs


def test_g1_robot_cfg_resolves_vendored_mjcf() -> None:
    load_extension_module("genelab_unitree.tasks")
    robot = ROBOTS.get("unitree-g1")
    entity_cfg = robot.to_entity_cfg()
    from pathlib import Path

    assert Path(entity_cfg.mjcf_path).exists()
    # G1 has 29 actuated joints; the default-pose regex map should fan out > 4 entries.
    assert len(entity_cfg.default_joint_pos) >= 4
    # Action scales must be positive.
    assert all(v > 0 for v in entity_cfg.action_scale.values())
