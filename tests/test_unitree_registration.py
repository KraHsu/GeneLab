"""Verify the unitree extension registers task / robot / env entries correctly."""

from pathlib import Path

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
    # The G1 robot itself is registered by `genelab.asset_zoo.unitree_g1`, not the example.
    assert "g1" in ROBOTS.names()
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
    assert task.cfg.play_env.simulation.num_envs <= task.cfg.env.simulation.num_envs


def test_g1_asset_zoo_cfg_resolves_fetched_mjcf() -> None:
    load_extension_module("genelab_unitree.tasks")
    entity_cfg = ROBOTS.get("g1")

    assert Path(entity_cfg.mjcf_path).exists()
    # G1 has 29 actuated joints; the default-pose regex map should fan out > 4 entries.
    assert len(entity_cfg.default_joint_pos) >= 4
    # 6 DCMotor actuator groups (5020 / 7520_14 / 7520_22 / 4010 / waist / ankle), all
    # carrying a positive per-group action scale.
    assert set(entity_cfg.actuators.keys()) == {
        "5020",
        "7520_14",
        "7520_22",
        "4010",
        "waist",
        "ankle",
    }
    assert all(cfg.action_scale > 0 for cfg in entity_cfg.actuators.values())


def test_g1_tasks_run_simulation_on_gpu_backend() -> None:
    """Guard against the CPU-backend perf regression: both G1 humanoid tasks must
    keep the GPU backend (``simulation.gpu=True``; ``None`` would also resolve to GPU
    since ``device='cuda'``, but G1 pins it explicitly). The original regression ran
    G1 physics on CPU — GPU idle, ~30x slower per step. Train and play cfgs are both
    checked (play renders, which also needs the GPU backend)."""
    load_extension_module("genelab_unitree.tasks")
    for task_id in (
        "Genelab-Velocity-Flat-Unitree-G1-v0",
        "Genelab-Tracking-Flat-Unitree-G1-v0",
    ):
        task = TASKS.get(task_id)
        assert task.cfg.env.simulation.gpu is True, f"{task_id}: train env not on GPU backend"
        assert task.cfg.play_env.simulation.gpu is True, f"{task_id}: play env not on GPU backend"
