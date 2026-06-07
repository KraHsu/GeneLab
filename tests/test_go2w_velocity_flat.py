"""``Genelab-Velocity-Flat-Unitree-Go2W-v0``: wheeled quadruped, mixed position+velocity control.

Cfg-level + registration checks run everywhere (no simulator). The build + step smoke is
gated on ``genesis_runtime`` so it skips cleanly on EGL-less CI, and it asserts the new
velocity-action path actually spins the wheels in Genesis.
"""

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from genelab.registry import ENVS, ROBOTS, TASKS, load_extension_module  # noqa: E402

TASK_ID = "Genelab-Velocity-Flat-Unitree-Go2W-v0"
ENV_NAME = "go2w-velocity-flat-env"


def _flat_cfg(play: bool = True):
    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg

    return unitree_go2w_velocity_env_cfg(play=play)


def test_go2w_task_registers() -> None:
    load_extension_module("genelab_unitree.tasks")
    # The Go2-W robot is registered by `genelab.asset_zoo.unitree_go2w`, not the example.
    assert "go2w" in ROBOTS.names()
    assert ENV_NAME in ENVS.names()
    assert TASK_ID in TASKS.names()


def test_actions_split_position_legs_and_velocity_wheels() -> None:
    from genelab.mdp.actions.joint_position import JointPositionActionCfg
    from genelab.mdp.actions.joint_velocity import JointVelocityActionCfg

    actions = _flat_cfg(play=True).actions_cfg
    # Legs are position-controlled, wheels are velocity-controlled — the Go2-W distinction.
    assert isinstance(actions["joint_pos"], JointPositionActionCfg)
    assert isinstance(actions["wheel_vel"], JointVelocityActionCfg)
    assert actions["joint_pos"].joint_names == (r".*_(hip|thigh|calf)_joint",)
    assert actions["wheel_vel"].joint_names == (r".*_wheel_joint",)


def test_actor_is_proprioception_only_critic_is_privileged() -> None:
    cfg = _flat_cfg(play=True)
    actor = cfg.observations_cfg["policy"].terms
    critic = cfg.observations_cfg["critic"].terms
    # Deployable actor never sees base linear velocity (no hardware sensor); critic does.
    assert "base_lin_vel" not in actor
    assert "base_lin_vel" in critic


def test_play_and_train_split() -> None:
    train = _flat_cfg(play=False)
    play = _flat_cfg(play=True)
    assert train.simulation.num_envs == 4096
    assert play.simulation.num_envs == 1
    assert train.auto_reset is True
    assert play.auto_reset is False
    assert "push_robot" in train.events_cfg
    assert "push_robot" not in play.events_cfg


def _build_cpu_env(cfg, num_envs: int = 4):
    from genelab.envs.manager_based_rl_env import ManagerBasedRlEnv

    cfg.simulation.num_envs = num_envs
    cfg.simulation.gpu = False
    cfg.simulation.vis = False
    cfg.device = "cpu"
    return ManagerBasedRlEnv(cfg)


def test_velocity_action_spins_the_wheels(genesis_runtime: Any) -> None:
    """Smoke-build the env and confirm a positive wheel-velocity action actually spins the
    wheels in Genesis — the end-to-end check on the new velocity-control path."""
    del genesis_runtime

    env = _build_cpu_env(_flat_cfg(play=True))
    try:
        env.reset()
        n_act = env.action_manager.total_action_dim
        # 12 leg joints + 4 wheels.
        assert n_act == 16

        wheel_idx = [i for i, n in enumerate(env._articulation.joint_names) if n.endswith("_wheel_joint")]
        assert len(wheel_idx) == 4

        # Command max forward wheel velocity (legs zero); the wheel action term is last, so the
        # trailing 4 action components drive the wheels.
        action = torch.zeros(env.num_envs, n_act, device=env.device)
        action[:, -4:] = 1.0
        for _ in range(5):
            env.step(action)

        wheel_vel = env._articulation.data.joint_vel[:, wheel_idx]
        # Wheels are actually spinning (velocity-controlled), not held at zero like a position joint.
        assert wheel_vel.abs().mean().item() > 1.0
    finally:
        env.close()
