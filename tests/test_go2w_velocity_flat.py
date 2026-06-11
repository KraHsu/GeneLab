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


def test_policy_and_critic_use_frame_stacking() -> None:
    cfg = _flat_cfg(play=True)
    # Both groups stack 5 control-step frames so the proprioception-only actor can infer
    # velocity / contact trends — the lever for robust sim2sim transfer. History is kept in
    # play too: the deployed policy expects the stacked observation.
    assert cfg.observations_cfg["policy"].history_length == 5
    assert cfg.observations_cfg["critic"].history_length == 5


def test_train_cfg_has_domain_randomization() -> None:
    train = _flat_cfg(play=False)
    play = _flat_cfg(play=True)
    # Sim2real startup DR (sampled once per env at training start), wired only for training.
    for key in (
        "wheel_friction",
        "base_mass",
        "base_com",
        "joint_gains",
        "encoder_bias",
    ):
        assert train.events_cfg[key].mode == "startup", key
        assert key not in play.events_cfg, key


def test_train_cfg_has_action_perturbation() -> None:
    train = _flat_cfg(play=False)
    play = _flat_cfg(play=True)
    # Training perturbs the applied command (latency + noise) so the policy tolerates a
    # noisy control loop in MuJoCo; play deploys the clean policy.
    assert train.action_noise_std > 0.0
    assert train.action_delay_steps != (0, 0)
    assert play.action_noise_std == 0.0
    assert play.action_delay_steps == (0, 0)


def test_lock_wheels_immobilizes_wheels_for_crab_walk_stage() -> None:
    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg

    locked = unitree_go2w_velocity_env_cfg(play=False, lock_wheels=True)
    free = unitree_go2w_velocity_env_cfg(play=False, lock_wheels=False)

    # Stage 1: the wheel velocity command is dead (scale 0) and the wheel joints are stiffly
    # damped, so the wheels act as rigid round feet — the robot must *step* (incl. sideways)
    # to track any command. The action space stays 16-dim so Stage 2 warm-starts cleanly.
    assert locked.actions_cfg["wheel_vel"].scale == 0.0
    assert locked.robot.actuators["wheel"].damping >= 20.0
    # Default (Stage 2 / shipped) still rolls the wheels — stance-stabilized damping (5.0)
    # but well below the stage-1 hard lock (20.0).
    assert free.actions_cfg["wheel_vel"].scale > 0.0
    assert free.robot.actuators["wheel"].damping < locked.robot.actuators["wheel"].damping
    # Lateral command is kept — that's the whole point of the crab-walk stage.
    assert locked.commands_cfg["twist"].ranges.lin_vel_y != (0.0, 0.0)


def test_lock_wheels_adds_gait_shaping_rewards() -> None:
    """Stage 1 needs the legged gait-shaping rewards (Go1 recipe) or it converges to the
    stand-still optimum — probed: a 6k run without them tracks 0–16 % of any command.
    ``feet_air_time`` is unearnable while standing (command-gated), so it is the gradient
    that forces stepping; ``feet_slip`` keeps the planted wheels from skating."""
    from genelab import mdp
    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg

    locked = unitree_go2w_velocity_env_cfg(play=False, lock_wheels=True)
    free = unitree_go2w_velocity_env_cfg(play=False, lock_wheels=False)

    air = locked.rewards_cfg["air_time"]
    assert air.func is mdp.feet_air_time
    assert air.weight > 0.0
    assert air.params["sensor_name"] == "wheel_contact"
    assert air.params["command_name"] == "twist"  # gated: no free reward while standing
    slip = locked.rewards_cfg["foot_slip"]
    assert slip.func is mdp.feet_slip
    assert slip.weight < 0.0
    # Stage 1 gates air_time on full command magnitude (every command needs stepping).
    assert "command_axes" not in air.params

    # The rolling-wheel task keeps a *lateral-gated* air_time: wheels can't roll sideways,
    # so stepping must stay rewarded when vy is demanded — without it, stage-2 warm-start
    # forgets the crab-walk (probed: vy tracking collapsed 93 % -> 4 %). Pure forward/yaw
    # commands keep the gate closed so rolling stays optimal there. No foot_slip — it
    # would penalize normal wheel rolling.
    lat = free.rewards_cfg["air_time"]
    assert lat.func is mdp.feet_air_time
    assert lat.weight >= 0.5  # 0.25 was drowned out by the stability terms (probed)
    # Gate on |cmd_vy| + |cmd_wz|: wheels can't strafe at all, and they can't *scrub-turn
    # slowly* either (stiction deadband at near-zero wheel speeds — probed: wz=0.2 tracked
    # at 32 % while wz=1.0 hit 96 %), so stepping must stay rewarded for both lateral and
    # rotation demand. Pure-vx keeps the gate closed (rolling is optimal there).
    assert lat.params["command_axes"] == (1, 2)
    assert "foot_slip" not in free.rewards_cfg


def test_unlocked_task_fights_lateral_abandonment() -> None:
    """The rolling-wheel task needs an unsaturated vy gradient + stable stance feet.

    Probed twice: warm-started stage 2 abandons the crab-walk (vy 93 % -> 4 %) because
    (a) the exp tracking kernel's gradient vanishes once vy is fully ignored, and
    (b) free-spinning stance wheels (damping 0.5) make stepping fall-prone. The L1 vy
    error keeps a constant pull back toward lateral tracking; the stiffer wheel damping
    gives the stance leg a braked foot to push from."""
    from genelab import mdp
    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg

    free = unitree_go2w_velocity_env_cfg(play=False, lock_wheels=False)
    vy_err = free.rewards_cfg["vy_error_l1"]
    assert vy_err.func is mdp.velocity_tracking_error_l1
    assert vy_err.weight < 0.0
    assert vy_err.params["axes"] == (1,)
    # Yaw needs the same unsaturated pull: in-place rotation collapsed to 2 % at 6k while
    # mixed-command yaw still scored (the exp kernel masks per-command abandonment).
    wz_err = free.rewards_cfg["wz_error_l1"]
    assert wz_err.func is mdp.angular_velocity_tracking_error_l1
    assert wz_err.weight < 0.0
    # Stance-stabilizing wheel damping: well above the free-rolling 0.5, below the
    # stage-1 hard lock (20).
    assert 2.0 <= free.robot.actuators["wheel"].damping <= 10.0


def test_ppo_cfg_enables_mirror_symmetry_augmentation() -> None:
    """Go2-W PPO trains with mirror data augmentation (rsl_rl ``Symmetry``).

    Probed across three stage-2 fine-tunes: without it, PPO collapses to a one-sided
    lateral gait (one direction tracks 64/64, the mirror direction falls 64/64, and the
    surviving side flips between runs). Mirroring every mini-batch makes both directions
    learn in lock-step."""
    from genelab_unitree.go2w import unitree_go2w_ppo_runner_cfg

    sym = unitree_go2w_ppo_runner_cfg().algorithm.symmetry_cfg
    assert sym is not None
    assert sym["use_data_augmentation"] is True
    assert sym["data_augmentation_func"] == (
        "genelab_unitree.go2w.symmetry:mirror_go2w_obs_actions"
    )


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

        wheel_idx = [
            i for i, n in enumerate(env._articulation.joint_names) if n.endswith("_wheel_joint")
        ]
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


def test_train_env_builds_with_dr_history_and_action_delay(genesis_runtime: Any) -> None:
    """End-to-end: the training cfg (frame stacking + startup DR + action latency) builds and
    steps in real Genesis. Exercises the DR setters against go2w's actual link / DoF indices
    and confirms the policy obs is the 5-frame stack the deployed policy will expect."""
    del genesis_runtime

    from genelab_unitree.go2w import unitree_go2w_velocity_env_cfg

    env = _build_cpu_env(unitree_go2w_velocity_env_cfg(play=False))
    try:
        obs, _ = env.reset()
        # The startup DR events (friction / mass / COM / PD gains / encoder bias) just fired
        # against the real articulation without raising on a bad link/DoF index.
        buf = env.observation_manager._history["policy"]
        assert buf.shape[1] == 5  # five stacked control-step frames
        assert obs["policy"].shape[1] == buf.shape[1] * buf.shape[2]

        n_act = env.action_manager.total_action_dim
        action = torch.zeros(env.num_envs, n_act, device=env.device)
        # Step through more than the max latency so the action-delay ring buffer rolls.
        for _ in range(4):
            obs, *_ = env.step(action)
        # History width is stable across steps (the stack stays 5 frames wide).
        assert obs["policy"].shape[1] == buf.shape[1] * buf.shape[2]
    finally:
        env.close()
