"""Trainable Unitree Go1 velocity tracking on analytic deformable (soft) terrain.

Stage 1: the feet are supported by the analytic compliance + traction model over a virtual
surface (no rigid floor under the feet), so the robot can get grip and learn to walk. Because
the feet make no *rigid* contact, Genesis contact sensors are blind here — the reward set is
built from non-contact terms (velocity tracking, base stability, actuation smoothness) plus
the privileged soft-terrain shaping (sinkage). Footprint memory and the privileged terrain
state are available to the critic.

Geometry mirrors :mod:`go1_soft_env_cfg` (the validated stand demo): the robot is raised so
its feet settle on the virtual surface at :data:`_SURFACE_HEIGHT` and stay above the ``z=0``
backstop plane.
"""

import math

from genelab import mdp
from genelab.asset_zoo.unitree_go1 import UnitreeGo1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import BodyVelocitySensorCfg
from genelab.terrains import DeformableTerrainCfg

_TRUNK_LINK = "trunk"
GO1_FOOT_LINKS: tuple[str, ...] = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
# Foot contact site in the calf's local frame (the Go1 foot is 0.213 m down the calf).
# Support + traction now act here (correct lever), so the surface is referenced to the real
# foot point: ~ground level, robot at its natural pose, feet on the (rendered) ground.
FOOT_OFFSET = (0.0, 0.0, -0.213)
_SURFACE_HEIGHT = 0.02
_GROUND_RENDER_HEIGHT = 0.0
_STIFFNESS = 8000.0
_DAMPING = 400.0
_FRICTION = 0.8
_LATERAL_STIFFNESS = 3000.0
_BASE_HEIGHT_TARGET = 0.34


def _trunk_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(name="robot", link_names=(_TRUNK_LINK,))


def track_velocity_l1(env, command_name: str):  # type: ignore[no-untyped-def]
    """``-|cmd_xy - base_vel_b_xy|`` — a *no-floor* velocity-tracking penalty.

    Unlike the ``exp`` tracking reward (which pays a positive floor even at zero velocity,
    making "stand still" profitable under the never-falling analytic support), this is
    proportional to the error: standing while commanded to move costs ``-|cmd|`` per step,
    so the policy is pushed to actually translate.
    """
    import torch

    robot = env.articulations["robot"]
    base_vel = robot.data.root_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return -torch.norm(base_vel - cmd, dim=-1)


def _proprio_obs() -> dict[str, ObservationTermCfg]:
    """Deployable proprioception (no base linear velocity — Go1 can't sense it)."""
    return {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data, params={"sensor_name": "imu_ang_vel"}, noise=Unoise(-0.2, 0.2)
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(-0.05, 0.05)
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(-1.5, 1.5)),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
    }


def go1_soft_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Trainable velocity-tracking env for the Go1 on analytic deformable terrain."""
    robot = UnitreeGo1Cfg()
    bx, by, _ = robot.init_pos
    robot.init_pos = (bx, by, 0.40)  # natural spawn; feet drop onto the ~ground-level surface

    policy_terms = _proprio_obs()
    critic_terms = _proprio_obs()
    # Privileged critic obs: true base linear velocity + the simulator's terrain sinkage.
    critic_terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.sensor_data, params={"sensor_name": "imu_lin_vel"}
    )
    critic_terms["terrain_sinkage"] = ObservationTermCfg(func=mdp.terrain_sinkage)

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            dt=0.005,
            substeps=1,
            vis=play,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            terrain=None,
            ground_plane_collision=False,
            ground_plane_height=_GROUND_RENDER_HEIGHT,
            sensors=(
                BodyVelocitySensorCfg(name="imu_lin_vel", link_name=_TRUNK_LINK, measure="lin_vel"),
                BodyVelocitySensorCfg(name="imu_ang_vel", link_name=_TRUNK_LINK, measure="ang_vel"),
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
        device="cuda",
        robot=robot,
        deformable_terrain=DeformableTerrainCfg(
            k=_STIFFNESS,
            c=_DAMPING,
            mu=_FRICTION,
            # Stick-slip lateral hold (static friction + sinkage shear): a planted foot
            # anchors and resists sideways displacement, giving a stable stance to push off
            # from — the physical ingredient missing in iters 1-5 (velocity-only friction).
            lateral_stiffness=_LATERAL_STIFFNESS,
            surface_height=_SURFACE_HEIGHT,
            foot_link_names=GO1_FOOT_LINKS,
            foot_offset=FOOT_OFFSET,
        ),
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot", joint_names=(".*",), use_default_offset=True
            )
        },
        observations_cfg={
            "policy": ObservationGroupCfg(enable_corruption=True, terms=policy_terms),
            "critic": ObservationGroupCfg(enable_corruption=False, terms=critic_terms),
        },
        commands_cfg={
            "twist": UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(3.0, 8.0),
                rel_standing_envs=0.0,  # always commanded to move — no degenerate standing
                rel_heading_envs=0.0,
                rel_forward_envs=0.2,
                heading_command=False,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-1.0, 1.0),
                    heading=(-math.pi, math.pi),
                ),
            )
        },
        rewards_cfg={
            # --- velocity tracking ---
            "track_lin_vel": RewardTermCfg(
                func=mdp.track_linear_velocity_xy_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            # No-floor movement penalty: makes standing-still cost ~-|cmd|/step so the
            # robot is forced to actually translate (not park on the exp floor).
            "track_lin_vel_l1": RewardTermCfg(
                func=track_velocity_l1, weight=1.5, params={"command_name": "twist"}
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=1.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            # --- base stability ---
            "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
            "ang_vel_xy": RewardTermCfg(
                func=mdp.body_angular_velocity_penalty,
                weight=-0.05,
                params={"asset_cfg": _trunk_cfg()},
            ),
            "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
            "base_height": RewardTermCfg(
                func=mdp.base_height_l2, weight=-1.0, params={"target_height": _BASE_HEIGHT_TARGET}
            ),
            # --- actuation / smoothness ---
            "joint_torques": RewardTermCfg(func=mdp.applied_torque_l2, weight=-2.0e-4),
            "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-5),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
            "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            # --- gait shaping (terrain-derived contact, since analytic support has no
            #     rigid contact for sensors): reward real steps so it walks, not stands ---
            "feet_air_time": RewardTermCfg(
                func=mdp.terrain_feet_air_time,
                weight=2.0,
                params={
                    "command_name": "twist",
                    "threshold_min": 0.0,  # reward any swing — don't punish short bootstrap steps
                    "threshold_max": 0.5,
                    "command_threshold": 0.1,
                },
            ),
            # Penalise sliding a planted foot — stops the "skating" gait (translate by
            # dragging feet) so the only cheap way to move is to actually step.
            "feet_slip": RewardTermCfg(
                func=mdp.terrain_feet_slip,
                weight=-0.1,  # lighter now: the stick-slip physics already resists skating
                params={"command_name": "twist", "command_threshold": 0.1},
            ),
            # --- soft-terrain shaping (paper §8.3) ---
            "terrain_sinkage": RewardTermCfg(func=mdp.terrain_sinkage_l2, weight=-0.5),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
            ),
        },
        events_cfg={
            "reset_base": EventTermCfg(
                mode="reset",
                func=mdp.reset_root_state_uniform,
                params={
                    "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
                    "velocity_range": {},
                },
            ),
            "reset_robot_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
    )

    if play:
        # Manual control during play: don't auto-reset on stumble/timeout — the Reset
        # button below re-spawns the robot, and the sliders drive the velocity command.
        cfg.auto_reset = False
        cfg.simulation.camera_lookat = (0.0, 0.0, 0.3)
        cfg.simulation.camera_pos = (1.8, -1.8, 1.0)
        from importlib.util import find_spec

        if find_spec("imgui_bundle") is not None:  # the in-viewport overlay needs it at draw time
            from genelab_soft_terrain.teleop import ImGuiTwistResetBridgeCfg

            cfg.simulation.viewer_imgui = True
            cfg.bridges_cfg["teleop"] = ImGuiTwistResetBridgeCfg(
                command_name="twist",
                vx_range=(-1.0, 1.0),
                vy_range=(-1.0, 1.0),
                wz_range=(-1.0, 1.0),
                default_vx=0.5,  # start walking forward (in-distribution); drag to 0 to stand
                default_vy=0.0,
                default_wz=0.0,
            )

    return cfg
