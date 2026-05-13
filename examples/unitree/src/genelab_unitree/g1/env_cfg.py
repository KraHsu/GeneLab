"""Velocity-tracking env config for the Unitree G1 on flat ground.

Mirrors ``mjlab.tasks.velocity.config.g1`` adapted to GeneLab's slim manager system.
"""

import math

from genelab import mdp
from genelab.configs import SceneCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.commands.velocity_command import UniformVelocityCommandCfg
from genelab.mdp.noise import Unoise
from genelab.sensor import BodyVelocitySensorCfg
from genelab_unitree.g1.constants import G1_ACTION_SCALE
from genelab_unitree.g1.robot import get_g1_robot_cfg

# IMU site offset from the pelvis link origin; matches g1.xml's <site name="imu_in_pelvis">.
_IMU_OFFSET = (0.04525, 0.0, -0.08339)


def _obs_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_lin_vel"},
            noise=Unoise(-0.5, 0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.sensor_data,
            params={"sensor_name": "imu_ang_vel"},
            noise=Unoise(-0.2, 0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(-0.05, 0.05)
        ),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "twist"}
        ),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01)),
        # Unoise lives in raw rad/s; the existing scale=0.05 brings it to ±0.075 final.
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(-1.5, 1.5)
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }


def _policy_obs_group() -> ObservationGroupCfg:
    return ObservationGroupCfg(enable_corruption=True, terms=_obs_terms())


def _critic_obs_group() -> ObservationGroupCfg:
    return ObservationGroupCfg(enable_corruption=False, terms=_obs_terms())


def unitree_g1_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity-tracking env config for the Unitree G1."""
    robot_entity_cfg = get_g1_robot_cfg().to_entity_cfg()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=4096 if not play else 50,
            dt=0.002,
            substeps=1,
            env_spacing=(2.5, 2.5),
            vis=play,
            sensors=(
                BodyVelocitySensorCfg(
                    name="imu_lin_vel",
                    link_name="pelvis",
                    offset=_IMU_OFFSET,
                    measure="lin_vel",
                ),
                BodyVelocitySensorCfg(
                    name="imu_ang_vel",
                    link_name="pelvis",
                    offset=_IMU_OFFSET,
                    measure="ang_vel",
                ),
            ),
        ),
        decimation=10,
        episode_length_s=20.0,
        device="cuda",
        robot=robot_entity_cfg,
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
                scale=dict(G1_ACTION_SCALE),
                use_default_offset=True,
            )
        },
        observations_cfg={
            "policy": _policy_obs_group(),
            "critic": _critic_obs_group(),
        },
        commands_cfg={
            "twist": UniformVelocityCommandCfg(
                asset_name="robot",
                resampling_time_range=(3.0, 8.0),
                rel_standing_envs=0.1,
                heading_command=True,
                heading_control_stiffness=0.5,
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(-1.0, 1.0),
                    lin_vel_y=(-0.5, 0.5),
                    ang_vel_z=(-0.7, 0.7),
                    heading=(-math.pi, math.pi),
                ),
            )
        },
        rewards_cfg={
            "track_lin_vel": RewardTermCfg(
                func=mdp.track_linear_velocity_xy_exp,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.25)},
            ),
            "track_ang_vel": RewardTermCfg(
                func=mdp.track_angular_velocity_z_exp,
                weight=2.0,
                params={"command_name": "twist", "std": math.sqrt(0.5)},
            ),
            "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
            "dof_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
            "feet_air_time": RewardTermCfg(
                func=mdp.feet_air_time, weight=0.5, params={"threshold": 0.4}
            ),
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
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05)},
                    "velocity_range": {},
                },
            ),
            "reset_joints": EventTermCfg(mode="reset", func=mdp.reset_joints_to_default),
        },
    )
    if not play:
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(10.0, 15.0),
            func=mdp.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
    return cfg
