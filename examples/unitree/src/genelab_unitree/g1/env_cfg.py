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
from genelab.sensor import BodyVelocitySensorCfg, ContactSensorCfg
from genelab_unitree.g1.constants import G1_ACTION_SCALE
from genelab_unitree.g1.robot import get_g1_robot_cfg

# IMU site offset from the pelvis link origin; matches g1.xml's <site name="imu_in_pelvis">.
_IMU_OFFSET = (0.04525, 0.0, -0.08339)
_G1_FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")

# Per-joint posture standard deviations used by the ``pose`` reward. Smaller std =
# tighter tolerance. Lower body looser at running speed; arms/wrists looser to allow
# natural swing. Mirrors mjlab's G1 std table verbatim.
_G1_POSE_STD_WALKING: dict[str, float] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
}

_G1_POSE_STD_RUNNING: dict[str, float] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
}


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
    terms = _obs_terms()
    terms["foot_air_time"] = ObservationTermCfg(
        func=mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    )
    terms["foot_contact"] = ObservationTermCfg(
        func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    )
    terms["foot_contact_forces"] = ObservationTermCfg(
        func=mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"}
    )
    return ObservationGroupCfg(enable_corruption=False, terms=terms)


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
                ContactSensorCfg(
                    name="feet_ground_contact",
                    link_names=_G1_FOOT_LINKS,
                    track_air_time=True,
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
                    lin_vel_y=(-1.0, 1.0),
                    ang_vel_z=(-0.5, 0.5),
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
            "upright": RewardTermCfg(
                func=mdp.upright_exp,
                weight=1.0,
                params={"std": math.sqrt(0.2)},
            ),
            "pose": RewardTermCfg(
                func=mdp.variable_posture,
                weight=1.0,
                params={
                    "command_name": "twist",
                    "walking_threshold": 0.05,
                    "running_threshold": 1.5,
                    "default_std": 0.3,
                    "std_standing": {".*": 0.05},
                    "std_walking": _G1_POSE_STD_WALKING,
                    "std_running": _G1_POSE_STD_RUNNING,
                },
            ),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
            "dof_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
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
                    "pose_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (0.01, 0.05),
                        "yaw": (-math.pi, math.pi),
                    },
                    "velocity_range": {},
                },
            ),
            "reset_joints": EventTermCfg(mode="reset", func=mdp.reset_joints_to_default),
        },
    )
    if not play:
        cfg.events_cfg["push_robot"] = EventTermCfg(
            mode="interval",
            interval_range_s=(1.0, 3.0),
            func=mdp.push_by_setting_velocity,
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        )
    return cfg
