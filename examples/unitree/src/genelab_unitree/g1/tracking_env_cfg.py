"""Motion-imitation env config for the Unitree G1 on flat ground.

Port of ``mjlab.tasks.tracking.config.g1.env_cfgs.unitree_g1_flat_tracking_env_cfg`` adapted to
GeneLab's slim manager system. The motion clip defaults to the LAFAN1 ``dance1_subject2``
NPZ vendored in the asset zoo (fetched lazily on first use via
:func:`genelab.asset_zoo.unitree_g1_motions.g1_lafan1_dance1_subject2`). Edit the cfg or
swap the helper to point at any other NPZ that matches
:class:`genelab.mdp.commands.motion_command.MotionLoader`'s schema.
"""

import math

from genelab import mdp
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.asset_zoo.unitree_g1 import UnitreeG1Cfg
from genelab.asset_zoo.unitree_g1_motions import (
    G1_MJLAB_BODY_NAMES,
    G1_MJLAB_JOINT_NAMES,
    g1_lafan1_dance1_subject2,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.commands.motion_command import MotionCommandCfg


_G1_TRACKING_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

_EE_BODIES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def _policy_obs_group(*, play: bool) -> ObservationGroupCfg:
    """Actor obs — fed to the policy.

    Drops privileged signals (per-body pose/ori). During play we keep the same signals as train
    so the same policy weights load.
    """
    return ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "motion"}
            ),
            "motion_anchor_pos_b": ObservationTermCfg(
                func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
            ),
            "motion_anchor_ori_b": ObservationTermCfg(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
            ),
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
        }
    )


def _critic_obs_group() -> ObservationGroupCfg:
    """Critic obs — adds per-body privileged signals."""
    return ObservationGroupCfg(
        terms={
            "command": ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "motion"}
            ),
            "motion_anchor_pos_b": ObservationTermCfg(
                func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
            ),
            "motion_anchor_ori_b": ObservationTermCfg(
                func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
            ),
            "body_pos": ObservationTermCfg(
                func=mdp.robot_body_pos_b, params={"command_name": "motion"}
            ),
            "body_ori": ObservationTermCfg(
                func=mdp.robot_body_ori_b, params={"command_name": "motion"}
            ),
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
        }
    )


def unitree_g1_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Flat-ground motion-imitation env config for the Unitree G1."""
    robot_entity_cfg = UnitreeG1Cfg()

    cfg = ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            dt=0.002,
            substeps=1,
            vis=play,
            # Run Genesis on the GPU — see the note in ``unitree_g1_velocity_env_cfg``.
            # ``SimulationCfg.gpu`` defaults to False (CPU backend), which would run the
            # humanoid physics on CPU (GPU idle, ~30x slower per step).
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
        ),
        decimation=10,
        episode_length_s=10.0 if not play else 1e9,
        device="cuda",
        robot=robot_entity_cfg,
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
                use_default_offset=True,
            )
        },
        observations_cfg={
            "policy": _policy_obs_group(play=play),
            "critic": _critic_obs_group(),
        },
        commands_cfg={
            "motion": MotionCommandCfg(
                asset_name="robot",
                # Sampling is driven by clip end (handled inside MotionCommand._update_command),
                # so we set a huge resampling period to disable the time-based path.
                resampling_time_range=(1.0e9, 1.0e9),
                motion_file=str(g1_lafan1_dance1_subject2()),
                anchor_body_name="torso_link",
                body_names=_G1_TRACKING_BODIES,
                motion_body_order=G1_MJLAB_BODY_NAMES,
                motion_joint_order=G1_MJLAB_JOINT_NAMES,
                pose_range=(
                    {
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (-0.01, 0.01),
                        "roll": (-0.1, 0.1),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-0.2, 0.2),
                    }
                    if not play
                    else {}
                ),
                velocity_range=_VELOCITY_RANGE if not play else {},
                joint_position_range=(-0.1, 0.1) if not play else (0.0, 0.0),
                sampling_mode="uniform" if not play else "start",
            )
        },
        rewards_cfg={
            "motion_global_root_pos": RewardTermCfg(
                func=mdp.motion_global_anchor_position_error_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.3},
            ),
            "motion_global_root_ori": RewardTermCfg(
                func=mdp.motion_global_anchor_orientation_error_exp,
                weight=0.5,
                params={"command_name": "motion", "std": 0.4},
            ),
            "motion_body_pos": RewardTermCfg(
                func=mdp.motion_relative_body_position_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.3},
            ),
            "motion_body_ori": RewardTermCfg(
                func=mdp.motion_relative_body_orientation_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.4},
            ),
            "motion_body_lin_vel": RewardTermCfg(
                func=mdp.motion_global_body_linear_velocity_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 1.0},
            ),
            "motion_body_ang_vel": RewardTermCfg(
                func=mdp.motion_global_body_angular_velocity_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": math.pi},
            ),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
            "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "anchor_pos": TerminationTermCfg(
                func=mdp.bad_anchor_pos_z_only,
                params={"command_name": "motion", "threshold": 0.25},
            ),
            "anchor_ori": TerminationTermCfg(
                func=mdp.bad_anchor_ori,
                params={"command_name": "motion", "threshold": 0.8},
            ),
            "ee_body_pos": TerminationTermCfg(
                func=mdp.bad_motion_body_pos_z_only,
                params={
                    "command_name": "motion",
                    "threshold": 0.25,
                    "body_names": _EE_BODIES,
                },
            ),
        },
        events_cfg={},
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
