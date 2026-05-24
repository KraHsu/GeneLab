"""Sensors showcase env: Franka + Camera + IMU + FrameTransformer + ForceTorque sensors."""

import math

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.sensor import (
    CameraSensorCfg,
    ForceTorqueSensorCfg,
    FrameTransformerSensorCfg,
    IMUSensorCfg,
    TargetFrameCfg,
)


def sensors_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with three sensors bolted to the end-effector link.

    The showcase runs with ``num_envs=1``, so the default Genesis Rasterizer suffices
    — ``CameraSensor._compute_data`` normalises the renderer's per-env output into a
    leading-batch torch tensor regardless. Switch to ``batch_render=True`` when
    scaling to many envs (Linux x86-64 + CUDA + recent nvJitLink only — Madrona's
    JIT linker is sensitive to driver / toolkit skew).
    """

    robot_cfg = FrankaPandaCfg()
    # Camera mounted on the Franka ``hand`` body, looking along the gripper's
    # closing axis. wxyz quat (cos(-45°), 0, sin(-45°), 0) rotates camera +x →
    # link +z, which is the gripper-forward direction in Franka's MJCF.
    camera_quat = (math.cos(math.radians(-45.0)), 0.0, math.sin(math.radians(-45.0)), 0.0)

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=0.01,
            substeps=2,
            steps=200,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            batch_render=False,
            sensors=(
                CameraSensorCfg(
                    name="hand_camera",
                    link_name="hand",
                    offset_pos=(0.0, 0.0, 0.06),
                    offset_quat=camera_quat,
                    width=160,
                    height=120,
                    fov=70.0,
                    near=0.02,
                    far=2.5,
                    render_rgb=True,
                    render_depth=True,
                ),
                IMUSensorCfg(
                    name="hand_imu",
                    link_name="hand",
                    offset=(0.0, 0.0, 0.04),
                    gravity_bias=True,
                ),
                FrameTransformerSensorCfg(
                    name="hand_in_base",
                    source_link_name="link0",
                    target_frames=(
                        TargetFrameCfg(link_name="hand", name="hand"),
                        TargetFrameCfg(link_name="link7", name="link7"),
                    ),
                ),
                # Per-joint reaction force/torque on the 7 actuated arm joints — the
                # ones the scripted joint1 sweep drives. ``data.force`` is (num_envs, 7).
                ForceTorqueSensorCfg(
                    name="arm_joint_ft",
                    joint_names_expr=r"^joint[1-7]$",
                ),
            ),
        ),
        decimation=2,
        episode_length_s=20.0,
        device="cuda",
        robot=robot_cfg,
        actions_cfg={
            "panda_arm": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(r"^joint[1-7]$",),
                use_default_offset=True,
            ),
            "panda_hand": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(r"finger_joint.*",),
                use_default_offset=True,
            ),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_to_default,
                params={"pos_jitter": 0.0, "vel_jitter": 0.0},
            ),
        },
    )
