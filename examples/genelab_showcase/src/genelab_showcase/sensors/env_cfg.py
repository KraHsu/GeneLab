"""Sensors showcase env: Franka + Camera + IMU + FrameTransformer + ForceTorque sensors."""

from typing import TYPE_CHECKING

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.recording import PyQtPlotCfg, RecordingCfg
from genelab.sensor import (
    CameraSensorCfg,
    ForceTorqueSensorCfg,
    FrameTransformerSensorCfg,
    IMUSensorCfg,
    TargetFrameCfg,
)

if TYPE_CHECKING:
    import torch


def _imu_accel(env: object) -> "dict[str, torch.Tensor]":
    """Live source: hand-IMU linear + angular acceleration (body frame), env 0.

    Returned as a dict so the PyQt plot renders two stacked subplots (one per label key).
    """
    data = env.sensors["hand_imu"].data  # type: ignore[attr-defined]
    return {
        "linear (m/s^2)": data.lin_acc_b[0],
        "angular (rad/s^2)": data.ang_acc_b[0],
    }


def _frame_positions(env: object) -> "dict[str, torch.Tensor]":
    """Live source: hand and link7 positions expressed in the link0 (base) frame, env 0."""
    data = env.sensors["hand_in_base"].data  # type: ignore[attr-defined]
    return {
        "hand in link0 (m)": data.target_pos_source[0, 0],
        "link7 in link0 (m)": data.target_pos_source[0, 1],
    }


def sensors_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with three sensors bolted to the end-effector link.

    The showcase runs with ``num_envs=1``, so the default Genesis Rasterizer suffices
    — ``CameraSensor._compute_data`` normalises the renderer's per-env output into a
    leading-batch torch tensor regardless. Switch to ``batch_render=True`` when
    scaling to many envs (Linux x86-64 + CUDA + recent nvJitLink only — Madrona's
    JIT linker is sensitive to driver / toolkit skew).
    """

    robot_cfg = FrankaPandaCfg()
    # Wrist camera looks along the gripper's approach axis — the Franka ``hand`` link's +z,
    # i.e. the end-effector forward direction. Genesis cameras look down their own local -z,
    # so ``offset_quat`` is a 180° rotation about y (wxyz ``(0, 0, 1, 0)``) that maps camera
    # -z → hand +z. (Solved empirically: gripper-forward is hand +z; the previous quat aimed
    # the camera at hand +x, perpendicular to the gripper.)
    camera_quat = (0.0, 0.0, 1.0, 0.0)

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
            # Static props on the table in front of the robot so the wrist camera's RGB and
            # depth streams show real geometry (boxes + spheres at varied size / distance)
            # instead of an empty plane as joint1 sweeps the camera across them.
            entities={
                "prop_box_a": RigidObjectCfg(morph="box", size=(0.08, 0.08, 0.08), init_pos=(0.45, 0.0, 0.04)),
                "prop_box_b": RigidObjectCfg(morph="box", size=(0.06, 0.12, 0.10), init_pos=(0.40, 0.22, 0.05)),
                "prop_box_c": RigidObjectCfg(morph="box", size=(0.12, 0.06, 0.06), init_pos=(0.52, -0.20, 0.03)),
                "prop_tower": RigidObjectCfg(morph="box", size=(0.05, 0.05, 0.16), init_pos=(0.58, -0.04, 0.08)),
                "prop_ball_a": RigidObjectCfg(morph="sphere", size=(0.05,), init_pos=(0.55, 0.13, 0.05)),
                "prop_ball_b": RigidObjectCfg(morph="sphere", size=(0.035,), init_pos=(0.37, -0.10, 0.035)),
            },
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
            # Real-time display (no disk dumps): one window per sensor. Numeric sensors
            # stream to live PyQt curves; the wrist camera streams to live image windows.
            recordings=(
                RecordingCfg(
                    name="hand_imu_accel",
                    source=_imu_accel,
                    env_idx=None,
                    outputs=(
                        PyQtPlotCfg(
                            title="hand IMU acceleration",
                            labels={
                                "linear (m/s^2)": ("ax", "ay", "az"),
                                "angular (rad/s^2)": ("wx", "wy", "wz"),
                            },
                            history_length=200,
                        ),
                    ),
                ),
                RecordingCfg(
                    name="arm_joint_force",
                    source="arm_joint_ft",
                    field="force",
                    outputs=(
                        PyQtPlotCfg(
                            title="arm joint reaction force/torque",
                            labels=("j1", "j2", "j3", "j4", "j5", "j6", "j7"),
                            history_length=200,
                        ),
                    ),
                ),
                RecordingCfg(
                    name="frames_in_base",
                    source=_frame_positions,
                    env_idx=None,
                    outputs=(
                        PyQtPlotCfg(
                            title="frame transformer (positions in link0)",
                            labels={
                                "hand in link0 (m)": ("x", "y", "z"),
                                "link7 in link0 (m)": ("x", "y", "z"),
                            },
                            history_length=200,
                        ),
                    ),
                ),
                # The wrist camera (RGB + depth) is rendered into a live pyqtgraph window by
                # the runner (Qt, smooth) instead of a recording sink — see the runner.
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
