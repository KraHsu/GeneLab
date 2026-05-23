"""Recording showcase env: Franka with an IMU on the hand, wired to live plot + file dumps.

Demonstrates three sinks for two data sources:

* IMU ``lin_acc_b`` → live PyQt plot + CSV dump
* IMU ``orientation`` → NPZ dump (per-episode via ``save_on_reset=True``)
* Joint 1 position (custom callable, not a sensor) → live MPL plot

Run with ``genelab play GeneLab-Recording-Showcase-v0 --vis --steps 400``. Two plot
windows appear (PyQt + MPL); on exit ``logs/showcase/recording/{lin_acc.csv,
orientation_*.npz}`` are written.
"""

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.recording import (
    CSVFileCfg,
    MPLPlotCfg,
    NPZFileCfg,
    PyQtPlotCfg,
    RecordingCfg,
)
from genelab.sensor import IMUSensorCfg


def _joint1_pos(env: object) -> object:
    """Custom callable source: pull the first arm joint's position via the env handle."""
    # ``env`` is the live ``ManagerBasedRlEnv``. ``robot_state.joint_pos`` is a
    # ``(num_envs, n_joints)`` tensor; the env_idx squeeze on the RecordingCfg
    # reduces it to a single scalar per sample.
    return env.articulations["robot"].data.joint_pos[:, 0]  # type: ignore[attr-defined]


def recording_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with one IMU plus three recording sinks."""

    robot_cfg = FrankaPandaCfg()

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=0.01,
            substeps=2,
            steps=400,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            sensors=(
                IMUSensorCfg(
                    name="hand_imu",
                    link_name="hand",
                    offset=(0.0, 0.0, 0.04),
                    gravity_bias=True,
                ),
            ),
            recordings=(
                RecordingCfg(
                    name="lin_acc",
                    source="hand_imu",
                    field="lin_acc_b",
                    outputs=(
                        PyQtPlotCfg(
                            title="hand IMU linear acceleration (body frame)",
                            labels=("ax", "ay", "az"),
                            history_length=200,
                        ),
                        CSVFileCfg(
                            filename="logs/showcase/recording/lin_acc.csv",
                            header=("ax", "ay", "az"),
                        ),
                    ),
                ),
                RecordingCfg(
                    name="orientation",
                    source="hand_imu",
                    field="orientation",
                    outputs=(
                        NPZFileCfg(
                            filename="logs/showcase/recording/orientation.npz",
                            save_on_reset=True,
                        ),
                    ),
                ),
                RecordingCfg(
                    name="joint1",
                    source=_joint1_pos,
                    outputs=(
                        MPLPlotCfg(
                            title="joint1 position (rad)",
                            labels=("q1",),
                            history_length=200,
                        ),
                    ),
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
