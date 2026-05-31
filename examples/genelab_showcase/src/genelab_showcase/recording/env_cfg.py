"""Recording showcase env: Franka with an IMU on the hand, wired to live plots.

Demonstrates the recording layer's live-plot sinks for two data sources:

* IMU ``lin_acc_b`` → live PyQt plot
* Joint 1 position (custom callable, not a sensor) → live PyQt plot

**File dumps are opt-in.** By default the showcase only opens the two live plot windows and
writes nothing to disk. Set the ``GENELAB_SHOWCASE_RECORD_FILES=1`` environment variable to
also dump ``logs/showcase/recording/{lin_acc.csv, orientation_*.npz}`` (the CSV is added to
the lin-acc recording and an extra IMU ``orientation`` → NPZ recording, written per-episode
via ``save_on_reset=True``). This keeps the default run from silently filling the disk.

Run with ``genelab play GeneLab-Recording-Showcase-v0 --vis --steps 400`` (live plots only),
or ``GENELAB_SHOWCASE_RECORD_FILES=1 genelab play …`` to also write the files.
"""

import os

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.recording import (
    CSVFileCfg,
    NPZFileCfg,
    OutputCfg,
    PyQtPlotCfg,
    RecordingCfg,
)
from genelab.sensor import IMUSensorCfg

# When set (to any non-empty value), the showcase also writes CSV + NPZ files to disk.
_RECORD_FILES_ENV = "GENELAB_SHOWCASE_RECORD_FILES"


def _joint1_pos(env: object) -> object:
    """Custom callable source: pull the first arm joint's position via the env handle."""
    # ``env`` is the live ``ManagerBasedRlEnv``. ``robot_state.joint_pos`` is a
    # ``(num_envs, n_joints)`` tensor; the env_idx squeeze on the RecordingCfg
    # reduces it to a single scalar per sample.
    return env.articulations["robot"].data.joint_pos[:, 0]  # type: ignore[attr-defined]


def recording_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with one IMU; live plots always, file dumps opt-in (see module doc)."""

    robot_cfg = FrankaPandaCfg()
    record_files = bool(os.environ.get(_RECORD_FILES_ENV))

    # lin_acc always drives the live PyQt plot; the CSV file is opt-in.
    lin_acc_outputs: list[OutputCfg] = [
        PyQtPlotCfg(
            title="hand IMU linear acceleration (body frame)",
            labels=("ax", "ay", "az"),
            history_length=200,
        ),
    ]
    if record_files:
        lin_acc_outputs.append(
            CSVFileCfg(
                filename="logs/showcase/recording/lin_acc.csv",
                header=("ax", "ay", "az"),
            )
        )
    recordings: list[RecordingCfg] = [
        RecordingCfg(
            name="lin_acc", source="hand_imu", field="lin_acc_b", outputs=tuple(lin_acc_outputs)
        ),
        RecordingCfg(
            name="joint1",
            source=_joint1_pos,
            outputs=(
                PyQtPlotCfg(title="joint1 position (rad)", labels=("q1",), history_length=200),
            ),
        ),
    ]
    if record_files:
        # orientation has only a file sink, so it exists only when file dumps are enabled.
        recordings.append(
            RecordingCfg(
                name="orientation",
                source="hand_imu",
                field="orientation",
                outputs=(
                    NPZFileCfg(
                        filename="logs/showcase/recording/orientation.npz", save_on_reset=True
                    ),
                ),
            )
        )

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
            recordings=tuple(recordings),
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
