"""Contact showcase env: Unitree G1 marching in place with foot contact / air-time tracking.

A virtual spring (configured by the runner) suspends the pelvis so the passive G1 stays
upright while the runner scripts an alternating in-place step. The foot ContactSensor's
``current_air_time`` streams to a live PyQt plot (left / right), and the runner draws a
contact-state sphere under each foot. Nothing is written to disk.
"""

from typing import TYPE_CHECKING

from genelab import mdp
from genelab.asset_zoo import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.recording import PyQtPlotCfg, RecordingCfg
from genelab.sensor import ContactSensorCfg

if TYPE_CHECKING:
    import torch

FOOT_LINKS: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")


def _foot_air_time(env: object) -> "dict[str, torch.Tensor]":
    """Live source: per-foot time since lift-off (zero while the foot is planted)."""
    data = env.sensors["feet_contact"].data  # type: ignore[attr-defined]
    return {"air-time (s)": data.current_air_time[0]}


def contact_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env G1 marching in place, with foot contact / air-time tracking."""

    robot_cfg = UnitreeG1Cfg()
    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=1,
            dt=0.005,
            substeps=1,
            steps=200,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.5, 2.5),
            sensors=(
                ContactSensorCfg(
                    name="feet_contact",
                    link_names=FOOT_LINKS,
                    force_threshold=1.0,
                    track_air_time=True,
                ),
            ),
            recordings=(
                RecordingCfg(
                    name="foot_air_time",
                    source=_foot_air_time,
                    env_idx=None,
                    outputs=(
                        PyQtPlotCfg(
                            title="foot air-time (s)",
                            labels={"air-time (s)": ("left", "right")},
                            history_length=250,
                        ),
                    ),
                ),
            ),
        ),
        decimation=4,
        # Long episode so the march is continuous (no periodic reset mid-step).
        episode_length_s=1000.0,
        device="cuda",
        robot=robot_cfg,
        actions_cfg={
            "g1_joints": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(r".*",),
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
