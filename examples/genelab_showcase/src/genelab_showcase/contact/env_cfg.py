"""Contact showcase env: Unitree G1 with ContactSensor on both ankle-roll links."""

from genelab import mdp
from genelab.asset_zoo import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg

from genelab.sensor import ContactSensorCfg

_FOOT_LINKS: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")


def contact_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env G1 standing on flat ground with foot contact-and-air-time tracking.

    The robot is reset to the default pose every episode (``episode_length_s=4``) so the
    body settles, the feet make contact, and the contact sensor's ``current_contact_time``
    starts ticking. The runner prints the running per-foot air / contact time every
    20 control steps.
    """

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
                    link_names=_FOOT_LINKS,
                    force_threshold=1.0,
                    track_air_time=True,
                ),
            ),
        ),
        decimation=4,
        episode_length_s=4.0,
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
