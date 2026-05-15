"""RayCast showcase env: Franka base with three ray-cast patterns side by side.

The env binds three :class:`RayCastSensor` instances against ``link0`` — one per
pattern — so a single play loop prints the distance summary for each. Swap any of the
three pattern objects in this file to retune (e.g., change ``num_rays_target``,
``polar_fov_deg``, or use a flat ``GridPattern`` for height-scan).
"""

from genelab import mdp
from genelab.asset_zoo import FrankaPandaCfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.sensor import (
    GridPattern,
    HemispherePattern,
    RayCastSensorCfg,
    RingPattern,
)


def raycast_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env Franka with three RayCast sensors anchored on ``link0``.

    All three patterns scan downward into the flat ground plane (``ground_height=0``).
    The grid is a small floor swatch; the ring is a planar 32-line sweep; the
    hemisphere is a Fibonacci cap looking straight down. Distances are clamped to
    ``max_distance`` so misses do not poison aggregate stats.
    """

    robot_cfg = FrankaPandaCfg()
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
            sensors=(
                RayCastSensorCfg(
                    name="raycast_grid",
                    link_name="hand",
                    pattern=GridPattern(
                        resolution=0.1,
                        size=(0.8, 0.8),
                        direction=(0.0, 0.0, -1.0),
                    ),
                    attach_yaw_only=True,
                    max_distance=2.0,
                    ground_height=0.0,
                ),
                RayCastSensorCfg(
                    name="raycast_ring",
                    link_name="hand",
                    pattern=RingPattern(
                        num_horizontal=32,
                        num_vertical=1,
                        horizontal_fov_deg=(-180.0, 180.0),
                        vertical_fov_deg=(-30.0, -30.0),
                    ),
                    attach_yaw_only=True,
                    max_distance=3.0,
                ),
                RayCastSensorCfg(
                    name="raycast_hemi",
                    link_name="hand",
                    pattern=HemispherePattern(
                        num_rays_target=128,
                        pole_axis=(0.0, 0.0, -1.0),
                        polar_fov_deg=70.0,
                    ),
                    attach_yaw_only=True,
                    max_distance=3.0,
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
