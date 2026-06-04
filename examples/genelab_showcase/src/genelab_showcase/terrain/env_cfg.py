"""Terrain showcase env: a spring-suspended G1 toured across a 1×5 sub-terrain row.

Layout (one row, five columns) hits all five built-in sub-terrains in a single scene:

    | flat | stairs | rough | slope | wave |

The runner suspends the G1 with a virtual spring and walks it from cell to cell. A downward
pelvis ray-cast scans the surface under it; the runner draws the rays in the scene and renders
the hit point cloud + a height cross-section in live windows. Nothing is written to disk.
"""

from genelab import mdp
from genelab.asset_zoo import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.sensor import RayCastSensorCfg, GridPattern
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SlopeCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
    WaveCfg,
)


def terrain_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """Single-env G1 toured across a 1×5 row that exercises every sub-terrain type.

    A downward :class:`GridPattern` ray-cast on the pelvis scans the surface under the robot
    (11×11 rays); the runner turns it into the live point cloud + height cross-section.
    """

    robot_cfg = UnitreeG1Cfg()
    sub_terrains: dict[str, SubTerrainCfg] = {
        "flat": FlatPatchCfg(proportion=1.0),
        "stairs": PyramidStairsCfg(step_width=0.4, step_height=-0.08, proportion=1.0),
        "rough": RandomRoughCfg(min_height=-0.08, max_height=0.08, step=0.05, proportion=1.0),
        "slope": SlopeCfg(slope=-0.25, proportion=1.0),
        "wave": WaveCfg(num_waves=2.0, amplitude=0.08, proportion=1.0),
    }
    terrain = TerrainGeneratorCfg(
        num_rows=1,
        num_cols=5,
        subterrain_size=(6.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        pos=(-3.0, -15.0, 0.0),
        sub_terrains=sub_terrains,
        layout=(("flat", "stairs", "rough", "slope", "wave"),),
    )
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
            env_spacing=(6.0, 6.0),
            terrain=terrain,
            sensors=(
                RayCastSensorCfg(
                    name="pelvis_height",
                    link_name="pelvis",
                    pattern=GridPattern(
                        resolution=0.1,
                        size=(1.0, 1.0),
                        direction=(0.0, 0.0, -1.0),
                    ),
                    attach_yaw_only=True,
                    max_distance=5.0,
                ),
            ),
        ),
        decimation=4,
        # Long episode so the cell-to-cell tour is continuous (no periodic reset).
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
