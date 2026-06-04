"""Curriculum showcase env: 25 spring-held G1s on a 5×5 difficulty grid.

Each row of the grid is a difficulty level, rendered as a distinct terrain type that escalates
flat → wave → rough → stairs → slope (level 0 = flat, level 4 = steep slope).
The 25 envs fill the 5×5 grid (one robot per cell), held upright by a virtual spring. The
runner drives a controlled terrain-level curriculum (promote / demote) and shows the live
level distribution as a histogram, colouring each robot's head marker by its level — so the
curriculum's redistribution is visible without a trained walking policy.
"""

from genelab import mdp
from genelab.asset_zoo import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg, TerminationTermCfg
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.terrains import (
    FlatPatchCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SlopeCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
    WaveCfg,
)


def curriculum_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """25-env G1 grid on a 5×5 difficulty grid (one robot per cell, one terrain type per row)."""

    robot_cfg = UnitreeG1Cfg()
    # Each difficulty LEVEL is a *distinct* terrain type, escalating flat → wave → rough →
    # stairs → slope. They must be distinct genesis types: Genesis keys terrain parameters by
    # type, so five RandomRoughCfg with different amplitudes collapse to a single shape and the
    # rows render identically. Distinct types give an unmistakable visual gradient.
    sub_terrains: dict[str, SubTerrainCfg] = {
        "l0_flat": FlatPatchCfg(proportion=1.0),
        "l1_wave": WaveCfg(num_waves=3.0, amplitude=0.12, proportion=1.0),
        "l2_rough": RandomRoughCfg(
            min_height=-0.14, max_height=0.14, step=0.05, downsampled_scale=1.0, proportion=1.0
        ),
        "l3_stairs": PyramidStairsCfg(step_width=0.5, step_height=-0.14, proportion=1.0),
        "l4_slope": SlopeCfg(slope=-0.5, proportion=1.0),
    }
    # Row r (= difficulty level r) is tiled entirely with that level's terrain type.
    level_keys = ("l0_flat", "l1_wave", "l2_rough", "l3_stairs", "l4_slope")
    layout = tuple((key,) * 5 for key in level_keys)
    terrain = TerrainGeneratorCfg(
        num_rows=5,
        num_cols=5,
        subterrain_size=(6.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        pos=(-15.0, -15.0, 0.0),
        sub_terrains=sub_terrains,
        layout=layout,
    )
    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=25,
            # The G1 rigid solver is unstable at dt=0.005 (constraint forces diverge to NaN);
            # the known-good G1 tasks run at dt=0.002 + decimation=10 (50 Hz control). Match.
            dt=0.002,
            substeps=1,
            steps=400,
            vis=False,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(6.0, 6.0),
            terrain=terrain,
        ),
        decimation=10,
        # Long episode so the controlled curriculum runs continuously (no auto-reset).
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
