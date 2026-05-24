"""Curriculum showcase env: 16 envs on a 5×5 RandomRough grid + terrain_levels_vel.

Each row of the grid is a difficulty level (level 0 = lowest roughness, level 4 =
highest). The curriculum term :func:`genelab.mdp.curriculums.terrain_levels_vel`
promotes envs that walk past ``distance_threshold`` and demotes envs that don't —
the runner scripts a tiny XY drift on the root so promotion fires consistently.
"""

from genelab import mdp
from genelab.asset_zoo import UnitreeG1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.mdp.curriculums import terrain_levels_vel
from genelab.terrains import RandomRoughCfg, TerrainGeneratorCfg


def curriculum_showcase_env_cfg() -> ManagerBasedRlEnvCfg:
    """16-env G1 stack on a 5×5 RandomRough grid driven by ``terrain_levels_vel``."""

    robot_cfg = UnitreeG1Cfg()
    # Five rows of increasing roughness; columns are independent samples.
    sub_terrains = {
        f"rough_l{level}": RandomRoughCfg(
            min_height=-0.02 - 0.025 * level,
            max_height=0.02 + 0.025 * level,
            step=0.02,
            proportion=1.0,
        )
        for level in range(5)
    }
    layout = tuple(tuple(f"rough_l{level}" for _ in range(5)) for level in range(5))
    terrain = TerrainGeneratorCfg(
        num_rows=5,
        num_cols=5,
        subterrain_size=(6.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        pos=(-15.0, -15.0, 0.0),
        sub_terrains=sub_terrains,
        layout=layout,
        curriculum=True,
    )
    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=16,
            # The known-good G1 tasks (Velocity-Flat / Tracking-Flat) run this robot
            # at dt=0.002 because the Genesis rigid solver is unstable for the G1 at
            # dt=0.005 (constraint forces diverge to NaN even from a clean stance).
            # Match them: dt=0.002 + decimation=10 keeps the control rate at 50 Hz, so
            # the curriculum/teleport schedule and the 2 s episode (100 control steps)
            # are unchanged.
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
        episode_length_s=2.0,
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
        curriculum_cfg={
            "terrain_levels": CurriculumTermCfg(
                func=terrain_levels_vel,
                # Tuned so that a falling/rolling G1 does NOT auto-promote: a
                # natural drop on RandomRough drifts the root by <1 m in 2 s, well
                # below 5 m. The runner explicitly teleports half the envs ~6 m
                # forward of spawn to trip the promote branch while leaving the
                # other half to demote.
                params={"distance_threshold": 5.0, "demote_ratio": 0.4},
            ),
        },
    )
