"""Velocity-tracking env config for the Unitree Go1 on complex (rough) terrain.

The complex-terrain task is the flat velocity task with a heightfield ground, a
trunk-mounted ``height_scan`` observation, and a ``terrain_levels`` curriculum that
promotes / demotes envs by walked distance — applied through ``_velocity_env_cfg_base``.
"""

from genelab import mdp
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import CurriculumTermCfg, ObservationTermCfg
from genelab.mdp.curriculums import terrain_levels_vel
from genelab.mdp.noise import Unoise
from genelab.sensor import GridPattern, TerrainHeightSensorCfg
from genelab.terrains import (
    DiscreteObstaclesCfg,
    PyramidStairsCfg,
    RandomRoughCfg,
    SlopeCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
)

from genelab_unitree.go1.env_cfg import (
    _TRUNK_LINK,  # pyright: ignore[reportPrivateUsage]
    _velocity_env_cfg_base,  # pyright: ignore[reportPrivateUsage]
)

# Curriculum rows. Each row is a MIX of terrain types (stairs up/down, box clutter,
# random rough, slopes up/down) scaled by the row's difficulty d = L/(N-1). Difficulty
# ceilings are tuned down from the G1 humanoid set for the smaller Go1 quadruped (shorter
# legs / lower trunk): gentler stairs, lower clutter, milder slopes.
_NUM_LEVELS = 10


def _complex_terrain() -> TerrainGeneratorCfg:
    sub_terrains: dict[str, SubTerrainCfg] = {}
    for level in range(_NUM_LEVELS):
        d = level / (_NUM_LEVELS - 1)
        sub_terrains[f"stairsup_l{level}"] = PyramidStairsCfg(
            step_width=0.3, step_height=0.03 + 0.10 * d
        )
        sub_terrains[f"stairsdn_l{level}"] = PyramidStairsCfg(
            step_width=0.3, step_height=-(0.03 + 0.10 * d)
        )
        sub_terrains[f"box_l{level}"] = DiscreteObstaclesCfg(
            max_height=0.03 + 0.08 * d, min_size=0.4, max_size=1.0, num_rects=20
        )
        sub_terrains[f"rough_l{level}"] = RandomRoughCfg(
            min_height=-(0.02 + 0.05 * d),
            max_height=0.02 + 0.05 * d,
            step=0.02,
            downsampled_scale=0.2,
        )
        sub_terrains[f"slopeup_l{level}"] = SlopeCfg(slope=0.3 * d)
        sub_terrains[f"slopedn_l{level}"] = SlopeCfg(slope=-(0.3 * d))
    layout = tuple(
        (
            f"stairsup_l{level}",
            f"stairsup_l{level}",
            f"stairsdn_l{level}",
            f"stairsdn_l{level}",
            f"box_l{level}",
            f"box_l{level}",
            f"rough_l{level}",
            f"rough_l{level}",
            f"slopeup_l{level}",
            f"slopedn_l{level}",
        )
        for level in range(_NUM_LEVELS)
    )
    return TerrainGeneratorCfg(
        num_rows=_NUM_LEVELS,
        num_cols=10,
        subterrain_size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        pos=(-40.0, -20.0, 0.0),
        sub_terrains=sub_terrains,
        layout=layout,
        curriculum=True,
    )


def unitree_go1_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Complex-terrain velocity-tracking env config for the Unitree Go1."""
    # Trunk-mounted height scan: GridPattern(0.1, (1.6, 1.0)) -> 17x11 = 187 rays,
    # the body-frame grid the policy reads to see the terrain ahead.
    height_scan_sensor = TerrainHeightSensorCfg(
        name="height_scan",
        link_name=_TRUNK_LINK,
        pattern=GridPattern(resolution=0.1, size=(1.6, 1.0)),
    )
    extra_actor_obs = {
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scan"},
            noise=Unoise(-0.1, 0.1),
        ),
    }
    # The critic reads a clean (un-noised) height scan to keep the asymmetric setup.
    extra_critic_obs = {
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scan"},
        ),
    }
    extra_curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=terrain_levels_vel,
            # distance_threshold = subterrain_size[0] / 2: an env that traverses half a
            # sub-terrain cell gets promoted. Without it the terrain levels never move.
            params={"distance_threshold": 4.0, "command_name": "twist"},
        ),
    }
    return _velocity_env_cfg_base(
        play=play,
        terrain=_complex_terrain(),
        extra_sensors=(height_scan_sensor,),
        extra_actor_obs=extra_actor_obs,
        extra_critic_obs=extra_critic_obs,
        extra_curriculum=extra_curriculum,
    )
