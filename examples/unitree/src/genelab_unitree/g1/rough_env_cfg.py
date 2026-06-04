"""Velocity-tracking env config for the Unitree G1 on rough terrain.

The rough task is the flat velocity task (``env_cfg.unitree_g1_velocity_env_cfg``)
with four deltas, applied through ``_velocity_env_cfg_base``:

* the ground becomes a 10-level mixed heightfield (stairs / boxes / random rough /
  slopes, rows = difficulty), mirroring Isaac Lab's ``ROUGH_TERRAINS_CFG``,
* a body-frame ``height_scan`` observation lets the policy see the terrain ahead,
* a ``terrain_levels`` curriculum promotes / demotes envs by walked distance,
* the two terrain-sensitive foot penalties are lightened (rough-only) so they don't
  overwhelm velocity tracking as the curriculum climbs.

Events, domain randomisation and command ranges are shared with the flat task
verbatim — rough terrain supplies the added difficulty, so the command ranges are
not widened on top.
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

# Reuse the flat config's torso-link name and shared skeleton (same g1 package).
from genelab_unitree.g1.env_cfg import (
    _TORSO_LINK,  # pyright: ignore[reportPrivateUsage]
    _velocity_env_cfg_base,  # pyright: ignore[reportPrivateUsage]
)


def unitree_g1_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rough-terrain velocity-tracking env config for the Unitree G1."""
    # IsaacLab ROUGH_TERRAINS_CFG parity: 10 curriculum rows, each a MIX of terrain types
    # (stairs up/down, box clutter, random rough, slopes up/down) scaled by the row's
    # difficulty d = L/9. Structured terrain (stairs/slopes) stays learnable at high
    # difficulty so velocity tracking keeps a real reward as the curriculum climbs, unlike
    # 100% random rough where the high rows are unmasterable. ten independent columns per
    # row; subterrain_size=(8.0, 8.0) gives each env room to walk the promotion threshold.
    _NUM_LEVELS = 10
    sub_terrains: dict[str, SubTerrainCfg] = {}
    for _level in range(_NUM_LEVELS):
        _d = _level / (_NUM_LEVELS - 1)
        sub_terrains[f"stairsup_l{_level}"] = PyramidStairsCfg(
            step_width=0.3, step_height=0.05 + 0.18 * _d
        )
        sub_terrains[f"stairsdn_l{_level}"] = PyramidStairsCfg(
            step_width=0.3, step_height=-(0.05 + 0.18 * _d)
        )
        sub_terrains[f"box_l{_level}"] = DiscreteObstaclesCfg(
            max_height=0.05 + 0.15 * _d, min_size=0.4, max_size=1.0, num_rects=20
        )
        sub_terrains[f"rough_l{_level}"] = RandomRoughCfg(
            min_height=-(0.02 + 0.08 * _d),
            max_height=0.02 + 0.08 * _d,
            step=0.02,
            downsampled_scale=0.2,
        )
        sub_terrains[f"slopeup_l{_level}"] = SlopeCfg(slope=0.4 * _d)
        sub_terrains[f"slopedn_l{_level}"] = SlopeCfg(slope=-(0.4 * _d))
    layout = tuple(
        (
            f"stairsup_l{_level}",
            f"stairsup_l{_level}",
            f"stairsdn_l{_level}",
            f"stairsdn_l{_level}",
            f"box_l{_level}",
            f"box_l{_level}",
            f"rough_l{_level}",
            f"rough_l{_level}",
            f"slopeup_l{_level}",
            f"slopedn_l{_level}",
        )
        for _level in range(_NUM_LEVELS)
    )
    terrain = TerrainGeneratorCfg(
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

    # Torso-mounted height scan: GridPattern(0.1, (1.6, 1.0)) -> 17x11 = 187 rays,
    # the Isaac Lab default for G1 rough. Single-frame mode (link_name) returns the
    # full per-ray grid as a (B, 187) tensor — what mdp.height_scan expects.
    height_scan_sensor = TerrainHeightSensorCfg(
        name="height_scan",
        link_name=_TORSO_LINK,
        pattern=GridPattern(resolution=0.1, size=(1.6, 1.0)),
    )

    extra_actor_obs = {
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scan"},
            noise=Unoise(-0.1, 0.1),
        ),
    }
    # The critic already reads privileged clean signals (foot_height); give it a
    # clean (un-noised) height scan too so the asymmetric actor/critic setup holds.
    extra_critic_obs = {
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan,
            params={"sensor_name": "height_scan"},
        ),
    }

    extra_curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=terrain_levels_vel,
            # distance_threshold = subterrain_size[0] / 2 (Isaac Lab parity): an env that
            # traverses half a sub-terrain cell gets promoted. Without this term the terrain
            # levels never move and the curriculum is dead.
            params={"distance_threshold": 4.0, "command_name": "twist"},
        ),
    }

    cfg = _velocity_env_cfg_base(
        play=play,
        terrain=terrain,
        extra_sensors=(height_scan_sensor,),
        extra_actor_obs=extra_actor_obs,
        extra_critic_obs=extra_critic_obs,
        extra_curriculum=extra_curriculum,
    )
    # Rough-only reward retune: the shared (flat) weights make ``foot_clearance`` /
    # ``foot_swing_height`` terrain-sensitive penalties as large as the velocity-tracking
    # reward. As the curriculum roughens the ground their error grows and overwhelms
    # tracking, so the policy gives up (action std inflates) and de-learns. Lightening
    # them keeps tracking dominant as the level climbs; the flat task keeps the heavier
    # weights (it has no curriculum, so the penalties never run away).
    cfg.rewards_cfg["foot_clearance"].weight = -0.5
    cfg.rewards_cfg["foot_swing_height"].weight = -0.25
    return cfg
