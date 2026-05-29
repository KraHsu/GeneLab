"""Velocity-tracking env config for the Unitree G1 on rough terrain.

The rough task is the flat velocity task (``env_cfg.unitree_g1_velocity_env_cfg``)
with three deltas, applied through ``_velocity_env_cfg_base``:

* the ground becomes a 5-level RandomRough heightfield (rows = difficulty),
* a body-frame ``height_scan`` observation lets the policy see the terrain ahead,
* a ``terrain_levels`` curriculum promotes / demotes envs by walked distance.

Everything else (rewards, events, domain randomisation, command ranges) is shared
with the flat task verbatim — rough terrain supplies the added difficulty, so the
command ranges are not widened on top.
"""

from genelab import mdp
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import CurriculumTermCfg, ObservationTermCfg
from genelab.mdp.curriculums import terrain_levels_vel
from genelab.mdp.noise import Unoise
from genelab.sensor import GridPattern, TerrainHeightSensorCfg
from genelab.terrains import RandomRoughCfg, SubTerrainCfg, TerrainGeneratorCfg

# Reuse the flat config's torso-link name and shared skeleton (same g1 package).
from genelab_unitree.g1.env_cfg import (
    _TORSO_LINK,  # pyright: ignore[reportPrivateUsage]
    _velocity_env_cfg_base,  # pyright: ignore[reportPrivateUsage]
)


def unitree_g1_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Rough-terrain velocity-tracking env config for the Unitree G1."""
    # Five rows of increasing roughness (the curriculum levels); ten independent
    # columns per row. subterrain_size=(8.0, 8.0) gives each env room to walk the
    # full 8 m promotion threshold within one sub-terrain cell.
    sub_terrains: dict[str, SubTerrainCfg] = {
        f"rough_l{level}": RandomRoughCfg(
            min_height=-0.02 - 0.025 * level,
            max_height=0.02 + 0.025 * level,
            step=0.02,
            proportion=1.0,
        )
        for level in range(5)
    }
    layout = tuple(tuple(f"rough_l{level}" for _ in range(10)) for level in range(5))
    terrain = TerrainGeneratorCfg(
        num_rows=5,
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
            # distance_threshold = subterrain_size[0]: an env that traverses one
            # sub-terrain cell gets promoted. Without this term the terrain levels
            # never move and the curriculum is dead.
            params={"distance_threshold": 8.0},
        ),
    }

    return _velocity_env_cfg_base(
        play=play,
        terrain=terrain,
        extra_sensors=(height_scan_sensor,),
        extra_actor_obs=extra_actor_obs,
        extra_critic_obs=extra_critic_obs,
        extra_curriculum=extra_curriculum,
    )
