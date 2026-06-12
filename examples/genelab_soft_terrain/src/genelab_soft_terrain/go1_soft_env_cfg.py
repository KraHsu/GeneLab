"""Unitree Go1 standing on analytic deformable (soft) terrain — ADR-0001 stage-0 capstone.

The robot is held up purely by the analytic compliance force
(:class:`genelab.terrains.DeformableTerrainCfg`) injected at its feet over a *virtual*
surface; there is no rigid floor under the feet. The scene's default ground plane sits at
``z = 0``, well below the feet, acting only as a fall backstop.

Geometry: the Go1 "feet" are the ``*_calf`` links (the foot geom is a site at
``(0, 0, -0.213)`` on the calf). With the virtual surface at :data:`_SURFACE_HEIGHT` the
calves settle near it and the foot geoms sit ~0.2 m above ``z = 0`` — clear of the
backstop. The spawn is raised so the feet start just above the surface and drop onto it.

The normal force settles the feet at their sinkage equilibrium; the stage-1 ``mu`` traction
term gives the feet grip so they hold position instead of drifting. With zero actions this
is a *stand-and-hold* demo — actually walking on the soft terrain needs a trained policy.
"""

from genelab import mdp
from genelab.asset_zoo.unitree_go1 import UnitreeGo1Cfg
from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from genelab.mdp.actions.joint_position import JointPositionActionCfg
from genelab.terrains import DeformableTerrainCfg

# Go1 feet are the calf links (foot geom is a site at (0,0,-0.213) on the calf).
_GO1_FOOT_LINKS: tuple[str, ...] = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
_FOOT_SITE_DROP = 0.213
# Virtual soft-surface height (world z), referenced to the calf-link origin. This proven
# value keeps the home stance stable; the rendered (non-colliding) ground sits at the foot
# level (``_SURFACE_HEIGHT - _FOOT_SITE_DROP``) so the robot visibly stands on it, not floats.
_SURFACE_HEIGHT = 0.6
_GROUND_RENDER_HEIGHT = _SURFACE_HEIGHT - _FOOT_SITE_DROP
# Per-terrain compliance. Go1 ≈ 12 kg ⇒ ~118 N spread over 4 feet ⇒ a few-mm equilibrium
# sinkage at this stiffness; the damping settles the initial drop without bouncing.
_STIFFNESS = 8000.0
_DAMPING = 400.0
# Stage-1 traction: Coulomb friction at the feet (no granular shear cap for this firm demo
# surface). mu > 0 gives the feet grip — they hold position instead of drifting, the
# ingredient a policy needs to actually walk on soft terrain.
_FRICTION = 0.8


def go1_soft_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env config: Unitree Go1 standing on analytic deformable terrain (no rigid floor)."""
    robot = UnitreeGo1Cfg()
    # Spawn just above the virtual surface so the feet drop onto it and settle.
    bx, by, _ = robot.init_pos
    robot.init_pos = (bx, by, _SURFACE_HEIGHT + 0.15)

    return ManagerBasedRlEnvCfg(
        simulation=SimulationCfg(
            num_envs=4096 if not play else 1,
            dt=0.005,
            substeps=1,
            vis=play,
            gpu=True,
        ),
        scene=InteractiveSceneCfg(
            env_spacing=(2.0, 2.0),
            terrain=None,
            sensors=(),
            ground_plane_collision=False,
            ground_plane_height=_GROUND_RENDER_HEIGHT,
        ),
        decimation=4,
        episode_length_s=20.0,
        device="cuda",
        robot=robot,
        deformable_terrain=DeformableTerrainCfg(
            k=_STIFFNESS,
            c=_DAMPING,
            mu=_FRICTION,
            surface_height=_SURFACE_HEIGHT,
            foot_link_names=_GO1_FOOT_LINKS,
        ),
        actions_cfg={
            "joint_pos": JointPositionActionCfg(
                asset_name="robot",
                joint_names=(".*",),
                use_default_offset=True,
            )
        },
        observations_cfg={
            "policy": ObservationGroupCfg(
                enable_corruption=False,
                terms={
                    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                    # Privileged: the simulator's true per-foot sinkage (ADR-0001).
                    "terrain_sinkage": ObservationTermCfg(func=mdp.terrain_sinkage),
                },
            )
        },
        rewards_cfg={
            # Soft-terrain shaping (paper §8.3): keep feet near the surface and off deep
            # footprints. Small weights — these are demo terms, not a tuned gait reward set.
            "terrain_sinkage": RewardTermCfg(func=mdp.terrain_sinkage_l2, weight=-0.1),
            "footprint_revisit": RewardTermCfg(func=mdp.footprint_revisit, weight=-0.1),
        },
        terminations_cfg={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        },
        events_cfg={
            "reset_joints": EventTermCfg(
                mode="reset",
                func=mdp.reset_joints_by_offset,
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
    )
