"""Sand: train on the analytic deformable-terrain model, validate on granular MPM.

The paper's sim-to-sim recipe — train the policy on the cheap analytic surrogate
(``go1_sand_velocity_env_cfg``), then test it on real granular media simulated with
Genesis MPM (``go1_sand_mpm_env_cfg``), where the feet make true contact with sand
particles (no analytic injection).

Analytic *sand* parameters vs the firm-elastic demo: softer (more sinkage), lower
friction, a granular shear-strength cap ``eta`` (traction saturates instead of rising with
load), and plastic residual (footprints that persist) — the §6.1 "dry sand" regime.
"""

from genelab import mdp
from genelab.entity import RigidObjectCfg
from genelab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from genelab.managers import EventTermCfg
from genelab.materials import MpmOptionsCfg, MpmSandCfg, SolverOptionsCfg
from genelab.terrains import DeformableTerrainCfg
from genelab_soft_terrain.go1_soft_velocity_env_cfg import (
    FOOT_OFFSET,
    GO1_FOOT_LINKS,
    go1_soft_velocity_env_cfg,
)

# Granular MPM sand bed: top at z=0.15, resting on the ground plane at z=0.
_BED_TOP = 0.15
_BED_THICK = 0.15
_BED_HALF = 0.9  # bed is 1.8 m square

# Analytic "deep loose sand over hardpan", calibrated against the MPM bed (sim-to-sim):
# a point foot exceeds loose sand's bearing capacity and sinks through the 15 cm layer
# to the rigid floor (observed on MPM; Terzaghi agrees), so the layer is modeled as a
# *soft drag/anchor medium* over a real colliding ground plane, not as the support.
# k: at full 15 cm depth a foot carries ~15 N (~half the weight through sand, rest on
# the floor) — feet reach hardpan as on MPM.
_SAND = DeformableTerrainCfg(
    k=100.0,  # bearing far below foot pressure -> wading regime
    c=20.0,
    mu=0.7,
    lateral_stiffness=2500.0,  # buried shin anchors strongly sideways
    eta=20.0,  # granular shear-strength cap: traction saturates, doesn't rise with load
    # Plowing drag: makes paddling through sand cost real force so the policy must step
    # over it. Terzaghi scale for this bed (~26 N at 13 cm). Training ramps it in from 0
    # via :func:`drag_curriculum` (any significant drag at bootstrap knocks the robot
    # over before it discovers stepping); play/eval use this full value from step one.
    drag_coeff=1500.0,
    # Overburden extraction: lifting a buried foot costs K_e*d^2 (~8.5 N at 13 cm), so
    # high-stepping is priced the way the MPM bed prices it instead of being free.
    extraction_coeff=500.0,
    yield_depth=0.01,
    plastic_rate=0.5,  # footprints accumulate past yield
    recovery_time=1.0e9,  # sand keeps its footprints (no recovery)
    surface_height=_BED_TOP,  # sand surface matches the MPM bed top
    foot_link_names=GO1_FOOT_LINKS,
    foot_offset=FOOT_OFFSET,
)


def drag_curriculum(
    env,  # type: ignore[no-untyped-def]
    env_ids: object,
    targets: dict[str, float] | None = None,
    healthy_buf_len: float = 300.0,
    step_frac: float = 1.0 / 300.0,
) -> None:
    """Performance-keyed ratchet: raise resistance terms only while the gait is healthy.

    Any significant medium resistance before the gait takes off stalls PPO completely
    (observed at static drag 1500, static 500, and step-scheduled ramps): balance
    recovery in deep sand needs fast foot repositioning, which the resistance fights.
    Takeoff *time* varies run to run, so a step-count schedule gambles on hitting the
    window — instead, each firing advances all ``targets`` (cfg field -> full value) by
    ``step_frac`` of their target, only when the mean in-flight episode length clears
    ``healthy_buf_len`` (~half the achieved episode length), and never decreases them.
    Stalls pause the ramp instead of compounding.

    Training must start the ramped fields at ``0`` (see the train cfg); play/eval use
    the full values from step one.
    """
    del env_ids
    driver = env.deformable_terrain
    if driver is None or not targets:
        return
    if float(env.episode_length_buf.float().mean()) >= healthy_buf_len:
        terrain = driver.terrain
        for field, target in targets.items():
            params = getattr(terrain, field)  # per-env medium tensor, the live value
            params.add_(step_frac * target).clamp_max_(target)


def go1_sand_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity tracking on the analytic *sand* model (training env).

    Geometry-matched to :func:`go1_sand_mpm_env_cfg`: rigid floor at z=0 (the hardpan
    under the bed), sand surface at ``_BED_TOP`` — the robot wades, feet on the floor.
    """
    cfg = go1_soft_velocity_env_cfg(play=play)
    cfg.deformable_terrain = _SAND
    cfg.scene.ground_plane_collision = True  # hardpan: the floor under the sand layer
    cfg.scene.ground_plane_height = 0.0
    bx, by, _ = cfg.robot.init_pos
    cfg.robot.init_pos = (bx, by, _BED_TOP + 0.40)  # same spawn drop as the MPM env
    if not play:
        # Training only: domain-randomize the medium per env on every reset, so the
        # policy trains against a *distribution* of granular media (absorbs the
        # surrogate-vs-MPM residual). Ranges bracket the calibrated _SAND point —
        # firmer/looser support, slicker/grippier, lighter/heavier plowing cost.
        # Play/eval keep the calibrated point values from step one.
        cfg.events_cfg["randomize_medium"] = EventTermCfg(
            mode="reset",
            func=mdp.randomize_terrain_params,
            params={
                "ranges": {
                    "k": (50.0, 1500.0),
                    "mu": (0.4, 1.0),
                    "drag_coeff": (300.0, 3000.0),
                    "extraction_coeff": (100.0, 1500.0),
                }
            },
        )
    return cfg


def go1_sand_mpm_env_cfg(play: bool = True) -> ManagerBasedRlEnvCfg:
    """Validation env: the Go1 on a **real granular MPM sand bed** (no analytic terrain).

    Same robot, observations, and command as the analytic-sand training env, but the feet
    make true contact with Genesis MPM sand particles — the sim-to-sim test of whether a
    policy trained on the analytic surrogate transfers to granular media. Play/eval only
    (MPM is too slow to train on).
    """
    cfg = go1_soft_velocity_env_cfg(play=play)
    # No analytic injection — the MPM bed IS the terrain; feet collide with it normally.
    cfg.deformable_terrain = None
    cfg.observations_cfg["critic"].terms.pop("terrain_sinkage", None)  # no analytic state
    # The terrain-derived reward terms read the analytic terrain state, which doesn't
    # exist here. Play/eval only — the reward numbers aren't used anyway.
    for name in ("feet_air_time", "feet_slip", "terrain_sinkage"):
        cfg.rewards_cfg.pop(name, None)
    cfg.scene.ground_plane_collision = True  # real floor under the sand bed
    cfg.scene.ground_plane_height = 0.0
    # MPM stability: the solver inherits the sim dt, and 0.005 s is ~5x over the CFL
    # limit for sand (E=1e6, dx~1.6 cm) — the particles turn to mush and the robot
    # sinks straight through. Substep to 5e-4 s; the policy still runs at 50 Hz.
    cfg.simulation.substeps = 10
    cfg.scene.entities = {
        "sand_bed": RigidObjectCfg(
            morph="box",
            size=(2 * _BED_HALF, 2 * _BED_HALF, _BED_THICK),
            init_pos=(0.0, 0.0, _BED_TOP - _BED_THICK / 2),
            fixed=False,
            material=MpmSandCfg(rho=1600.0),  # dry sand; Genesis default 1000 is too light
        )
    }
    cfg.scene.solvers = SolverOptionsCfg(
        mpm=MpmOptionsCfg(
            # MPM clamps the boundary ~3 cells inside the domain (safety padding); the bed's
            # particles start at z=0, so the domain floor must sit well below that.
            lower_bound=(-_BED_HALF - 0.3, -_BED_HALF - 0.3, -0.15),
            upper_bound=(_BED_HALF + 0.3, _BED_HALF + 0.3, 1.0),
            # Genesis default (1.56 cm cells, 1 cm particles): the Go1 foot (~2 cm) needs
            # foot-scale cells or the contact is under-resolved and the feet punch through.
            grid_density=64,
        )
    )
    # Spawn just above the sand surface so the feet drop onto it.
    bx, by, _ = cfg.robot.init_pos
    cfg.robot.init_pos = (bx, by, _BED_TOP + 0.40)
    return cfg
