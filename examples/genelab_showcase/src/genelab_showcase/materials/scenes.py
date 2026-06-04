"""Material demo scenes, built straight on ``InteractiveScene``.

Each factory returns a :class:`MaterialsDemoCfg` (a ``SimulationCfg`` + an
``InteractiveSceneCfg``); the runner builds and steps it. One material property per
scene is exposed as a viewer slider (jelly stiffness / floating-object density); the
getter/setter pairs below let the runner read the current value and re-apply a new one
on rebuild.

Parameters were tuned headlessly: the FEM jelly squashes ~24 % at rest with the
implicit solver, and the light box settles on the liquid surface while the dense ball
sinks to the tank floor.
"""

from dataclasses import dataclass, field
from typing import Any

from genelab.configs import InteractiveSceneCfg, SimulationCfg
from genelab.entity import RigidObjectCfg
from genelab.materials import (
    FemElasticCfg,
    FemOptionsCfg,
    MetalCfg,
    MpmLiquidCfg,
    MpmOptionsCfg,
    SolverOptionsCfg,
)


@dataclass
class MaterialsDemoCfg:
    """A material demo: simulation settings + scene composition."""

    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    scene: InteractiveSceneCfg = field(default_factory=InteractiveSceneCfg)


# --------------------------------------------------------------------------- soft body

SOFT_STIFFNESS_RANGE = (1.5e4, 2.5e5)  # Young's modulus E; lower = softer


def soft_body_demo_cfg() -> MaterialsDemoCfg:
    """A single FEM-elastic jelly ball dropped onto the ground.

    The viewer slider drives the ball's stiffness ``E``: softer values squash and wobble
    more on impact. FEM renders a continuous deforming surface (not a particle cloud) and
    runs in implicit mode — explicit FEM explodes at these stiffnesses.
    """
    return MaterialsDemoCfg(
        simulation=SimulationCfg(num_envs=1, dt=0.005, substeps=2, steps=500, vis=False, gpu=True),
        scene=InteractiveSceneCfg(
            entities={
                "jelly": RigidObjectCfg(
                    morph="sphere",
                    size=(0.14,),
                    init_pos=(0.0, 0.0, 0.55),
                    fixed=False,
                    material=FemElasticCfg(E=3e4, nu=0.4, rho=1000.0),
                ),
            },
            solvers=SolverOptionsCfg(fem=FemOptionsCfg(use_implicit_solver=True, damping=2.0)),
        ),
    )


def get_soft_stiffness(cfg: MaterialsDemoCfg) -> float:
    material = cfg.scene.entities["jelly"].material
    return float(getattr(material, "E", 3e4) or 3e4)


def set_soft_stiffness(cfg: MaterialsDemoCfg, value: float) -> None:
    cast_material: Any = cfg.scene.entities["jelly"].material
    cast_material.E = float(value)


# ------------------------------------------------------------------------------- fluid

FLOAT_DENSITY_RANGE = (50.0, 1200.0)  # kg/m^3; below ~1000 (water) the box floats


def fluid_demo_cfg() -> MaterialsDemoCfg:
    """An MPM-liquid pool with a light box that floats and a heavy ball that sinks.

    The MPM domain bounds double as the tank walls, so the liquid settles into a pool.
    The viewer slider drives the floating box's density: below the water density (1000)
    it bobs on the surface; above it the box sinks too. The gold ball is fixed-heavy for
    a sinking reference. The boundary floor sits just below ``z=0`` so the pool's
    underside stays inside the padded domain.
    """
    return MaterialsDemoCfg(
        simulation=SimulationCfg(num_envs=1, dt=0.005, substeps=16, steps=600, vis=False, gpu=True),
        scene=InteractiveSceneCfg(
            entities={
                "pool": RigidObjectCfg(
                    morph="box",
                    size=(0.36, 0.36, 0.18),
                    init_pos=(0.0, 0.0, 0.10),
                    fixed=False,
                    material=MpmLiquidCfg(rho=1000.0),
                ),
                "float_box": RigidObjectCfg(
                    morph="box",
                    size=(0.10, 0.10, 0.10),
                    init_pos=(-0.10, 0.0, 0.50),
                    fixed=False,
                    density=150.0,
                    friction=0.5,
                ),
                "sink_ball": RigidObjectCfg(
                    morph="sphere",
                    size=(0.045,),
                    init_pos=(0.12, 0.0, 0.50),
                    fixed=False,
                    density=3000.0,
                    friction=0.5,
                    surface=MetalCfg(metal_type="gold"),
                ),
            },
            solvers=SolverOptionsCfg(
                mpm=MpmOptionsCfg(lower_bound=(-0.30, -0.30, -0.05), upper_bound=(0.30, 0.30, 0.8))
            ),
        ),
    )


def get_float_density(cfg: MaterialsDemoCfg) -> float:
    return float(cfg.scene.entities["float_box"].density or 150.0)


def set_float_density(cfg: MaterialsDemoCfg, value: float) -> None:
    cfg.scene.entities["float_box"].density = float(value)
