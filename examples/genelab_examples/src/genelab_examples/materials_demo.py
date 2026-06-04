"""Reference scene cfgs demonstrating ``genelab.materials``.

These factories are not registered tasks — they are minimal, importable examples of
attaching physics materials and rendering surfaces to scene objects, and of tuning
the solver a deformable body needs. They are exercised by
``tests/test_examples_materials.py``.
"""

from genelab.configs import InteractiveSceneCfg
from genelab.entity import RigidObjectCfg
from genelab.materials import (
    MetalCfg,
    MpmElasticCfg,
    MpmOptionsCfg,
    SolverOptionsCfg,
)

# Box geometry shared by the demo scenes.
_BOX_SIZE = (0.1, 0.1, 0.1)


def deformable_scene_cfg() -> InteractiveSceneCfg:
    """An MPM-elastic box that drops onto the ground plane.

    Carrying an ``MpmElasticCfg`` makes ``InteractiveScene`` enable the MPM solver;
    ``solvers.mpm`` widens the simulation domain so the box stays inside it.
    """
    return InteractiveSceneCfg(
        entities={
            "blob": RigidObjectCfg(
                morph="box",
                size=_BOX_SIZE,
                init_pos=(0.0, 0.0, 0.3),
                fixed=False,
                material=MpmElasticCfg(E=3e5, nu=0.3),
            )
        },
        solvers=SolverOptionsCfg(
            mpm=MpmOptionsCfg(lower_bound=(-1.0, -1.0, 0.0), upper_bound=(1.0, 1.0, 1.0))
        ),
    )


def surfaced_scene_cfg() -> InteractiveSceneCfg:
    """A rigid box re-surfaced as gold — surface is rendering-only, physics rigid."""
    return InteractiveSceneCfg(
        entities={
            "cube": RigidObjectCfg(
                morph="box",
                size=_BOX_SIZE,
                init_pos=(0.0, 0.0, 0.05),
                fixed=False,
                friction=1.0,
                surface=MetalCfg(metal_type="gold"),
            )
        }
    )
