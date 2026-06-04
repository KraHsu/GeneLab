"""Unit tests for the ``genelab.materials`` config wrappers and scene plumbing.

These exercise the testable seams without importing Genesis: the ``_kwargs`` field
mapping, ``required_solvers``, ``apply_overrides`` on material dotted paths, the
``RigidObjectCfg`` legacy-shortcut compatibility, and the scene's solver auto-wiring
(driven against a fake ``gs``).
"""

from typing import Any

import pytest

from genelab.configs import InteractiveSceneCfg, apply_overrides
from genelab.entity.rigid_object import RigidObject, RigidObjectCfg
from genelab.materials import (
    HybridMaterialCfg,
    MpmElasticCfg,
    RigidMaterialCfg,
    SphLiquidCfg,
)
from genelab.materials.options import MpmOptionsCfg, SolverOptionsCfg, SphOptionsCfg


# --------------------------------------------------------------------------- fake gs


class _Recorder:
    """Stands in for a constructed Genesis material / options object."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs


class _Node:
    """A namespace node that is both attribute-traversable and callable.

    ``gs.materials.MPM.Elastic(**kw)`` records ``_Recorder("MPM.Elastic", **kw)``;
    ``gs.options.MPMOptions(**kw)`` records ``_Recorder("MPMOptions", **kw)``.
    """

    def __init__(self, path: str = "") -> None:
        self._path = path

    def __getattr__(self, name: str) -> "_Node":
        return _Node(f"{self._path}.{name}" if self._path else name)

    def __call__(self, **kwargs: Any) -> _Recorder:
        return _Recorder(self._path, **kwargs)


class _FakeGs:
    def __init__(self) -> None:
        self.materials = _Node()
        self.options = _Node()


# --------------------------------------------------------------------------- _kwargs


def test_kwargs_drops_none_and_mirrors_genesis_names() -> None:
    assert MpmElasticCfg(E=3e5, nu=0.3)._kwargs() == {"E": 3e5, "nu": 0.3}
    # ``density`` is the GeneLab shortcut; the material field is the Genesis name ``rho``.
    assert RigidMaterialCfg(friction=1.5, rho=500.0)._kwargs() == {"friction": 1.5, "rho": 500.0}
    assert SphLiquidCfg()._kwargs() == {}


def test_required_solvers_per_family() -> None:
    assert MpmElasticCfg().required_solvers() == {"mpm"}
    assert SphLiquidCfg().required_solvers() == {"sph"}
    assert RigidMaterialCfg().required_solvers() == {"rigid"}


def test_hybrid_required_solvers_is_union() -> None:
    hybrid = HybridMaterialCfg(
        material_rigid=RigidMaterialCfg(friction=2.0),
        material_soft=MpmElasticCfg(E=1e5),
    )
    assert hybrid.required_solvers() == {"rigid", "mpm"}


# ----------------------------------------------------------------------- build (fake)


def test_material_build_forwards_set_fields() -> None:
    built = RigidMaterialCfg(friction=1.5, rho=500.0).build(_FakeGs())
    assert built.name == "Rigid"
    assert built.kwargs == {"friction": 1.5, "rho": 500.0}


# ----------------------------------------------------------------------- overrides


def test_apply_overrides_resolves_material_paths() -> None:
    cfg = InteractiveSceneCfg(entities={"cube": RigidObjectCfg(material=MpmElasticCfg())})
    apply_overrides(
        cfg,
        {
            "entities.cube.material.E": "300000",
            "entities.cube.material.sampler": "random",
        },
    )
    material = cfg.entities["cube"].material
    assert isinstance(material, MpmElasticCfg)
    assert material.E == 300000.0
    assert isinstance(material.E, float)
    assert material.sampler == "random"


def test_apply_overrides_coerces_solver_option_tuple() -> None:
    cfg = InteractiveSceneCfg(solvers=SolverOptionsCfg(mpm=MpmOptionsCfg()))
    apply_overrides(cfg, {"solvers.mpm.lower_bound": "-1,-1,0"})
    assert cfg.solvers.mpm is not None
    assert cfg.solvers.mpm.lower_bound == (-1.0, -1.0, 0.0)


# ------------------------------------------------------------- RigidObject shortcut


def test_legacy_shortcut_returns_none_when_unset() -> None:
    obj = RigidObject(RigidObjectCfg(), name="x")
    assert obj._material_or_default(_FakeGs()) is None


def test_legacy_shortcut_builds_rigid_from_friction_density() -> None:
    obj = RigidObject(RigidObjectCfg(friction=1.5, density=500.0), name="x")
    built = obj._material_or_default(_FakeGs())
    assert built.name == "Rigid"
    assert built.kwargs == {"friction": 1.5, "rho": 500.0}


def test_explicit_material_takes_precedence_over_shortcut() -> None:
    obj = RigidObject(RigidObjectCfg(friction=1.5, material=MpmElasticCfg(E=3e5)), name="x")
    built = obj._material_or_default(_FakeGs())
    assert built.name == "MPM.Elastic"
    assert built.kwargs == {"E": 3e5}


# ------------------------------------------------------------- scene solver wiring

pytest.importorskip("torch")

from genelab.configs import SimulationCfg  # noqa: E402
from genelab.scene import InteractiveScene  # noqa: E402


def _scene(scene_cfg: InteractiveSceneCfg) -> InteractiveScene:
    return InteractiveScene(SimulationCfg(), scene_cfg, device_hint="cpu")


def test_rigid_only_scene_adds_no_solver_options() -> None:
    scene = _scene(InteractiveSceneCfg(entities={"cube": RigidObjectCfg(friction=1.0)}))
    scene_kwargs: dict[str, Any] = {}
    scene._add_solver_options(_FakeGs(), scene_kwargs)
    assert scene_kwargs == {}


def test_mpm_material_enables_mpm_options_with_defaults() -> None:
    scene = _scene(
        InteractiveSceneCfg(
            entities={"blob": RigidObjectCfg(morph="box", material=MpmElasticCfg())}
        )
    )
    scene_kwargs: dict[str, Any] = {}
    scene._add_solver_options(_FakeGs(), scene_kwargs)
    assert "mpm_options" in scene_kwargs
    assert scene_kwargs["mpm_options"].name == "MPMOptions"
    assert scene_kwargs["mpm_options"].kwargs == {}  # Genesis defaults


def test_user_solver_cfg_overrides_defaults_and_enables_without_material() -> None:
    scene = _scene(
        InteractiveSceneCfg(
            entities={"cube": RigidObjectCfg(friction=1.0)},
            solvers=SolverOptionsCfg(sph=SphOptionsCfg(particle_size=0.01)),
        )
    )
    scene_kwargs: dict[str, Any] = {}
    scene._add_solver_options(_FakeGs(), scene_kwargs)
    assert "sph_options" in scene_kwargs
    assert scene_kwargs["sph_options"].kwargs == {"particle_size": 0.01}
