"""Tests for ``genelab.configs`` domain-config helpers.

Currently covers ``SimulationCfg.play_retargeted_keys`` — the set of
override paths the CLI rewrites ``env.`` →
``play_env.`` in play mode. The list moved off a private constant in
``cli/__init__.py`` onto the domain config so it lives next to the
``SimulationCfg`` fields the shortcut flags target.
"""

from __future__ import annotations

from genelab.configs import SimulationCfg


def test_play_retargeted_keys_exact_set() -> None:
    """The four ``env.simulation.*`` shortcut-override paths, verbatim and ordered."""
    assert SimulationCfg.play_retargeted_keys() == (
        "env.simulation.vis",
        "env.simulation.gpu",
        "env.simulation.steps",
        "env.simulation.dt",
    )


def test_base_env_cfg_has_device_field() -> None:
    """``device`` lives on the base cfg so non-RL envs (Rubiks/Wuji play scenes)
    constructed via ``ManagerBasedRlEnv`` can read ``cfg.device``."""
    from genelab.configs import ManagerBasedEnvCfg

    assert "device" in ManagerBasedEnvCfg.__dataclass_fields__
    assert ManagerBasedEnvCfg().device == "cuda"


def test_play_retargeted_keys_target_real_simulation_fields() -> None:
    """Every retargeted key names an actual ``SimulationCfg`` field (no stale paths)."""
    fields = SimulationCfg.__dataclass_fields__
    for key in SimulationCfg.play_retargeted_keys():
        assert key.startswith("env.simulation."), key
        field_name = key.rsplit(".", 1)[1]
        assert field_name in fields, f"{field_name!r} is not a SimulationCfg field"


def test_play_retargeted_keys_callable_on_class_and_instance() -> None:
    """Static method — same result whether called on the class or an instance."""
    assert SimulationCfg.play_retargeted_keys() == SimulationCfg().play_retargeted_keys()


def test_rigid_options_kwargs_empty_by_default() -> None:
    """An untouched SimulationCfg yields no rigid_options kwargs (Genesis defaults preserved)."""
    assert SimulationCfg().rigid_options_kwargs() == {}


def test_rigid_options_kwargs_maps_set_fields_to_genesis_names() -> None:
    cfg = SimulationCfg(
        enable_self_collision=False,
        enable_joint_limit=True,
        max_collision_pairs=200,
        solver_iterations=80,
        ls_iterations=20,
        solver_tolerance=1e-6,
        constraint_timeconst=0.02,
        integrator="implicitfast",
    )
    assert cfg.rigid_options_kwargs() == {
        "enable_self_collision": False,
        "enable_joint_limit": True,
        "max_collision_pairs": 200,
        "iterations": 80,  # solver_iterations → RigidOptions.iterations
        "ls_iterations": 20,
        "tolerance": 1e-6,  # solver_tolerance → RigidOptions.tolerance
        "constraint_timeconst": 0.02,
        "integrator": "implicitfast",  # stays a string; the scene resolves to gs.integrator
    }


def test_rigid_options_kwargs_skips_none_only() -> None:
    # enable_self_collision=False must survive (it's a real value, not "unset").
    cfg = SimulationCfg(enable_self_collision=False, solver_iterations=50)
    assert cfg.rigid_options_kwargs() == {"enable_self_collision": False, "iterations": 50}
