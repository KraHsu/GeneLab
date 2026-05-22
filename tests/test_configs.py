"""Tests for ``genelab.configs`` domain-config helpers.

Currently covers ``SimulationCfg.play_retargeted_keys`` (ROADMAP §9 R3.2 /
ADR-0005) — the set of override paths the CLI rewrites ``env.`` →
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
