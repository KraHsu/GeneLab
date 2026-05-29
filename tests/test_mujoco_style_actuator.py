"""Unit tests for :class:`genelab.actuator.MujocoStyleActuatorCfg`.

The cfg's only job is translating Mujoco's ``(gear, gain_prm, bias_prm)`` triple
into the underlying ``IdealPDActuatorCfg`` knobs and rejecting unsupported
``dyntype`` / ``gaintype`` / ``biastype`` combinations up front. Tests cover the
translation arithmetic and each rejection path.
"""

import pytest

torch = pytest.importorskip("torch")

from genelab.actuator import IdealPDActuator, MujocoStyleActuatorCfg  # noqa: E402


def _cfg(**overrides: object) -> MujocoStyleActuatorCfg:
    base: dict[str, object] = {
        "target_names_expr": (".*",),
        "gear": 1.0,
        "bias_prm": (0.0, -100.0, -2.0),
        "effort_limit": 80.0,
    }
    base.update(overrides)
    return MujocoStyleActuatorCfg(**base)  # type: ignore[arg-type]


def test_translation_to_stiffness_damping() -> None:
    """``stiffness = -gear * bias_prm[1]``, ``damping = -gear * bias_prm[2]``."""
    cfg = _cfg(gear=1.5, bias_prm=(0.0, -200.0, -3.0))
    # -1.5 * -200 = 300; -1.5 * -3 = 4.5
    assert cfg.stiffness == pytest.approx(300.0)
    assert cfg.damping == pytest.approx(4.5)
    assert cfg.effort_limit == pytest.approx(80.0)
    assert cfg.class_type is IdealPDActuator


def test_gear_one_is_identity_for_bias_prm() -> None:
    cfg = _cfg(gear=1.0, bias_prm=(0.0, -50.0, -1.0))
    assert cfg.stiffness == pytest.approx(50.0)
    assert cfg.damping == pytest.approx(1.0)


def test_non_zero_constant_bias_raises() -> None:
    """``bias_prm[0]`` (constant force offset) has no IdealPD analogue."""
    with pytest.raises(ValueError, match="constant bias"):
        _cfg(bias_prm=(0.5, -100.0, -2.0))


def test_unsupported_dyntype_raises() -> None:
    with pytest.raises(ValueError, match="dyntype"):
        _cfg(dyntype="integrator")  # type: ignore[arg-type]


def test_unsupported_gaintype_raises() -> None:
    with pytest.raises(ValueError, match="gaintype"):
        _cfg(gaintype="muscle")  # type: ignore[arg-type]


def test_unsupported_biastype_raises() -> None:
    with pytest.raises(ValueError, match="biastype"):
        _cfg(biastype="muscle")  # type: ignore[arg-type]


def test_explicit_stiffness_or_damping_raises() -> None:
    """The cfg computes these from ``gear / bias_prm`` — direct overrides are confusing."""
    with pytest.raises(ValueError, match="stiffness / damping"):
        _cfg(stiffness=10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stiffness / damping"):
        _cfg(damping=0.5)  # type: ignore[arg-type]


def test_preserves_passthrough_fields() -> None:
    """``effort_limit`` / ``velocity_limit`` / ``armature`` / ``friction`` pass through unchanged."""
    cfg = _cfg(effort_limit=50.0, velocity_limit=12.0, armature=0.05, friction=0.1)
    assert cfg.effort_limit == pytest.approx(50.0)
    assert cfg.velocity_limit == pytest.approx(12.0)
    assert cfg.armature == pytest.approx(0.05)
    assert cfg.friction == pytest.approx(0.1)
