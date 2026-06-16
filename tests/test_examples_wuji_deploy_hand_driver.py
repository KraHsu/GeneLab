"""Hand-driver abstraction for deploy: hardware-agnostic interface + mock.

The control loop talks to ``HandDriverBase``; ``MockHandDriver`` lets the whole
pipeline run (and be tested) without the real ``wujihandpy`` hand. The encoder
joint order matches the GeneLab policy order (finger1_joint1..4, finger2...),
which is also ``wujihandpy``'s (5, 4) row-major flatten — so no remap is needed.
"""

import numpy as np

from genelab_wuji.deploy.config import default_joint_pos
from genelab_wuji.deploy.hand_driver import MockHandDriver, _home_ramp  # pyright: ignore[reportPrivateUsage]
from genelab_wuji.reorient.constants import REORIENT_JOINT_POS


def test_mock_driver_echoes_written_target() -> None:
    driver = MockHandDriver()
    target = np.linspace(-0.3, 0.3, 20)
    driver.write_target(target)
    assert np.allclose(driver.read_encoders(), target)


def test_mock_driver_encoder_order_matches_policy_joint_order() -> None:
    driver = MockHandDriver()
    assert tuple(driver.joint_names_in_encoder_order()) == tuple(REORIENT_JOINT_POS)


def test_mock_driver_home_sets_grasp_keyframe() -> None:
    driver = MockHandDriver()
    driver.write_target(np.zeros(20))
    driver.home()
    expected = np.array(list(REORIENT_JOINT_POS.values()))
    assert np.allclose(driver.read_encoders(), expected)


def test_mock_driver_home_accepts_duration_arg() -> None:
    # The 3s ramp is a real-driver concern; the mock ignores duration_s but must
    # accept it so the shared HandDriverBase / DeployController.reset call works.
    driver = MockHandDriver()
    driver.write_target(np.zeros(20))
    driver.home(duration_s=3.0)
    assert np.allclose(driver.read_encoders(), default_joint_pos())


def test_home_ramp_is_monotone_smoothstep_ending_at_target() -> None:
    current = np.zeros(20)
    target = default_joint_pos()
    ramp = _home_ramp(current, target, steps=150)  # 3s @ 50 Hz
    assert ramp.shape == (150, 20)
    # Smoothstep: starts eased-in near current, ends exactly at target.
    assert np.allclose(ramp[-1], target)
    assert not np.allclose(ramp[0], target)
    # Each joint moves monotonically from current toward target (no overshoot).
    deltas = np.diff(ramp, axis=0)
    signs = np.sign(target - current)
    assert np.all(deltas * signs[None, :] >= -1e-9)
    assert ramp.max() <= max(target.max(), current.max()) + 1e-9
    assert ramp.min() >= min(target.min(), current.min()) - 1e-9


def test_home_ramp_single_step_hits_target() -> None:
    target = default_joint_pos()
    ramp = _home_ramp(np.zeros(20), target, steps=1)
    assert ramp.shape == (1, 20)
    assert np.allclose(ramp[0], target)
