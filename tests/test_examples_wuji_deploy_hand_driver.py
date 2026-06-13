"""Hand-driver abstraction for deploy: hardware-agnostic interface + mock.

The control loop talks to ``HandDriverBase``; ``MockHandDriver`` lets the whole
pipeline run (and be tested) without the real ``wujihandpy`` hand. The encoder
joint order matches the GeneLab policy order (finger1_joint1..4, finger2...),
which is also ``wujihandpy``'s (5, 4) row-major flatten — so no remap is needed.
"""

import numpy as np

from genelab_wuji.deploy.hand_driver import MockHandDriver
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
