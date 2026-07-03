"""Pin the deploy joint-order remap against the real Genesis articulation order.

The encoder / wujihandpy order (``JOINT_NAMES_20``) is finger-major; the shipped
policy's order (``POLICY_JOINT_NAMES``) is joint-major — Genesis 1.0.x enumerated
the articulation breadth-first, and the deployed checkpoint was trained under that
order. ``DeployController`` remaps between them. If they drift, the real hand gets
scrambled joint obs + actions and twitches without manipulating the cube (the
real-hand 0%-success bug).

Genesis 1.2 parses kinematic trees depth-first, so the live articulation now
enumerates finger-major (identical to the encoder order). ``POLICY_JOINT_NAMES``
deliberately stays the *trained artifact's* order; the env-order drift guard below
pins the new depth-first enumeration instead, so any future Genesis ordering change
is caught again.
"""

import numpy as np
import pytest

from genelab_wuji.deploy.config import (
    ENC_TO_POLICY,
    JOINT_NAMES_20,
    POLICY_JOINT_NAMES,
    default_joint_pos,
    default_joint_pos_policy,
)


def test_enc_to_policy_is_a_valid_permutation() -> None:
    assert sorted(ENC_TO_POLICY) == list(range(20))
    assert set(POLICY_JOINT_NAMES) == set(JOINT_NAMES_20)
    # Policy order is joint-major: the first five are every finger's joint1.
    assert POLICY_JOINT_NAMES[:5] == tuple(f"right_finger{f}_joint1" for f in range(1, 6))
    # Encoder order is finger-major: the first four are finger1's joints 1..4.
    assert JOINT_NAMES_20[:4] == tuple(f"right_finger1_joint{j}" for j in range(1, 5))


def test_default_policy_is_default_reordered() -> None:
    d = default_joint_pos()
    dp = default_joint_pos_policy()
    assert np.allclose(dp, d[list(ENC_TO_POLICY)])
    # Round-trips back to encoder order via the inverse permutation.
    assert np.allclose(dp[np.argsort(ENC_TO_POLICY)], d)


def test_env_joint_order_is_depth_first() -> None:
    """Drift guard: the built env must enumerate joints depth-first (finger-major).

    Genesis 1.2 parses kinematic trees depth-first, which for the hand coincides
    with the encoder order ``JOINT_NAMES_20``. A policy trained under Genesis 1.2
    therefore has finger-major obs/action layouts, while the shipped checkpoint
    (trained under 1.0.x, breadth-first) keeps the joint-major ``POLICY_JOINT_NAMES``
    — sim2sim replay of that checkpoint needs the name-based remap, and a retrained
    policy needs an updated ``ENC_TO_POLICY``. This assert exists so any future
    Genesis enumeration change is caught here instead of as scrambled joints.
    """
    pytest.importorskip("genesis")
    from genelab_wuji.deploy.scripts._env import build_reorient_env

    env = None
    try:
        env = build_reorient_env(num_envs=1)
        assert list(env.scene["robot"].joint_names) == list(JOINT_NAMES_20)
    except Exception as exc:  # asset download / GPU / display unavailable in minimal CI
        if env is None:
            pytest.skip(f"reorient env unavailable: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
