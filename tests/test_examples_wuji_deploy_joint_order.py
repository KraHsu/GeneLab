"""Pin the deploy joint-order remap against the real Genesis articulation order.

The encoder / wujihandpy order (``JOINT_NAMES_20``) is finger-major; the policy /
Genesis articulation order (``POLICY_JOINT_NAMES``) is joint-major. ``DeployController``
remaps between them. If they drift, the real hand gets scrambled joint obs + actions and
twitches without manipulating the cube (the real-hand 0%-success bug).
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


def test_policy_joint_order_matches_env() -> None:
    """Drift guard: POLICY_JOINT_NAMES must equal the built env's articulation order."""
    pytest.importorskip("genesis")
    from genelab_wuji.deploy.scripts._env import build_reorient_env

    env = None
    try:
        env = build_reorient_env(num_envs=1)
        assert list(env.scene["robot"].joint_names) == list(POLICY_JOINT_NAMES)
    except Exception as exc:  # asset download / GPU / display unavailable in minimal CI
        if env is None:
            pytest.skip(f"reorient env unavailable: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
