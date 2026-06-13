"""Action post-processing for deploy (pure numpy).

Mirrors ``JointPositionOffsetEMAAction`` from the reorient task: the policy emits
raw actions ~[-1, 1]; the joint target is ``default + scale * clamp(action)``,
clamped to joint limits, EMA-smoothed against the previous target, and held at the
default pose for a warmup window after each reset. (Deploy drops the training-only
``encoder_bias`` / ``action_noise`` terms.)
"""

import numpy as np

from genelab_wuji.deploy.action import ActionProcessor

_N = 20
_DEFAULT = np.linspace(0.1, 0.9, _N)


def _proc(**kw: object) -> ActionProcessor:
    return ActionProcessor(default_joint_pos=_DEFAULT, **kw)  # type: ignore[arg-type]


def test_warmup_holds_default_pose() -> None:
    proc = _proc(action_scale=0.5, ema_alpha=0.5, warmup_steps=3)
    proc.reset()
    for _ in range(3):
        target = proc.process(np.ones(_N))  # large action ignored during warmup
        assert np.allclose(target, _DEFAULT)


def test_first_post_warmup_step_applies_scaled_offset_under_ema() -> None:
    proc = _proc(action_scale=0.5, ema_alpha=0.5, warmup_steps=0)
    proc.reset()
    action = np.full(_N, 0.4)
    target = proc.process(action)
    # prev == default on the first step, so:
    #   smoothed = alpha*(default + scale*action) + (1-alpha)*default
    #            = default + alpha*scale*action
    expected = _DEFAULT + 0.5 * 0.5 * action
    assert np.allclose(target, expected)


def test_target_clamped_to_joint_limits() -> None:
    lo = _DEFAULT - 0.05
    hi = _DEFAULT + 0.05
    proc = _proc(
        action_scale=1.0,
        ema_alpha=1.0,  # no smoothing, so the raw clamp is directly observable
        warmup_steps=0,
        joint_pos_limits=(lo, hi),
    )
    proc.reset()
    target = proc.process(np.ones(_N))  # would push +1.0 past hi without clamping
    assert np.all(target <= hi + 1e-9)
    assert np.allclose(target, hi)
