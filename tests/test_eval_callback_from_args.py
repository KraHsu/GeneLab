"""Tests for ``EvalCallbackCfg.from_args``.

This moves the `--eval-*` runner-arg parsing out of
``cli/__init__.py:_build_eval_callback`` and onto the domain config
(``EvalCallbackCfg.from_args``). These tests pin the parse behaviour —
the matrix of which `--eval-*` flags are set — directly against the
classmethod, so the CLI dispatcher only has to forward the raw flag
dict. Behaviour is byte-for-byte what ``_build_eval_callback`` did
before the move.
"""

from __future__ import annotations

from genelab.rl.eval_callback import EvalCallbackCfg


def test_from_args_none_when_eval_every_unset() -> None:
    """No ``--eval-every`` → ``None`` (keeps the legacy single-shot train path)."""
    assert EvalCallbackCfg.from_args({}) is None
    # Other --eval-* flags without --eval-every are still a no-op.
    assert EvalCallbackCfg.from_args({"eval_episodes": "50", "eval_seed": "7"}) is None


def test_from_args_eval_every_only_uses_field_defaults() -> None:
    """``--eval-every`` alone → enabled cfg with the dataclass defaults for the rest."""
    cfg = EvalCallbackCfg.from_args({"eval_every": "100"})
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.eval_every_iters == 100
    assert cfg.eval_episodes == 10  # default
    assert cfg.eval_num_envs is None  # default
    assert cfg.eval_seed == 0  # default


def test_from_args_full_flag_set() -> None:
    """Every ``--eval-*`` flag set is parsed and coerced to int."""
    cfg = EvalCallbackCfg.from_args(
        {
            "eval_every": "250",
            "eval_episodes": "32",
            "eval_num_envs": "16",
            "eval_seed": "3",
        }
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.eval_every_iters == 250
    assert cfg.eval_episodes == 32
    assert cfg.eval_num_envs == 16
    assert cfg.eval_seed == 3


def test_from_args_ignores_unrelated_keys() -> None:
    """Non-``--eval-*`` runner args are ignored (the dispatcher forwards the whole dict)."""
    cfg = EvalCallbackCfg.from_args(
        {"eval_every": "10", "max_iterations": "1000", "seed": "42", "gpus": "2"}
    )
    assert cfg is not None
    assert cfg.eval_every_iters == 10
    # The unrelated keys do not bleed into the cfg.
    assert cfg.eval_seed == 0  # 'seed' is NOT 'eval_seed'
    assert cfg.eval_num_envs is None


def test_from_args_returns_eval_callback_cfg_type() -> None:
    """The classmethod returns the concrete ``EvalCallbackCfg`` (so ``cls`` is honoured)."""
    cfg = EvalCallbackCfg.from_args({"eval_every": "5"})
    assert type(cfg) is EvalCallbackCfg
