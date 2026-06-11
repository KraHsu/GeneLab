"""Frame-stacking (observation history) in the core ObservationManager.

A group declared with ``history_length=N`` feeds the policy the last ``N`` control-step
frames concatenated oldest->newest (frame-major). Lets a proprioception-only policy infer
velocities / contact trends it can't read instantaneously — the main lever for robust
sim2sim transfer. History is opt-in: ``history_length=1`` (default) is the legacy single-frame
behaviour, byte-for-byte.
"""

from typing import Any

import torch

from genelab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg


class _FakeEnv:
    """Manager-facing env surface plus the reset signal history needs."""

    def __init__(self, num_envs: int = 4, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.device = device
        # Fresh-env detector for history backfill; managers read it after _reset_idx.
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)


def _counter_obs(state: dict[str, float], dim: int):
    """A term whose value advances each compute so history rolls are observable."""

    def fn(env: Any) -> torch.Tensor:
        state["t"] += 1.0
        return torch.full((env.num_envs, dim), state["t"], device=env.device)

    return fn


def test_history_widens_group_and_backfills_first_frame() -> None:
    env = _FakeEnv(num_envs=2)
    cfg = {
        "policy": ObservationGroupCfg(
            history_length=3,
            terms={"a": ObservationTermCfg(func=lambda e: torch.full((e.num_envs, 4), 7.0))},
        )
    }
    mgr = ObservationManager(cfg, env)
    obs = mgr.compute()
    # 3 frames x 4 dims = 12 wide; declared dim reflects the stack.
    assert obs["policy"].shape == (2, 12)
    assert mgr.group_obs_dim("policy") == 12
    # First post-reset frame backfills the whole history (no zero padding).
    assert torch.allclose(obs["policy"], torch.full((2, 12), 7.0))


def test_history_rolls_oldest_to_newest() -> None:
    env = _FakeEnv(num_envs=1)
    state = {"t": 0.0}
    cfg = {
        "policy": ObservationGroupCfg(
            history_length=3,
            terms={"a": ObservationTermCfg(func=_counter_obs(state, 2))},
        )
    }
    mgr = ObservationManager(cfg, env)
    mgr.compute()  # t=1, fresh env -> backfill [1,1,1]
    env.episode_length_buf += 1  # env has now taken a step
    mgr.compute()  # t=2, roll -> [1,1,2]
    env.episode_length_buf += 1
    obs = mgr.compute()  # t=3, roll -> [1,2,3]
    # frame-major, oldest block first, newest last; each block is the 2-dim term value.
    assert torch.allclose(obs["policy"][0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))


def test_history_backfills_only_freshly_reset_envs() -> None:
    env = _FakeEnv(num_envs=2)
    state = {"t": 0.0}
    cfg = {
        "policy": ObservationGroupCfg(
            history_length=3,
            terms={"a": ObservationTermCfg(func=_counter_obs(state, 1))},
        )
    }
    mgr = ObservationManager(cfg, env)
    mgr.compute()  # t=1 backfill both -> [1,1,1]
    env.episode_length_buf += 1
    mgr.compute()  # t=2 roll both -> [1,1,2]
    # Env 0 gets reset (its buf returns to 0); env 1 keeps stepping.
    env.episode_length_buf[0] = 0
    obs = mgr.compute()  # t=3
    # Env 0 backfills the fresh frame across the whole history; env 1 rolls.
    assert torch.allclose(obs["policy"][0], torch.tensor([3.0, 3.0, 3.0]))
    assert torch.allclose(obs["policy"][1], torch.tensor([1.0, 2.0, 3.0]))
