"""Gymnasium-flavoured VectorEnv protocol used by GeneLab RL wrappers."""

from typing import Any, Protocol

import torch


class VecEnvBase(Protocol):
    num_envs: int
    device: str

    def reset(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]: ...

    def close(self) -> None: ...
