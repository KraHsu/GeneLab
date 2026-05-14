"""Additive noise models for observation corruption (matches mjlab's ``Unoise`` / ``Gnoise``)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class NoiseCfg(ABC):
    """Base config for additive noise injected into an observation term."""

    @abstractmethod
    def apply(self, data: torch.Tensor) -> torch.Tensor: ...


@dataclass
class Unoise(NoiseCfg):
    """Uniform additive noise in ``[n_min, n_max]``."""

    n_min: float = -1.0
    n_max: float = 1.0

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return data + torch.empty_like(data).uniform_(self.n_min, self.n_max)


@dataclass
class Gnoise(NoiseCfg):
    """Gaussian additive noise: ``data + N(mean, std^2)``."""

    mean: float = 0.0
    std: float = 1.0

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return data + torch.randn_like(data) * self.std + self.mean
