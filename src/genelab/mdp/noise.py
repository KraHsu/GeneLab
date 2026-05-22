"""Additive noise models for observation corruption (matches mjlab's ``Unoise`` / ``Gnoise``)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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


@dataclass
class ScaledNoise(NoiseCfg):
    """Multiplicative scale-factor noise: ``data * (1 + U(n_min, n_max))``.

    Models a sensor gain / scale-factor error — the corruption grows with the signal
    magnitude (unlike additive ``Unoise`` / ``Gnoise``). Stateless. ``(n_min, n_max)`` are
    fractional bounds, e.g. ``(-0.05, 0.05)`` for ±5% gain error.
    """

    n_min: float = -0.05
    n_max: float = 0.05

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        return data * (1.0 + torch.empty_like(data).uniform_(self.n_min, self.n_max))


@dataclass
class CorrelatedNoise(NoiseCfg):
    """Temporally-correlated (AR(1)) additive noise.

    ``x_t = alpha · x_{t-1} + sqrt(1 − alpha²) · N(0, std²)``; returns ``data + x_t``.
    ``alpha`` in ``[0, 1)`` sets the correlation (``0`` = white = ``Gnoise``; → ``1`` =
    slow drift). The ``sqrt(1 − alpha²)`` mixing keeps the *stationary* std at ``std`` for
    any ``alpha``.

    **Stateful**: the AR state is a per-element buffer carried across steps (lazily sized
    to the observation term, reallocated if the shape changes). It is **not** reset on
    episode boundaries — like a real correlated sensor error that doesn't know about
    resets. Each observation term needs its own instance; the observation manager
    deep-copies the cfg, so separate terms already get separate state.
    """

    std: float = 1.0
    alpha: float = 0.8
    _state: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        state = self._state
        if state is None or state.shape != data.shape:
            state = torch.zeros_like(data)
        white = torch.randn_like(data) * self.std
        state = self.alpha * state + (1.0 - self.alpha**2) ** 0.5 * white
        self._state = state
        return data + state


@dataclass
class BiasDrift(NoiseCfg):
    """Slowly-drifting additive bias (random walk).

    ``bias_t = clip(bias_{t-1} + N(0, drift_std²), ±max_bias)``; returns ``data + bias_t``.
    Models sensor bias instability / thermal drift. ``max_bias`` bounds the walk
    (``None`` = unbounded).

    **Stateful** with the same semantics as :class:`CorrelatedNoise`: a per-element bias
    buffer carried across steps, lazily sized, not reset per episode.
    """

    drift_std: float = 0.01
    max_bias: float | None = None
    _bias: torch.Tensor | None = field(default=None, init=False, repr=False, compare=False)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        bias = self._bias
        if bias is None or bias.shape != data.shape:
            bias = torch.zeros_like(data)
        bias = bias + torch.randn_like(data) * self.drift_std
        if self.max_bias is not None:
            bias = bias.clamp(-self.max_bias, self.max_bias)
        self._bias = bias
        return data + bias
