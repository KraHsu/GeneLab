# pyright: reportIncompatibleMethodOverride=false
# (skrl's ``Model.compute`` return annotation is a malformed 1-tuple; our concrete
#  ``(output, extras)`` returns trip the override check — a skrl-side stub quirk.)
"""Per-algorithm skrl model factory.

Builds the policy / value / critic networks each skrl algorithm needs. We hand-write
small MLP models on skrl's stable mixin API (``GaussianMixin`` / ``DeterministicMixin``
+ ``compute``) rather than the ``model_instantiators`` declarative spec, whose
network-spec format has shifted across skrl releases — the mixin API has not.

This module imports skrl at top level; it is only ever imported lazily (from
``backends.skrl`` when actually building an agent), so ``genelab.rl`` stays
importable without skrl.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

if TYPE_CHECKING:
    import gymnasium

    from genelab.rl.skrl_config import SkrlModelCfg
    from genelab.rl.skrl_wrapper import GenelabSkrlWrapper

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}


def _activation(name: str) -> type[nn.Module]:
    cls = _ACTIVATIONS.get(name.lower())
    if cls is None:
        raise ValueError(f"unsupported activation {name!r}; known: {sorted(_ACTIVATIONS)}")
    return cls


def _mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, activation: str) -> nn.Sequential:
    """Plain MLP: ``in_dim -> hidden... -> out_dim`` with no output activation."""
    act = _activation(activation)
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), act()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(GaussianMixin, Model):
    """Stochastic policy for PPO / A2C / SAC: MLP mean + a learnable log-std."""

    def __init__(
        self,
        observation_space: "gymnasium.spaces.Space[Any]",
        action_space: "gymnasium.spaces.Space[Any]",
        device: Any,
        model_cfg: "SkrlModelCfg",
        clip_actions: bool,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=True,
            min_log_std=model_cfg.min_log_std,
            max_log_std=model_cfg.max_log_std,
        )
        num_obs = cast(int, self.num_observations)
        num_act = cast(int, self.num_actions)
        self.net = _mlp(num_obs, model_cfg.hidden_dims, num_act, model_cfg.activation)
        self.log_std_parameter = nn.Parameter(model_cfg.initial_log_std * torch.ones(num_act))
        self.to(device)

    def compute(self, inputs: Mapping[str, Any], role: str = "") -> tuple[Any, Any, dict[str, Any]]:
        return self.net(inputs["states"]), self.log_std_parameter, {}


class DeterministicPolicy(DeterministicMixin, Model):
    """Deterministic actor for DDPG / TD3: MLP with a tanh-squashed output."""

    def __init__(
        self,
        observation_space: "gymnasium.spaces.Space[Any]",
        action_space: "gymnasium.spaces.Space[Any]",
        device: Any,
        model_cfg: "SkrlModelCfg",
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        num_obs = cast(int, self.num_observations)
        num_act = cast(int, self.num_actions)
        self.net = nn.Sequential(
            _mlp(num_obs, model_cfg.hidden_dims, num_act, model_cfg.activation),
            nn.Tanh(),
        )
        self.to(device)

    def compute(self, inputs: Mapping[str, Any], role: str = "") -> tuple[Any, dict[str, Any]]:
        return self.net(inputs["states"]), {}


class ValueCritic(DeterministicMixin, Model):
    """State-value V(s) for PPO / A2C."""

    def __init__(
        self,
        observation_space: "gymnasium.spaces.Space[Any]",
        action_space: "gymnasium.spaces.Space[Any]",
        device: Any,
        model_cfg: "SkrlModelCfg",
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        num_obs = cast(int, self.num_observations)
        self.net = _mlp(num_obs, model_cfg.hidden_dims, 1, model_cfg.activation)
        self.to(device)

    def compute(self, inputs: Mapping[str, Any], role: str = "") -> tuple[Any, dict[str, Any]]:
        return self.net(inputs["states"]), {}


class QCritic(DeterministicMixin, Model):
    """Action-value Q(s, a) for SAC / TD3 / DDPG."""

    def __init__(
        self,
        observation_space: "gymnasium.spaces.Space[Any]",
        action_space: "gymnasium.spaces.Space[Any]",
        device: Any,
        model_cfg: "SkrlModelCfg",
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        num_obs = cast(int, self.num_observations)
        num_act = cast(int, self.num_actions)
        self.net = _mlp(num_obs + num_act, model_cfg.hidden_dims, 1, model_cfg.activation)
        self.to(device)

    def compute(self, inputs: Mapping[str, Any], role: str = "") -> tuple[Any, dict[str, Any]]:
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


def build_models(
    algorithm: str,
    wrapper: "GenelabSkrlWrapper",
    model_cfg: "SkrlModelCfg",
    device: Any,
) -> dict[str, Model]:
    """Return the model dict skrl's agent constructor expects for ``algorithm``.

    Keys per algorithm:
      * PPO / A2C: ``policy``, ``value``
      * DDPG: ``policy``, ``target_policy``, ``critic``, ``target_critic``
      * TD3: ``policy``, ``target_policy``, ``critic_1/2``, ``target_critic_1/2``
      * SAC: ``policy``, ``critic_1/2``, ``target_critic_1/2``
    """
    obs, act = wrapper.observation_space, wrapper.action_space

    def gaussian(clip_actions: bool) -> Model:
        return GaussianPolicy(obs, act, device, model_cfg, clip_actions)

    def det_policy() -> Model:
        return DeterministicPolicy(obs, act, device, model_cfg)

    def value() -> Model:
        return ValueCritic(obs, act, device, model_cfg)

    def q() -> Model:
        return QCritic(obs, act, device, model_cfg)

    if algorithm in ("PPO", "A2C"):
        return {"policy": gaussian(model_cfg.clip_actions), "value": value()}
    if algorithm == "DDPG":
        return {
            "policy": det_policy(), "target_policy": det_policy(),
            "critic": q(), "target_critic": q(),
        }
    if algorithm == "TD3":
        return {
            "policy": det_policy(), "target_policy": det_policy(),
            "critic_1": q(), "critic_2": q(),
            "target_critic_1": q(), "target_critic_2": q(),
        }
    if algorithm == "SAC":
        return {
            "policy": gaussian(clip_actions=True),
            "critic_1": q(), "critic_2": q(),
            "target_critic_1": q(), "target_critic_2": q(),
        }
    raise ValueError(f"unsupported skrl algorithm {algorithm!r}")
