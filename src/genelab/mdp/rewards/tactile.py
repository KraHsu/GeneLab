"""Tactile-aware reward primitives.

Generic over the new tactile sensor wrappers (:class:`~genelab.sensor.ElastomerTactileSensor`,
:class:`~genelab.sensor.PointCloudTactileSensor`): both expose ``data.raw`` as the
Genesis sensor's per-step tensor — ``(num_envs, [history,] ...)`` — and these
primitives reduce over every non-batch axis. ``contact_count`` thresholds and
counts; ``contact_intensity_l2`` returns the squared sum across probes.

Reward shape is ``(num_envs,)`` so the reward manager can apply weights and
combine with other terms. Pair with negative weights to penalize tactile
intensity, positive weights to encourage contact-presence.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from genelab.contracts import EnvContext


def contact_intensity_l2(env: "EnvContext", sensor_name: str) -> torch.Tensor:
    """``Σ raw²`` over every non-batch axis of the named tactile sensor.

    Maps small-magnitude touches to small reward, hard pushes to large reward.
    Sum is taken over every axis after the batch axis so the same primitive works
    for ``(B, P)``, ``(B, P, C)``, and ``(B, H, P, C)`` raw outputs.
    """
    raw = env.sensors[sensor_name].data.raw
    return raw.flatten(start_dim=1).pow(2).sum(dim=-1)


def contact_count(env: "EnvContext", sensor_name: str, threshold: float = 0.0) -> torch.Tensor:
    """Per-env count of probes with ``|raw| > threshold``.

    Sum is taken over every non-batch axis. For sensors whose ``raw`` is signed
    (displacement / force), the absolute value masks the sign; for non-negative
    outputs (depth, distance) it's a no-op. Threshold defaults to ``0.0`` so any
    non-zero reading counts.
    """
    raw = env.sensors[sensor_name].data.raw
    mask = raw.abs() > threshold
    return mask.flatten(start_dim=1).sum(dim=-1).to(raw.dtype)


def slip_penalty(env: "EnvContext", sensor_name: str) -> torch.Tensor:
    """``Σ raw[..., :2]²`` — squared lateral (xy) magnitude per env.

    Both :class:`~genelab.sensor.ElastomerTactileSensor` (displacement vector per
    probe) and :class:`~genelab.sensor.PointCloudTactileSensor` (force vector per
    probe — ``data.raw`` aliases ``data.force``) expose ``data.raw`` of shape
    ``(num_envs, [history,] num_probes, 3)`` with the third channel as the probe
    normal. The xy slice is therefore the tangential plane: a non-zero lateral
    signal while in contact is slip. Pair with a negative weight to penalize.
    """
    raw = env.sensors[sensor_name].data.raw
    lateral = raw[..., :2]
    return lateral.flatten(start_dim=1).pow(2).sum(dim=-1)
