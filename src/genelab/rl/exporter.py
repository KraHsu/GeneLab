"""``genelab export`` — TorchScript / ONNX export of a trained policy.

Wraps the backend's actor ``nn.Module`` in :class:`ExportedPolicy`, which bakes
per-term obs ``scale`` / ``clip`` into a single ``forward(raw_obs) -> actions`` pass.
Deployment environments only need ``torch`` (TorchScript) or an ONNX runtime; no
``rsl_rl`` / ``skrl`` / ``stable_baselines3`` import is required at inference time.

The exported file is accompanied by ``<output>.metadata.json`` recording the obs
schema (term names, dims, scale/clip), action dim/range, and provenance fields,
so deployers know which raw-obs layout to feed.
"""

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass
class ExportConfig:
    format: Literal["torchscript", "onnx"]
    output: Path
    opset: int = 17  # ONNX-only; ignored for torchscript


@dataclass
class _TermSpec:
    name: str
    start: int
    end: int
    scale: float | None
    clip: tuple[float, float] | None


@dataclass
class _GroupSpec:
    """Flat-obs layout of one observation group within the concatenated input."""

    name: str
    start: int
    dim: int
    terms: list[_TermSpec]


_BIG = 1.0e30  # finite stand-in for ±inf in TorchScript / ONNX-friendly clamps


def _resolve_obs_specs(
    env: Any, groups: Sequence[str]
) -> tuple[list[_TermSpec], int, list[_GroupSpec]]:
    """Return ``(flat_term_specs, total_obs_dim, group_specs)`` for ``groups``.

    Resolves each group's terms in order with a *global* cursor, so term ``start`` /
    ``end`` are offsets into the single flat obs tensor that concatenates the groups
    (`[groups[0], groups[1], ...]`). Goal-conditioned (SAC+HER) policies pass the
    policy group plus the achieved/desired-goal groups here so the exported model
    takes one flat tensor and reconstructs the Dict the SB3 actor expects.

    Probes the env once to read each term's tensor width — the observation manager
    only materializes shapes during ``compute``, so we go through it instead of
    introspecting cfg.
    """
    obs_dict = env.observation_manager.compute()
    specs: list[_TermSpec] = []
    group_specs: list[_GroupSpec] = []
    cursor = 0
    for group_name in groups:
        if group_name not in obs_dict:
            raise SystemExit(
                f"obs group {group_name!r} not produced by env; available: {sorted(obs_dict)}"
            )
        group_cfg = env.observation_manager.cfg[group_name]
        group_start = cursor
        group_terms: list[_TermSpec] = []
        for term_name, term_cfg in group_cfg.terms.items():
            value = term_cfg.func(env, **term_cfg.params)
            width = int(value.shape[-1]) if value.dim() > 1 else 1
            spec = _TermSpec(
                name=term_name,
                start=cursor,
                end=cursor + width,
                scale=term_cfg.scale,
                clip=term_cfg.clip,
            )
            specs.append(spec)
            group_terms.append(spec)
            cursor += width
        group_specs.append(
            _GroupSpec(
                name=group_name, start=group_start, dim=cursor - group_start, terms=group_terms
            )
        )
    return specs, cursor, group_specs


def _scale_clip_vecs(specs: list[_TermSpec], obs_dim: int) -> tuple[Any, Any, Any]:
    """Return ``(scale, clip_lo, clip_hi)`` tensors baking each term's scale/clip by offset."""
    import torch

    scale_vec = torch.ones(obs_dim, dtype=torch.float32)
    lo_vec = torch.full((obs_dim,), -_BIG, dtype=torch.float32)
    hi_vec = torch.full((obs_dim,), _BIG, dtype=torch.float32)
    for spec in specs:
        if spec.scale is not None:
            scale_vec[spec.start : spec.end] = float(spec.scale)
        if spec.clip is not None:
            lo_vec[spec.start : spec.end] = float(spec.clip[0])
            hi_vec[spec.start : spec.end] = float(spec.clip[1])
    return scale_vec, lo_vec, hi_vec


def _build_exported_module(actor: "nn.Module", specs: list[_TermSpec], obs_dim: int) -> Any:
    """Return an ``nn.Module`` that bakes per-term scale/clip in front of ``actor``."""
    import torch
    import torch.nn as nn

    scale_vec, lo_vec, hi_vec = _scale_clip_vecs(specs, obs_dim)

    class _ExportedPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = actor
            self.register_buffer("_obs_scale", scale_vec)
            self.register_buffer("_obs_clip_lo", lo_vec)
            self.register_buffer("_obs_clip_hi", hi_vec)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # ``register_buffer`` declares ``Tensor | Module``; the assignment above
            # is always a tensor so the cast is safe and keeps pyright quiet.
            scale = cast(torch.Tensor, self._obs_scale)
            clip_lo = cast(torch.Tensor, self._obs_clip_lo)
            clip_hi = cast(torch.Tensor, self._obs_clip_hi)
            x = obs * scale
            x = torch.maximum(x, clip_lo)
            x = torch.minimum(x, clip_hi)
            return self.actor(x)

    return _ExportedPolicy()


def _recurrent_meta(actor: Any) -> dict[str, Any]:
    """Read ``{rnn_type, rnn_num_layers, rnn_hidden_dim}`` off an rsl_rl ``RNNModel``.

    Inspects the underlying ``nn.LSTM`` / ``nn.GRU`` (``actor.rnn.rnn``) so the metadata
    reflects the trained network rather than the requested cfg.
    """
    import torch.nn as nn

    # Defensive access (mirrors ``_extract_rsl_rl_actor``): ``actor.rnn.rnn`` is the raw
    # torch nn.LSTM / nn.GRU inside rsl_rl's RNN wrapper. Fall back gracefully rather than
    # raising an opaque AttributeError if a future rsl_rl renames the attribute chain.
    rnn = getattr(getattr(actor, "rnn", None), "rnn", None)
    rnn_type = "gru" if isinstance(rnn, nn.GRU) else "lstm"
    return {
        "rnn_type": rnn_type,
        "rnn_num_layers": int(getattr(rnn, "num_layers", 1)),
        "rnn_hidden_dim": int(getattr(rnn, "hidden_size", 0)),
    }


def _build_recurrent_jit_module(actor: Any, specs: list[_TermSpec], obs_dim: int) -> Any:
    """Return a *scriptable* module: per-term scale/clip in front of ``actor.as_jit()``.

    rsl_rl's ``as_jit()`` wrapper is stateful — ``forward(obs) -> actions`` keeps the
    hidden state in an internal buffer (fixed batch 1) and exposes ``reset()``. We bake
    scale/clip in front and re-export ``reset`` so the deployment interface stays the
    single-input MLP shape plus an explicit episode-boundary reset.
    """
    import torch
    import torch.nn as nn

    scale_vec, lo_vec, hi_vec = _scale_clip_vecs(specs, obs_dim)
    inner = actor.as_jit()

    class _RecurrentExportedPolicy(nn.Module):
        # Class-level annotations type the registered buffers as tensors for both pyright
        # and TorchScript's scripting compiler (which reads them when building the graph).
        _obs_scale: torch.Tensor
        _obs_clip_lo: torch.Tensor
        _obs_clip_hi: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.actor = inner
            self.register_buffer("_obs_scale", scale_vec)
            self.register_buffer("_obs_clip_lo", lo_vec)
            self.register_buffer("_obs_clip_hi", hi_vec)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            x = obs * self._obs_scale
            x = torch.maximum(x, self._obs_clip_lo)
            x = torch.minimum(x, self._obs_clip_hi)
            return self.actor(x)

        @torch.jit.export
        def reset(self) -> None:
            """Zero the recurrent hidden state — call at episode boundaries."""
            self.actor.reset()

    return _RecurrentExportedPolicy()


def _build_recurrent_onnx_module(onnx_inner: Any, specs: list[_TermSpec], obs_dim: int) -> Any:
    """Return a module wrapping rsl_rl's ``as_onnx()`` with per-term scale/clip in front.

    rsl_rl's ONNX wrapper takes explicit hidden state: ``forward(obs, h_in[, c_in]) ->
    (actions, h_out[, c_out])`` (the cell tensors are LSTM-only; GRU passes/returns
    ``None``). We only transform ``obs`` and delegate, preserving the signature so the
    exported graph keeps the hidden-state ports rsl_rl declares in ``input_names`` /
    ``output_names``.
    """
    import torch
    import torch.nn as nn

    scale_vec, lo_vec, hi_vec = _scale_clip_vecs(specs, obs_dim)

    class _RecurrentOnnxPolicy(nn.Module):
        _obs_scale: torch.Tensor
        _obs_clip_lo: torch.Tensor
        _obs_clip_hi: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.actor = onnx_inner
            self.register_buffer("_obs_scale", scale_vec)
            self.register_buffer("_obs_clip_lo", lo_vec)
            self.register_buffer("_obs_clip_hi", hi_vec)

        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
        ) -> Any:
            x = obs * self._obs_scale
            x = torch.maximum(x, self._obs_clip_lo)
            x = torch.minimum(x, self._obs_clip_hi)
            return self.actor(x, h_in, c_in)

    return _RecurrentOnnxPolicy()


def _export_recurrent(
    actor: Any, specs: list[_TermSpec], obs_dim: int, cfg: ExportConfig, output: Path
) -> None:
    """Serialize a recurrent ``RNNModel`` to TorchScript or ONNX via rsl_rl's wrappers."""
    import torch

    if cfg.format == "torchscript":
        module = _build_recurrent_jit_module(actor, specs, obs_dim)
        module.eval()
        # Script (not trace): the stateful ``as_jit()`` wrapper mutates its hidden-state
        # buffer in ``forward`` and exposes ``reset()`` — tracing would freeze both.
        with torch.inference_mode():
            scripted = torch.jit.script(module)
        torch.jit.save(scripted, str(output))  # type: ignore[arg-type]
    elif cfg.format == "onnx":
        onnx_inner = actor.as_onnx()
        onnx_inner.eval()
        module = _build_recurrent_onnx_module(onnx_inner, specs, obs_dim)
        module.eval()
        dummy_obs, *dummy_state = onnx_inner.get_dummy_inputs()
        # The obs values are arbitrary for tracing (the graph is shape/structure, not data);
        # keep the rsl_rl-provided hidden-state dummies so the tracer sees the right ranks.
        dummy = (torch.zeros_like(dummy_obs), *dummy_state)
        input_names = list(onnx_inner.input_names)
        output_names = list(onnx_inner.output_names)
        dynamic_axes: dict[str, dict[int, str]] = {
            "obs": {0: "batch"},
            "actions": {0: "batch"},
        }
        # Hidden-state tensors are (num_layers, batch, hidden_dim) → batch is axis 1.
        for name in input_names[1:] + output_names[1:]:
            dynamic_axes[name] = {1: "batch"}
        with torch.inference_mode():
            torch.onnx.export(
                module,
                tuple(dummy),
                str(output),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=cfg.opset,
                dynamo=False,
            )
    else:
        raise SystemExit(f"unknown export format {cfg.format!r}; use torchscript or onnx")


def _metadata(
    task_id: str,
    checkpoint: Path,
    group_specs: list[_GroupSpec],
    obs_dim: int,
    action_dim: int,
    cfg: ExportConfig,
    recurrent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    meta: dict[str, Any] = {
        "task": task_id,
        "checkpoint": str(checkpoint),
        # One entry per group, in the order they are concatenated into the flat
        # ``obs`` input. ``start`` is the group's offset into that flat tensor;
        # term ``start`` is the term's (global) flat-obs offset.
        "obs_dim": obs_dim,
        "obs_groups": {
            g.name: {
                "start": g.start,
                "dim": g.dim,
                "terms": [
                    {
                        "name": s.name,
                        "dim": s.end - s.start,
                        "start": s.start,
                        "scale": s.scale,
                        "clip": list(s.clip) if s.clip is not None else None,
                    }
                    for s in g.terms
                ],
            }
            for g in group_specs
        },
        "action_dim": action_dim,
        "action_range": [-1.0, 1.0],
        "normalization_baked": True,
        "is_recurrent": recurrent is not None,
        "format": cfg.format,
        "opset": cfg.opset if cfg.format == "onnx" else None,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
    }
    if recurrent is not None:
        # Hidden-state interface for deployers. The TorchScript export hides the state in
        # an internal buffer (call ``model.reset()`` at episode boundaries); the ONNX
        # export takes/returns it explicitly via the named ports below.
        num_layers = recurrent["rnn_num_layers"]
        hidden_dim = recurrent["rnn_hidden_dim"]
        is_lstm = recurrent["rnn_type"] == "lstm"
        meta["recurrent"] = {
            **recurrent,
            # ``batch`` is dynamic in ONNX; the TorchScript buffer is fixed at batch 1.
            "hidden_state_shape": [num_layers, "batch", hidden_dim],
            "onnx_inputs": ["obs", "h_in", "c_in"] if is_lstm else ["obs", "h_in"],
            "onnx_outputs": ["actions", "h_out", "c_out"] if is_lstm else ["actions", "h_out"],
        }
    return meta


def export_policy(
    *,
    task_id: str,
    env: Any,
    checkpoint: Path,
    actor: "nn.Module",
    actor_input_dim: int,
    action_dim: int,
    policy_group: str,
    goal_groups: Sequence[str] = (),
    cfg: ExportConfig,
    is_recurrent: bool = False,
) -> Path:
    """Serialize ``actor`` (with baked per-term obs normalization) to ``cfg.output``.

    ``goal_groups`` are extra observation groups concatenated after ``policy_group``
    into the single flat ``obs`` input — used by goal-conditioned (SAC+HER) policies
    whose actor consumes a ``Dict`` of ``observation`` + ``achieved_goal`` +
    ``desired_goal``; the actor module reconstructs that Dict from the flat tensor.

    When ``is_recurrent`` is set, ``actor`` is an rsl_rl ``RNNModel``: TorchScript export
    scripts its stateful ``as_jit()`` wrapper (hidden state internal, plus ``reset()``)
    and ONNX export wraps its ``as_onnx()`` wrapper (explicit ``h_in``/``c_in`` ports).

    Writes a sibling ``<output>.metadata.json`` describing the obs schema. Returns
    the path to the serialized model file.
    """
    import torch

    specs, obs_dim, group_specs = _resolve_obs_specs(env, [policy_group, *goal_groups])
    if actor_input_dim and obs_dim != actor_input_dim:
        # Don't hard-fail — the backend may have a deeper preprocessing layer; just
        # warn so the user can compare shapes against the metadata.
        import warnings

        warnings.warn(
            f"obs dim mismatch: env produces {obs_dim}, actor expects {actor_input_dim}. "
            "Exported obs schema reflects the env; deployment-side shapes may need "
            "adjustment.",
            UserWarning,
            stacklevel=2,
        )

    actor.eval()
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    recurrent_meta: dict[str, Any] | None = _recurrent_meta(actor) if is_recurrent else None

    if is_recurrent:
        _export_recurrent(actor, specs, obs_dim, cfg, output)
    else:
        module = _build_exported_module(actor, specs, obs_dim)
        module.eval()
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        if cfg.format == "torchscript":
            with torch.inference_mode():
                scripted = torch.jit.trace(module, dummy, check_trace=False)
            # ``torch.jit.trace`` returns a ``ScriptModule`` at runtime; pyright's stub
            # over-narrows the return so we go through the module-level ``save``.
            torch.jit.save(scripted, str(output))  # type: ignore[arg-type]
        elif cfg.format == "onnx":
            with torch.inference_mode():
                torch.onnx.export(
                    module,
                    (dummy,),
                    str(output),
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                    opset_version=cfg.opset,
                    # Use the legacy TorchScript-based exporter (matches the torchscript
                    # trace path above). The torch.export-based default (torch>=2.9)
                    # can't trace SB3 SAC's ``Normal`` distribution construction inside
                    # ``_predict`` (data-dependent guard error); the legacy tracer
                    # handles every backend's actor.
                    dynamo=False,
                )
        else:
            raise SystemExit(f"unknown export format {cfg.format!r}; use torchscript or onnx")

    meta = _metadata(
        task_id=task_id,
        checkpoint=checkpoint,
        group_specs=group_specs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=cfg,
        recurrent=recurrent_meta,
    )
    meta_path = output.with_suffix(output.suffix + ".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return output
