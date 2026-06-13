"""Thin ONNX policy wrapper for deploy (ported from wuji-mjlab, GeneLab metadata).

Obs assembly and action post-processing live elsewhere (``DeployObsBuilder`` /
the action term), so this is just: load the session, introspect dims, optionally
read the sibling ``<file>.metadata.json`` GeneLab's exporter writes, and run a
single forward pass. Normalization is baked into the graph, so no normalizer is
needed at deploy time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort


class ONNXPolicy:
    """Single-step ONNX policy.

    Args:
        onnx_path: Path to the ``policy.onnx`` exported by ``genelab export``.
        metadata_path: Optional metadata sidecar. When ``None``, looks for
            ``<onnx_path>.metadata.json`` then ``<dir>/metadata.json``.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        onnx_path = str(onnx_path)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        self.onnx_path: str = onnx_path
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name: str = inp.name
        self.output_name: str = out.name
        # Shapes are (batch, N); axis 0 may be a dynamic symbol — take the last dim.
        self.input_dim: int = int(inp.shape[-1])
        self.action_dim: int = int(out.shape[-1])
        self.metadata: dict[str, Any] = self._load_metadata(onnx_path, metadata_path)

    @staticmethod
    def _load_metadata(
        onnx_path: str, metadata_path: Optional[str | Path]
    ) -> dict[str, Any]:
        candidates = (
            [str(metadata_path)]
            if metadata_path is not None
            else [
                onnx_path + ".metadata.json",
                os.path.join(os.path.dirname(onnx_path), "metadata.json"),
            ]
        )
        for candidate in candidates:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    return json.load(f)
        return {}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Single forward pass; accepts ``(input_dim,)`` or ``(1, input_dim)``."""
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape != (1, self.input_dim):
            raise ValueError(
                f"obs shape {obs.shape}, expected (1, {self.input_dim})"
            )
        obs = obs.astype(np.float32, copy=False)
        result = self.session.run([self.output_name], {self.input_name: obs})[0]
        return result.squeeze(0)
