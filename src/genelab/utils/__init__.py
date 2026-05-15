"""Shared utility helpers for GeneLab."""

from typing import TYPE_CHECKING

__all__ = [
    "PROGRESS_CALLBACK",
    "AssetDownloadError",
    "AssetSpec",
    "ProgressCallback",
    "fetch_asset",
    "matrix_from_quat",
    "normalize",
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_error_magnitude",
    "quat_from_euler_xyz",
    "quat_inv",
    "quat_mul",
    "sample_uniform",
    "subtract_frame_transforms",
    "yaw_quat",
]


if TYPE_CHECKING:
    from genelab.utils.download import (
        PROGRESS_CALLBACK,
        AssetDownloadError,
        AssetSpec,
        ProgressCallback,
        fetch_asset,
    )
    from genelab.utils.math import (
        matrix_from_quat,
        normalize,
        quat_apply,
        quat_apply_inverse,
        quat_conjugate,
        quat_error_magnitude,
        quat_from_euler_xyz,
        quat_inv,
        quat_mul,
        sample_uniform,
        subtract_frame_transforms,
        yaw_quat,
    )


# Re-export torch-bound math helpers lazily so importing the torch-free
# ``genelab.utils.download`` submodule does not transitively drag torch in via
# ``genelab.utils.__init__``. First access resolves the real attribute and caches it.
_LAZY_MATH_EXPORTS = frozenset(
    {
        "matrix_from_quat",
        "normalize",
        "quat_apply",
        "quat_apply_inverse",
        "quat_conjugate",
        "quat_error_magnitude",
        "quat_from_euler_xyz",
        "quat_inv",
        "quat_mul",
        "sample_uniform",
        "subtract_frame_transforms",
        "yaw_quat",
    }
)
_LAZY_DOWNLOAD_EXPORTS = frozenset(
    {
        "PROGRESS_CALLBACK",
        "AssetDownloadError",
        "AssetSpec",
        "ProgressCallback",
        "fetch_asset",
    }
)


def __getattr__(name: str) -> object:
    if name in _LAZY_MATH_EXPORTS:
        from genelab.utils import math as _math

        value = getattr(_math, name)
        globals()[name] = value
        return value
    if name in _LAZY_DOWNLOAD_EXPORTS:
        from genelab.utils import download as _download

        value = getattr(_download, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MATH_EXPORTS | _LAZY_DOWNLOAD_EXPORTS)
