"""Asset download helper for the GeneLab asset zoo.

Robot configurations registered under :mod:`genelab.asset_zoo` declare their MJCF / URDF
sources as :class:`AssetSpec` instances. :func:`fetch_asset` downloads the file once,
verifies its md5, and caches it under ``<project_root>/.cache/assets/<name>/<md5>/`` so
repeated CLI invocations stay offline. Failures (network error, md5 mismatch, malformed
URL) all surface as :class:`AssetDownloadError` with the expected vs actual digest
included in the message — easier to copy-paste into a follow-up commit that updates the
hash after a new upload.

Stdlib only: no ``requests`` / ``httpx`` dependency, no concurrent / resumable downloads.
"""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from genelab.cache import CACHE_DIR

_ASSET_ROOT = CACHE_DIR / "assets"
_CHUNK_SIZE = 1 << 16  # 64 KiB


class AssetDownloadError(RuntimeError):
    """Raised when an asset cannot be retrieved or its md5 does not match the spec."""


@dataclass(frozen=True)
class AssetSpec:
    """Declarative descriptor for a downloadable asset.

    Robot cfg modules build a module-level constant such as
    ``MJCF = AssetSpec(name="cartpole", url="https://.../cartpole.xml", md5=..., filename=...)``
    and then call :func:`fetch_asset` lazily inside the registered factory so commands
    like ``genelab list robots`` never touch the network.
    """

    name: str
    url: str
    md5: str
    filename: str


def fetch_asset(spec: AssetSpec, *, force: bool = False) -> Path:
    """Return the local path to ``spec``, downloading and md5-verifying if needed.

    The cached layout is ``<project_root>/.cache/assets/<name>/<md5>/<filename>``. A
    md5-matching cache entry is reused unless ``force=True``. The actual download lands
    in a ``.<filename>.part`` sibling and is atomically renamed once the digest passes,
    so an interrupted run never leaves a half-written file in the canonical location.
    """

    target_dir = _ASSET_ROOT / spec.name / spec.md5
    target = target_dir / spec.filename
    if target.exists() and not force:
        actual = _md5_of(target)
        if actual == spec.md5:
            return target
        target.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    staging = target_dir / f".{spec.filename}.part"
    if staging.exists():
        staging.unlink()
    try:
        with urlopen(spec.url) as resp, staging.open("wb") as out:
            shutil.copyfileobj(resp, out, length=_CHUNK_SIZE)
    except URLError as exc:
        if staging.exists():
            staging.unlink()
        raise AssetDownloadError(
            f"failed to download asset {spec.name!r} from {spec.url!r}: {exc}"
        ) from exc
    actual = _md5_of(staging)
    if actual != spec.md5:
        staging.unlink()
        raise AssetDownloadError(
            f"md5 mismatch for asset {spec.name!r} downloaded from {spec.url!r}: "
            f"expected {spec.md5}, got {actual}"
        )
    staging.replace(target)
    return target


def _md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
