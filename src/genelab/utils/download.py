"""Asset download helper for the GeneLab asset zoo.

Robot configurations registered under :mod:`genelab.asset_zoo` declare their MJCF / URDF
sources as :class:`AssetSpec` instances. :func:`fetch_asset` downloads the file once,
verifies its md5, and caches it under ``<project_root>/.cache/assets/<name>/<md5>/`` so
repeated CLI invocations stay offline. Failures (network error, md5 mismatch, malformed
URL) all surface as :class:`AssetDownloadError` with the expected vs actual digest
included in the message — easier to copy-paste into a follow-up commit that updates the
hash after a new upload.

Two delivery modes:

* **single-file** (``archive_member`` unset) — the URL points to a standalone MJCF blob;
  the local path is just the cached file.
* **archive** (``archive_member`` set) — the URL points to a ``.tar.gz`` produced from a
  full Menagerie-style folder (MJCF + meshes + textures). After md5 verification the
  archive is extracted into ``<md5>/extracted/`` using ``tarfile``'s ``data`` filter
  (rejects symlinks, absolute paths, and parent-directory escapes), and
  :func:`fetch_asset` returns the path to the entry MJCF inside the extracted tree.

Stdlib only: no ``requests`` / ``httpx`` dependency, no concurrent / resumable downloads.
"""

import hashlib
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from genelab.cache import CACHE_DIR

_ASSET_ROOT = CACHE_DIR / "assets"
_CHUNK_SIZE = 1 << 16  # 64 KiB
_EXTRACT_DIR = "extracted"
_STAGING_EXTRACT_DIR = ".extracting"


class AssetDownloadError(RuntimeError):
    """Raised when an asset cannot be retrieved or its md5 does not match the spec."""


@dataclass(frozen=True)
class AssetSpec:
    """Declarative descriptor for a downloadable asset.

    Robot cfg modules build a module-level constant such as
    ``MJCF = AssetSpec(name="cartpole", url="https://.../cartpole.xml", md5=..., filename=...)``
    and then call :func:`fetch_asset` lazily inside the registered factory so commands
    like ``genelab list robots`` never touch the network.

    Set ``archive_member`` (relative path inside the archive) to switch from single-file
    to ``.tar.gz`` archive mode; ``md5`` then verifies the archive blob as a whole.
    """

    name: str
    url: str
    md5: str
    filename: str
    archive_member: str | None = None


def fetch_asset(spec: AssetSpec, *, force: bool = False) -> Path:
    """Return the local path to ``spec``, downloading and md5-verifying if needed.

    Single-file mode caches ``<project_root>/.cache/assets/<name>/<md5>/<filename>``.
    Archive mode keeps the same parent directory layout but extracts under
    ``<md5>/extracted/`` and returns ``<md5>/extracted/<archive_member>``. A
    md5-matching cache entry is reused unless ``force=True``. The actual download lands
    in a ``.<filename>.part`` sibling and is atomically renamed once the digest passes,
    so an interrupted run never leaves a half-written file in the canonical location.
    """

    target_dir = _ASSET_ROOT / spec.name / spec.md5
    if spec.archive_member is None:
        return _fetch_single_file(spec, target_dir, force=force)
    return _fetch_archive(spec, target_dir, force=force)


def _fetch_single_file(spec: AssetSpec, target_dir: Path, *, force: bool) -> Path:
    target = target_dir / spec.filename
    if target.exists() and not force:
        actual = _md5_of(target)
        if actual == spec.md5:
            return target
        target.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    staging = target_dir / f".{spec.filename}.part"
    _download_to(spec, staging)
    actual = _md5_of(staging)
    if actual != spec.md5:
        staging.unlink()
        raise AssetDownloadError(
            f"md5 mismatch for asset {spec.name!r} downloaded from {spec.url!r}: "
            f"expected {spec.md5}, got {actual}"
        )
    staging.replace(target)
    return target


def _fetch_archive(spec: AssetSpec, target_dir: Path, *, force: bool) -> Path:
    assert spec.archive_member is not None  # narrows type for pyright
    extracted_dir = target_dir / _EXTRACT_DIR
    entry = extracted_dir / spec.archive_member
    if entry.exists() and not force:
        return entry
    target_dir.mkdir(parents=True, exist_ok=True)
    staging_archive = target_dir / f".{spec.filename}.part"
    _download_to(spec, staging_archive)
    actual = _md5_of(staging_archive)
    if actual != spec.md5:
        staging_archive.unlink()
        raise AssetDownloadError(
            f"md5 mismatch for asset {spec.name!r} downloaded from {spec.url!r}: "
            f"expected {spec.md5}, got {actual}"
        )
    staging_extract = target_dir / _STAGING_EXTRACT_DIR
    if staging_extract.exists():
        shutil.rmtree(staging_extract)
    staging_extract.mkdir()
    try:
        with tarfile.open(staging_archive, mode="r:gz") as tar:
            tar.extractall(path=staging_extract, filter="data")
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(staging_extract, ignore_errors=True)
        staging_archive.unlink(missing_ok=True)
        raise AssetDownloadError(
            f"failed to extract archive {spec.name!r} from {spec.url!r}: {exc}"
        ) from exc
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    staging_extract.replace(extracted_dir)
    staging_archive.unlink(missing_ok=True)
    if not entry.exists():
        raise AssetDownloadError(
            f"archive_member {spec.archive_member!r} not found inside asset "
            f"{spec.name!r} downloaded from {spec.url!r}"
        )
    return entry


def _download_to(spec: AssetSpec, staging: Path) -> None:
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


def _md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
