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

Progress reporting hook
-----------------------

Callers that want a UI (e.g. the CLI) can install a :class:`ProgressCallback` to receive
byte-level updates during the network read loop. Pass ``progress=...`` explicitly to
:func:`fetch_asset`, or set the module-level :data:`PROGRESS_CALLBACK` ContextVar so
asset-zoo factories called indirectly (during cfg construction) pick it up without each
factory threading a callback through. The utils package itself stays free of UI
dependencies — only the Protocol type is defined here.

Stdlib only: no ``requests`` / ``httpx`` dependency, no concurrent / resumable downloads.
"""

import hashlib
import shutil
import tarfile
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen

from genelab.utils.paths import CACHE_DIR

_ASSET_ROOT = CACHE_DIR / "assets"
_CHUNK_SIZE = 1 << 16  # 64 KiB
_EXTRACT_DIR = "extracted"
_STAGING_EXTRACT_DIR = ".extracting"


class ProgressCallback(Protocol):
    """Receives cumulative byte counts during :func:`fetch_asset` network I/O.

    ``done`` is the number of bytes streamed so far. ``total`` is the
    ``Content-Length`` reported by the server, or ``None`` when unknown (chunked
    transfer encoding, missing header). ``name`` is :attr:`AssetSpec.name` so callers
    rendering multiple concurrent fetches can route updates to the right widget.

    A new callback is created per-asset at the start of the download. Implementations
    may rely on the first call carrying ``done == 0`` (suitable for initialising a
    progress bar) and the last call carrying ``done == total`` when ``total`` is known.
    """

    def __call__(self, done: int, total: int | None, *, name: str) -> None: ...


PROGRESS_CALLBACK: ContextVar[ProgressCallback | None] = ContextVar(
    "genelab_fetch_asset_progress", default=None
)


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


def fetch_asset(
    spec: AssetSpec,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Return the local path to ``spec``, downloading and md5-verifying if needed.

    Single-file mode caches ``<project_root>/.cache/assets/<name>/<md5>/<filename>``.
    Archive mode keeps the same parent directory layout but extracts under
    ``<md5>/extracted/`` and returns ``<md5>/extracted/<archive_member>``. A
    md5-matching cache entry is reused unless ``force=True``. The actual download lands
    in a ``.<filename>.part`` sibling and is atomically renamed once the digest passes,
    so an interrupted run never leaves a half-written file in the canonical location.

    ``progress`` overrides the :data:`PROGRESS_CALLBACK` ContextVar for this call.
    """

    cb = progress if progress is not None else PROGRESS_CALLBACK.get()
    target_dir = _ASSET_ROOT / spec.name / spec.md5
    if spec.archive_member is None:
        return _fetch_single_file(spec, target_dir, force=force, progress=cb)
    return _fetch_archive(spec, target_dir, force=force, progress=cb)


def _fetch_single_file(
    spec: AssetSpec, target_dir: Path, *, force: bool, progress: ProgressCallback | None
) -> Path:
    target = target_dir / spec.filename
    if target.exists() and not force:
        actual = _md5_of(target)
        if actual == spec.md5:
            return target
        target.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    staging = target_dir / f".{spec.filename}.part"
    _download_to(spec, staging, progress=progress)
    actual = _md5_of(staging)
    if actual != spec.md5:
        staging.unlink()
        raise AssetDownloadError(
            f"md5 mismatch for asset {spec.name!r} downloaded from {spec.url!r}: "
            f"expected {spec.md5}, got {actual}"
        )
    staging.replace(target)
    return target


def _fetch_archive(
    spec: AssetSpec, target_dir: Path, *, force: bool, progress: ProgressCallback | None
) -> Path:
    assert spec.archive_member is not None  # narrows type for pyright
    extracted_dir = target_dir / _EXTRACT_DIR
    entry = extracted_dir / spec.archive_member
    if entry.exists() and not force:
        return entry
    target_dir.mkdir(parents=True, exist_ok=True)
    staging_archive = target_dir / f".{spec.filename}.part"
    _download_to(spec, staging_archive, progress=progress)
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


def _download_to(spec: AssetSpec, staging: Path, *, progress: ProgressCallback | None) -> None:
    if staging.exists():
        staging.unlink()
    try:
        with urlopen(spec.url) as resp, staging.open("wb") as out:
            total = _content_length(resp)
            done = 0
            if progress is not None:
                progress(0, total, name=spec.name)
            while True:
                chunk = resp.read(_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total, name=spec.name)
    except URLError as exc:
        if staging.exists():
            staging.unlink()
        raise AssetDownloadError(
            f"failed to download asset {spec.name!r} from {spec.url!r}: {exc}"
        ) from exc


def _content_length(resp: object) -> int | None:
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
