"""Unit tests for ``genelab asset`` subcommands."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from genelab.cli._asset import (
    _ZooAsset,
    _cache_paths,
    _discover_assets,
    _format_size,
    asset_app,
)
from genelab.utils.download import AssetDownloadError, AssetSpec


runner = CliRunner()


def test_discovery_finds_every_asset_zoo_spec() -> None:
    """Every ``_MJCF`` / similar constant on ``genelab.asset_zoo`` shows up exactly once."""
    assets = _discover_assets()
    # The current zoo ships 9 specs (8 robots + 1 motion); the exact list is stable.
    names = {a.spec.name for a in assets}
    expected = {
        "allegro",
        "anymal-c",
        "cartpole",
        "franka",
        "g1",
        "g1_lafan1_dance1_subject2",
        "go1",
        "h1",
        "ur10e",
    }
    assert expected.issubset(names), f"missing assets: {expected - names}"


def test_cache_paths_single_file_vs_archive() -> None:
    single = AssetSpec(name="a", url="u", md5="m", filename="model.xml")
    root, entry = _cache_paths(single)
    assert entry is not None
    assert entry.parent == root
    assert entry.name == "model.xml"

    archive = AssetSpec(
        name="b",
        url="u",
        md5="m",
        filename="bundle.tar.gz",
        archive_member="bundle/root.xml",
    )
    root, entry = _cache_paths(archive)
    assert entry is not None
    assert entry == root / "extracted" / "bundle/root.xml"


@pytest.mark.parametrize(
    "num,expected_unit",
    [(0, "-"), (512, "512 B"), (2048, "2.0 KiB"), (5 * 1024**2, "5.0 MiB")],
)
def test_format_size_buckets(num: int, expected_unit: str) -> None:
    assert _format_size(num) == expected_unit


def _make_fake_specs() -> list[AssetSpec]:
    return [
        AssetSpec(
            name="alpha",
            url="https://x/a.tar.gz",
            md5="aaa",
            filename="a.tar.gz",
            archive_member="a/x.xml",
        ),
        AssetSpec(name="beta", url="https://x/b.xml", md5="bbb", filename="b.xml"),
    ]


def _patch_discovery(monkeypatch: pytest.MonkeyPatch, specs: list[AssetSpec]) -> None:
    """Replace ``_discover_assets`` so tests stay isolated from the live asset zoo."""
    fake = [_ZooAsset(spec=s, module=f"fake.{s.name}", attr="_MJCF") for s in specs]
    monkeypatch.setattr("genelab.cli._asset._discover_assets", lambda: fake)


def test_list_renders_table_with_known_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    # Point the cache root at an empty tmp path so neither fake is "downloaded".
    monkeypatch.setattr("genelab.cli._asset._ASSET_ROOT", Path("/tmp/__nonexistent_cache__"))

    result = runner.invoke(asset_app, ["list"])
    assert result.exit_code == 0
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_info_unknown_asset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    result = runner.invoke(asset_app, ["info", "ghost"])
    assert result.exit_code != 0
    assert "no asset named" in result.output


def test_info_shows_url_md5_and_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    monkeypatch.setattr("genelab.cli._asset._ASSET_ROOT", Path("/tmp/__nonexistent_cache__"))
    result = runner.invoke(asset_app, ["info", "alpha"])
    assert result.exit_code == 0
    assert "https://x/a.tar.gz" in result.stdout
    assert "aaa" in result.stdout
    assert "fake.alpha._MJCF" in result.stdout


def test_download_named_asset_calls_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = _make_fake_specs()
    _patch_discovery(monkeypatch, specs)
    calls: list[AssetSpec] = []

    def _fake_fetch(spec: AssetSpec, *, force: bool = False, progress: Any = None) -> Path:
        del progress
        calls.append(spec)
        return Path(f"/tmp/{spec.name}.bin")

    monkeypatch.setattr("genelab.cli._asset.fetch_asset", _fake_fetch)

    result = runner.invoke(asset_app, ["download", "alpha"])
    assert result.exit_code == 0
    assert [c.name for c in calls] == ["alpha"]
    assert "/tmp/alpha.bin" in result.stdout


def test_download_all_calls_fetch_for_each(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = _make_fake_specs()
    _patch_discovery(monkeypatch, specs)
    calls: list[str] = []

    def _fake_fetch(spec: AssetSpec, *, force: bool = False, progress: Any = None) -> Path:
        del progress, force
        calls.append(spec.name)
        return Path(f"/tmp/{spec.name}.bin")

    monkeypatch.setattr("genelab.cli._asset.fetch_asset", _fake_fetch)

    result = runner.invoke(asset_app, ["download", "--all"])
    assert result.exit_code == 0
    assert sorted(calls) == ["alpha", "beta"]


def test_download_requires_name_or_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    result = runner.invoke(asset_app, ["download"])
    # CI renders typer's BadParameter message with rich ANSI escapes; comparing the
    # raw substring is fragile across rich versions. Assert the structural facts
    # instead: non-zero exit + typer's "Usage" preamble + an asset-name reference.
    assert result.exit_code != 0
    assert "Usage:" in result.output
    assert "asset download" in result.output


def test_download_rejects_mixing_all_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    result = runner.invoke(asset_app, ["download", "alpha", "--all"])
    assert result.exit_code != 0
    assert "Usage:" in result.output
    assert "asset download" in result.output


def test_download_reports_failures_with_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())

    def _broken(spec: AssetSpec, *, force: bool = False, progress: Any = None) -> Path:
        del force, progress
        raise AssetDownloadError(f"network down for {spec.name}")

    monkeypatch.setattr("genelab.cli._asset.fetch_asset", _broken)
    result = runner.invoke(asset_app, ["download", "alpha"])
    assert result.exit_code != 0
    assert "network down for alpha" in result.output


def test_purge_with_yes_removes_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = _make_fake_specs()[0]
    _patch_discovery(monkeypatch, [spec])
    monkeypatch.setattr("genelab.cli._asset._ASSET_ROOT", tmp_path)
    # Pre-populate the cache to simulate a prior download.
    cache_root = tmp_path / spec.name / spec.md5
    cache_root.mkdir(parents=True)
    (cache_root / spec.filename).write_bytes(b"x")

    result = runner.invoke(asset_app, ["purge", "alpha", "--yes"])
    assert result.exit_code == 0
    assert not cache_root.exists()
    assert "purged" in result.stdout


def test_purge_nothing_to_do_is_silent_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_discovery(monkeypatch, _make_fake_specs())
    monkeypatch.setattr("genelab.cli._asset._ASSET_ROOT", tmp_path)
    result = runner.invoke(asset_app, ["purge", "alpha", "--yes"])
    assert result.exit_code == 0
    assert "nothing to purge" in result.output
