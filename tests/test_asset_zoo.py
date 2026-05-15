"""Asset zoo registration + download helper tests.

Robots are registered as an import side-effect of :mod:`genelab.asset_zoo` (loaded by
``load_builtin_registries()``). These tests assume that import has already happened by
the time any test runs, then monkeypatch ``fetch_asset`` so the factories return without
touching the network. The :func:`fetch_asset` happy / failure paths are exercised
separately against a stdlib HTTP server bound to a free localhost port.
"""

import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from genelab.asset_zoo import CartpoleCfg, FrankaPandaCfg
from genelab.entity import ArticulationCfg
from genelab.registry import ROBOTS, load_builtin_registries
from genelab.utils.download import AssetDownloadError, AssetSpec, fetch_asset

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def ensure_builtin_registry() -> None:
    """Idempotent; import side-effect runs once per process."""
    load_builtin_registries()


def test_cartpole_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FIXTURES_DIR / "cartpole.xml"
    monkeypatch.setattr("genelab.asset_zoo.cartpole.fetch_asset", lambda spec: fake_path)
    cfg = ROBOTS.get("cartpole")
    assert isinstance(cfg, ArticulationCfg)
    assert cfg.mjcf_path == str(fake_path)
    assert set(cfg.actuators) == {"cart", "pole"}
    assert cfg.actuators["cart"].stiffness == 80.0
    assert cfg.actuators["pole"].stiffness == 0.0
    # Factory yields independent dataclasses so callers can mutate freely.
    other = CartpoleCfg()
    assert other is not cfg
    assert other.actuators is not cfg.actuators


def test_franka_actuator_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FIXTURES_DIR / "franka.xml"
    monkeypatch.setattr("genelab.asset_zoo.franka.fetch_asset", lambda spec: fake_path)
    cfg = FrankaPandaCfg()
    arm = cfg.actuators["panda_arm"]
    hand = cfg.actuators["panda_hand"]
    assert arm.target_names_expr == (r"panda_joint[1-7]",)
    assert arm.stiffness == 400.0
    assert arm.damping == 80.0
    assert hand.target_names_expr == (r"panda_finger_joint.*",)
    assert hand.stiffness == 1.0e4
    assert hand.damping == 200.0
    assert cfg.default_joint_pos["panda_joint4"] == pytest.approx(-2.356)


def test_registry_examples() -> None:
    entry = ROBOTS.entry("cartpole")
    assert entry.description
    assert any("genelab info" in ex for ex in entry.examples)
    franka_entry = ROBOTS.entry("franka")
    assert franka_entry.cfg_type is ArticulationCfg


@pytest.fixture
def asset_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    monkeypatch.setattr("genelab.utils.download._ASSET_ROOT", root)
    return root


class _ServerState:
    def __init__(self) -> None:
        self.payload: bytes = b""
        self.hit_count: int = 0


@pytest.fixture
def fake_http_server() -> Iterator[tuple[str, _ServerState]]:
    state = _ServerState()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802  (http.server API)
            state.hit_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(state.payload)))
            self.end_headers()
            self.wfile.write(state.payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/asset.bin", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_fetch_asset_md5_mismatch(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = b"genelab-fixture-bytes"
    bad_spec = AssetSpec(name="bad", url=url, md5="0" * 32, filename="bad.bin")
    with pytest.raises(AssetDownloadError, match="md5 mismatch"):
        fetch_asset(bad_spec)
    # Staging file should be cleaned up so the cache directory is empty.
    cached = list((asset_root / "bad" / bad_spec.md5).iterdir())
    assert cached == []


def test_fetch_asset_cache_hit(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = b"genelab-cache-bytes"
    correct_md5 = hashlib.md5(state.payload).hexdigest()
    spec = AssetSpec(name="cache", url=url, md5=correct_md5, filename="cache.bin")
    first = fetch_asset(spec)
    second = fetch_asset(spec)
    assert first == second == asset_root / "cache" / correct_md5 / "cache.bin"
    assert first.read_bytes() == state.payload
    assert state.hit_count == 1
