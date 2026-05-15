"""Asset zoo registration + download helper tests.

Robots are registered as an import side-effect of :mod:`genelab.asset_zoo` (loaded by
``load_builtin_registries()``). These tests assume that import has already happened by
the time any test runs, then monkeypatch ``fetch_asset`` so the factories return without
touching the network. The :func:`fetch_asset` happy / failure paths are exercised
separately against a stdlib HTTP server bound to a free localhost port.
"""

import hashlib
import http.server
import io
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from genelab.asset_zoo import (
    AnymalCCfg,
    CartpoleCfg,
    FrankaPandaCfg,
    UnitreeG1Cfg,
    UnitreeGo1Cfg,
)
from genelab.entity import ArticulationCfg
from genelab.registry import ROBOTS, load_builtin_registries
from genelab.utils.download import (
    PROGRESS_CALLBACK,
    AssetDownloadError,
    AssetSpec,
    fetch_asset,
)

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
    # Menagerie panda.xml uses unprefixed joint names. The regex is anchored so it
    # does not substring-match ``finger_joint{1,2}`` through ``re.search`` and
    # collide with the ``panda_hand`` group.
    assert arm.target_names_expr == (r"^joint[1-7]$",)
    assert arm.stiffness == 400.0
    assert arm.damping == 80.0
    assert hand.target_names_expr == (r"finger_joint.*",)
    assert hand.stiffness == 1.0e4
    assert hand.damping == 200.0
    # Home keyframe qpos[3] (joint4) = -1.57079.
    assert cfg.default_joint_pos["joint4"] == pytest.approx(-1.57079)


def test_registry_examples() -> None:
    entry = ROBOTS.entry("cartpole")
    assert entry.description
    assert any("genelab info" in ex for ex in entry.examples)
    franka_entry = ROBOTS.entry("franka")
    assert franka_entry.cfg_type is ArticulationCfg


def test_unitree_g1_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FIXTURES_DIR / "cartpole.xml"  # any extant path; cfg only stores str
    monkeypatch.setattr("genelab.asset_zoo.unitree_g1.fetch_asset", lambda spec: fake_path)
    cfg = UnitreeG1Cfg()
    assert isinstance(cfg, ArticulationCfg)
    assert set(cfg.actuators) == {"5020", "7520_14", "7520_22", "4010", "waist", "ankle"}
    assert cfg.foot_link_names == ("left_ankle_roll_link", "right_ankle_roll_link")
    # 7520_22 group (hip roll + knee) carries the highest effort
    assert cfg.actuators["7520_22"].effort_limit == pytest.approx(139.0)
    # waist + ankle groups are 5020 motors in parallel — armature should double
    assert cfg.actuators["waist"].armature == pytest.approx(
        2 * cfg.actuators["5020"].armature  # type: ignore[operator]
    )


def test_unitree_go1_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FIXTURES_DIR / "cartpole.xml"
    monkeypatch.setattr("genelab.asset_zoo.unitree_go1.fetch_asset", lambda spec: fake_path)
    cfg = UnitreeGo1Cfg()
    assert set(cfg.actuators) == {"hip", "thigh", "calf"}
    assert cfg.foot_link_names == ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    assert cfg.actuators["calf"].effort_limit == pytest.approx(35.55)
    assert cfg.actuators["hip"].stiffness == 25.0


def test_anymal_c_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_path = FIXTURES_DIR / "cartpole.xml"
    monkeypatch.setattr("genelab.asset_zoo.anymal_c.fetch_asset", lambda spec: fake_path)
    cfg = AnymalCCfg()
    assert set(cfg.actuators) == {"legs"}
    legs = cfg.actuators["legs"]
    assert legs.stiffness == 80.0
    assert legs.damping == 2.0
    assert legs.effort_limit == 80.0
    assert cfg.foot_link_names == ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
    # Hind legs use mirrored HFE / KFE so the robot spawns in a stable stand pose.
    assert cfg.default_joint_pos[r"(LH|RH)_HFE"] == -0.4
    assert cfg.default_joint_pos[r"(LH|RH)_KFE"] == 0.8


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


def _build_test_tarball(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory .tar.gz with the given relative path → bytes entries."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_fetch_asset_archive(asset_root: Path, fake_http_server: tuple[str, _ServerState]) -> None:
    url, state = fake_http_server
    tarball = _build_test_tarball(
        {
            "dummy/dummy.xml": b"<mujoco model='dummy'><worldbody/></mujoco>",
            "dummy/meshes/cube.stl": b"binary-stl-stub",
        }
    )
    state.payload = tarball
    md5 = hashlib.md5(tarball).hexdigest()
    spec = AssetSpec(
        name="dummy-archive",
        url=url,
        md5=md5,
        filename="dummy.tar.gz",
        archive_member="dummy/dummy.xml",
    )
    entry = fetch_asset(spec)
    expected = asset_root / "dummy-archive" / md5 / "extracted" / "dummy" / "dummy.xml"
    assert entry == expected
    assert entry.read_bytes() == b"<mujoco model='dummy'><worldbody/></mujoco>"
    # Mesh dependency travels alongside so MJCF relative references resolve.
    assert (entry.parent / "meshes" / "cube.stl").read_bytes() == b"binary-stl-stub"
    # Second call serves from cache (no extra HTTP hit, no re-extraction).
    again = fetch_asset(spec)
    assert again == entry
    assert state.hit_count == 1


def test_fetch_asset_progress_callback_via_argument(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = b"x" * (3 * (1 << 16) + 137)  # span >3 chunks so we see >1 callback
    md5 = hashlib.md5(state.payload).hexdigest()
    spec = AssetSpec(name="progress", url=url, md5=md5, filename="progress.bin")

    events: list[tuple[int, int | None, str]] = []

    def _cb(done: int, total: int | None, *, name: str) -> None:
        events.append((done, total, name))

    fetch_asset(spec, progress=_cb)

    assert events, "expected at least one progress callback"
    # First call carries done=0 so a UI can initialise its bar.
    assert events[0][0] == 0
    # Final cumulative count equals the payload size.
    assert events[-1][0] == len(state.payload)
    # Content-Length is propagated as ``total``.
    assert all(total == len(state.payload) for _, total, _ in events)
    # ``name`` is the spec's name verbatim — keeps multi-asset routing trivial.
    assert {ev[2] for ev in events} == {"progress"}


def test_fetch_asset_progress_callback_via_contextvar(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = b"y" * 4096
    md5 = hashlib.md5(state.payload).hexdigest()
    spec = AssetSpec(name="ctxvar", url=url, md5=md5, filename="ctxvar.bin")

    events: list[tuple[int, int | None]] = []

    def _cb(done: int, total: int | None, *, name: str) -> None:
        del name
        events.append((done, total))

    token = PROGRESS_CALLBACK.set(_cb)
    try:
        fetch_asset(spec)
    finally:
        PROGRESS_CALLBACK.reset(token)

    assert events
    assert events[-1] == (len(state.payload), len(state.payload))


def test_fetch_asset_explicit_progress_overrides_contextvar(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = b"z" * 2048
    md5 = hashlib.md5(state.payload).hexdigest()
    spec = AssetSpec(name="prio", url=url, md5=md5, filename="prio.bin")

    via_arg: list[int] = []
    via_ctx: list[int] = []

    def _arg_cb(done: int, total: int | None, *, name: str) -> None:
        del total, name
        via_arg.append(done)

    def _ctx_cb(done: int, total: int | None, *, name: str) -> None:
        del total, name
        via_ctx.append(done)

    token = PROGRESS_CALLBACK.set(_ctx_cb)
    try:
        fetch_asset(spec, progress=_arg_cb)
    finally:
        PROGRESS_CALLBACK.reset(token)

    # Explicit kwarg wins; ContextVar callback should not have been invoked.
    assert via_arg and via_arg[-1] == len(state.payload)
    assert via_ctx == []


def test_fetch_asset_archive_md5_mismatch(
    asset_root: Path, fake_http_server: tuple[str, _ServerState]
) -> None:
    url, state = fake_http_server
    state.payload = _build_test_tarball({"dummy/dummy.xml": b"<mujoco/>"})
    spec = AssetSpec(
        name="bad-archive",
        url=url,
        md5="0" * 32,
        filename="dummy.tar.gz",
        archive_member="dummy/dummy.xml",
    )
    with pytest.raises(AssetDownloadError, match="md5 mismatch"):
        fetch_asset(spec)
    # No extracted dir should linger after the failed verification.
    cache_dir = asset_root / "bad-archive" / spec.md5
    assert not (cache_dir / "extracted").exists()
