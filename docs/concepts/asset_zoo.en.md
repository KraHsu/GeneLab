# Asset zoo

`genelab.asset_zoo` ships curated robot configurations as part of the core package, so
`from genelab.lab import CartpoleCfg` works out of the box. Each entry pairs a
declarative `AssetSpec` (URL + md5 + filename, optionally + `archive_member`) with a
lazy factory that returns a fresh `ArticulationCfg`. The factories trigger an
md5-verified download only when invoked, so read-only commands like
`genelab list robots` never touch the network.

## Why a curated zoo

Isaac Lab ships 40+ ready-to-use robot configurations and MjLab ships three; both burn a
significant fraction of their distribution size on bundled MJCF / URDF files. GeneLab
keeps the runtime install lean by separating the **configuration code** (in-tree, in
this package) from the **asset blobs** (off-tree, in the
[`KraHsu/genelab-assets`](https://github.com/KraHsu/genelab-assets) repository). The
first call to a robot factory downloads its MJCF once into the project-local cache and
every subsequent call resolves to the cached path.

## Lifecycle of an asset

1. A robot module (e.g. `asset_zoo/cartpole.py`) declares a module-level constant
   `AssetSpec(name=..., url=..., md5=..., filename=...)`.
2. Import of `genelab.asset_zoo` calls `register_robot(name, factory, ...)` as a side
   effect. The factory is the function `CartpoleCfg`, not the eager result.
3. `genelab list robots` enumerates registered names without invoking factories.
4. `ROBOTS.get("cartpole")` invokes the factory, which calls `fetch_asset(spec)`. On a
   cold cache the file lands in `<project_root>/.cache/assets/cartpole/<md5>/cartpole.xml`
   after md5 verification. On warm cache the path resolves immediately.
5. The factory returns a fresh `ArticulationCfg` so downstream callers can mutate
   `init_pos`, `default_joint_pos`, etc. without affecting siblings.

## Five built-in robots

| Name | Cfg factory | DoF | Actuator groups |
|---|---|---|---|
| `cartpole` | `CartpoleCfg()` | 2 | `cart` (active PD), `pole` (passive zero-gain) |
| `franka` | `FrankaPandaCfg()` | 9 | `panda_arm` (high-PD, k=400), `panda_hand` (stiff grasp, k=1e4) |
| `g1` | `UnitreeG1Cfg()` | 29 | `5020` / `7520_14` / `7520_22` / `4010` / `waist` / `ankle` — six DCMotor families |
| `go1` | `UnitreeGo1Cfg()` | 12 | `hip` / `thigh` / `calf` (ImplicitPD, k=25) |
| `anymal-c` | `AnymalCCfg()` | 12 | `legs` (single ImplicitPD group, k=80) |

Use `genelab info <name>` to print the full override-path tree once registered.
Cartpole mirrors `examples/inverted_pendulum`'s gains; G1 mirrors
`examples/unitree/.../g1/constants.py`; Franka, Go1, and Anymal C follow Isaac Lab's
published defaults so cross-stack experiments stay comparable. The four Menagerie-sourced
entries (`franka`, `g1`, `go1`, `anymal-c`) ship as `.tar.gz` archives so meshes and
textures travel with the MJCF.

## Cache layout and md5 verification

Single-file mode (cartpole):

```
.cache/assets/<name>/<md5>/<filename>
```

Archive mode (franka, g1, go1, anymal-c):

```
.cache/assets/<name>/<md5>/extracted/<archive_member>
```

`fetch_asset` writes the download to a `.<filename>.part` sibling and renames atomically
once the digest matches. Archive blobs go through a second atomic step: the tar is
expanded into a `.extracting/` sibling using `tarfile`'s `data` filter (rejects symlinks
and parent-directory escapes) and renamed to `extracted/` only on success. A mismatch
raises `AssetDownloadError` carrying both expected and actual digests; cached files
keyed by a stale md5 are auto-pruned at the next call with the corrected hash, since
the path itself moves.

!!! warning "Update the md5 every time the asset moves"
    The cache key is `(name, md5)`; bumping the md5 invalidates old copies cleanly.
    Forgetting to bump it after re-uploading a fixed MJCF can leave clients reading a
    stale local file forever.

## Adding a new robot config

Single-file MJCF (no external mesh dependencies):

```python
_MJCF = AssetSpec(
    name="my-robot",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/my-robot/my-robot.xml",
    md5="<32 hex chars>",
    filename="my-robot.xml",
)
```

Menagerie-style folder with meshes / textures (pack as `.tar.gz`, name the entry MJCF):

```python
_MJCF = AssetSpec(
    name="my-robot",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/my_robot/my_robot.tar.gz",
    md5="<32 hex chars>",
    filename="my_robot.tar.gz",
    archive_member="my_robot/my_robot.xml",
)
```

Either way the factory is the same:

```python
def MyRobotCfg() -> ArticulationCfg:
    return ArticulationCfg(
        mjcf_path=str(fetch_asset(_MJCF)),
        actuators={"all": ImplicitPDActuatorCfg(target_names_expr=(r".*",), stiffness=100.0, damping=5.0)},
    )

register_robot("my-robot", MyRobotCfg, description="...", cfg_type=ArticulationCfg)
```

Then add the module to `asset_zoo/__init__.py` so import side-effects run during
`load_builtin_registries()`. Downstream projects that prefer to keep robots out-of-tree
should use the same `AssetSpec` + `register_robot` pattern from their own extension
package — the helpers are public.

## Failure modes worth knowing

* **Network unavailable** — `AssetDownloadError` wraps `URLError`; the staging file is
  removed so a retry starts clean.
* **md5 mismatch** — raises `AssetDownloadError` with both digests; the staging file is
  removed before the exception propagates.
* **Stale cache after upstream re-upload** — the path is keyed by md5, so updating the
  spec naturally points at a fresh subdirectory; old digests linger on disk until the
  user deletes `.cache/assets/<name>/` manually.
* **Factory called before `load_builtin_registries()`** — `ROBOTS.get("cartpole")`
  raises `KeyError`; the CLI calls the loader at startup so this only bites direct API
  use. Import `genelab.asset_zoo` explicitly to register without going through the CLI.

## See also

- [Registry](registry.md)
- [Actuators](actuators.md)
- [Extensions](extensions.md)
