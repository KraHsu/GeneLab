# 资产库

`genelab.asset_zoo` 把精选的机器人配置作为核心包的预置扩展一起发布（不属于 `genelab.lab` 公开门面），所以 `from genelab.asset_zoo import CartpoleCfg` 开箱即用。每个条目把一个声明式的 `AssetSpec`（URL + md5 + 文件名，可选 `archive_member`）和一个返回新 `ArticulationCfg` 的 lazy factory 配对。factory 仅在被调用时才触发带 md5 校验的下载，所以像 `genelab list robots` 这种只读命令不会触网。

## 为什么需要预置库

Isaac Lab 自带 40+ 现成机器人配置，MjLab 自带 3 个；两者都把可观比例的发行体积花在内置 MJCF / URDF 文件上。GeneLab 把**配置代码**（在树内、本包内）和**资产二进制**（在树外，放在 [`KraHsu/genelab-assets`](https://github.com/KraHsu/genelab-assets) 仓库）分开，让运行时安装保持精简。第一次调用机器人 factory 会把 MJCF 下载一次到项目本地缓存，后续每次调用都解析到缓存路径。

## 资产生命周期

1. 机器人模块（例如 `asset_zoo/cartpole.py`）在模块级别声明一个常量 `AssetSpec(name=..., url=..., md5=..., filename=...)`。
2. 导入 `genelab.asset_zoo` 时副作用调用 `register_robot(name, factory, ...)`。factory 是函数 `CartpoleCfg`，不是立即求值的结果。
3. `genelab list robots` 枚举已注册名称而不调用 factory。
4. `ROBOTS.get("cartpole")` 调用 factory，factory 内调 `fetch_asset(spec)`。冷缓存时文件落到 `<project_root>/.cache/assets/cartpole/<md5>/cartpole.xml`，md5 校验通过后才生效。热缓存时路径立刻解析。
5. factory 返回一个新的 `ArticulationCfg`，下游可以随意改 `init_pos`、`default_joint_pos` 等字段而不影响兄弟实例。

## 自带 5 种机器人

| 名称 | Cfg factory | DoF | actuator 分组 |
|---|---|---|---|
| `cartpole` | `CartpoleCfg()` | 2 | `cart`（带 PD）、`pole`（被动零增益） |
| `franka` | `FrankaPandaCfg()` | 9 | `panda_arm`（高 PD，k=400）、`panda_hand`（强抓握，k=1e4） |
| `g1` | `UnitreeG1Cfg()` | 29 | `5020` / `7520_14` / `7520_22` / `4010` / `waist` / `ankle` —— 6 个 DCMotor 家族 |
| `go1` | `UnitreeGo1Cfg()` | 12 | `hip` / `thigh` / `calf`（ImplicitPD，k=25） |
| `anymal-c` | `AnymalCCfg()` | 12 | `legs`（单 ImplicitPD 组，k=80） |

注册完成后用 `genelab info <name>` 打印完整的 override 路径树。Cartpole 的增益与 `examples/inverted_pendulum` 保持一致；G1 与 `examples/unitree/.../g1/constants.py` 对齐；Franka、Go1 与 Anymal C 沿用 Isaac Lab 公开的默认值，方便跨栈对比实验。4 个 Menagerie 来源条目（`franka`、`g1`、`go1`、`anymal-c`）以 `.tar.gz` archive 形式发布，mesh 与纹理随 MJCF 一并发送。

## 缓存目录与 md5 校验

单文件模式（cartpole）：

```
.cache/assets/<name>/<md5>/<filename>
```

archive 模式（franka、g1、go1、anymal-c）：

```
.cache/assets/<name>/<md5>/extracted/<archive_member>
```

`fetch_asset` 先把下载内容写到 `.<filename>.part` 兄弟文件，digest 校验通过后再原子 rename。archive 模式多一步原子语义：tar 先解到 `.extracting/` 临时目录（使用 `tarfile` 的 `data` filter，拒绝软链接与父目录逃逸），成功后再 rename 为 `extracted/`。不匹配会抛 `AssetDownloadError`，错误信息携带期望与实际两份 digest；旧 md5 对应的缓存目录会在下次用新 hash 调用时被自然绕过，因为路径本身改变了。

!!! warning "资产每次替换都要更新 md5"
    缓存 key 是 `(name, md5)`；更新 md5 可以干净失效旧副本。如果重新上传了修复版 MJCF 但忘了改 md5，客户端会一直读到陈旧的本地文件。

## 增加新的 robot 配置

单文件 MJCF（无外部 mesh 依赖）：

```python
_MJCF = AssetSpec(
    name="my-robot",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/my-robot/my-robot.xml",
    md5="<32 hex chars>",
    filename="my-robot.xml",
)
```

Menagerie 风格带 mesh / texture 的目录（打成 `.tar.gz`，声明入口 MJCF）：

```python
_MJCF = AssetSpec(
    name="my-robot",
    url="https://raw.githubusercontent.com/KraHsu/genelab-assets/main/my_robot/my_robot.tar.gz",
    md5="<32 hex chars>",
    filename="my_robot.tar.gz",
    archive_member="my_robot/my_robot.xml",
)
```

两种模式共用同一个 factory：

```python
def MyRobotCfg() -> ArticulationCfg:
    return ArticulationCfg(
        mjcf_path=str(fetch_asset(_MJCF)),
        actuators={"all": ImplicitPDActuatorCfg(target_names_expr=(r".*",), stiffness=100.0, damping=5.0)},
    )

register_robot("my-robot", MyRobotCfg, description="...", cfg_type=ArticulationCfg)
```

随后把模块加进 `asset_zoo/__init__.py`，让 `load_bundled_asset_zoo()` 触发导入副作用。下游项目若想把机器人放在树外，可以在自己的扩展包里用同样的 `AssetSpec` + `register_robot` 模式 —— 这两个 helper 都是公开 API。

## 需要知道的失败模式

* **网络不可达** —— `AssetDownloadError` 包住 `URLError`；staging 文件已删除，重试从干净状态开始。
* **md5 不匹配** —— 抛 `AssetDownloadError`，错误带两份 digest；staging 文件在异常抛出前已删除。
* **上游重新上传后缓存陈旧** —— 路径按 md5 键控，更新 spec 自然指到新子目录；旧 digest 对应的目录留在磁盘上直到用户手动删除 `.cache/assets/<name>/`。
* **factory 在 `load_bundled_asset_zoo()` 之前被调** —— `ROBOTS.get("cartpole")` 抛 `KeyError`；CLI 启动时会调用 loader，所以这只有在直接用 API 时才会出现。需要绕过 CLI 时显式 `import genelab.asset_zoo` 即可注册。

## See also

- [Registry](registry.md)
- [Actuators](actuators.md)
- [Extensions](extensions.md)
