# 材料

`genelab.materials` 把 Genesis 的材料、表面、纹理构造器封装为声明式的配置
dataclass。材料配置通过 `RigidObjectCfg.material` 挂到场景物体上；非刚体材料会
把物体变为可形变体或流体，并让 `InteractiveScene` 启用对应的物理求解器。

## 设计

每个 `*Cfg` 与一个 `gs.materials` / `gs.surfaces` / `gs.textures` 构造器逐字段
对应，沿用 Genesis 的参数名（`friction`、`rho`、`E`、`nu`、`sampler` …）。所有
字段默认 `None`，表示"用 Genesis 的默认值"：`build(gs)` 只转发被显式设置的字段，
因此未改动的配置产生的对象与不带参数调用 Genesis 构造器完全一致。

模块在加载时不 `import genesis` —— `build(gs)` 把 `gs` 模块作为参数接收 —— 所以
`import genelab.configs` 不会牵入 torch。

## 材料目录

| 族 | 配置类 | 求解器 |
|---|---|---|
| Rigid | `RigidMaterialCfg`、`KinematicMaterialCfg` | rigid |
| Tool | `ToolMaterialCfg` | tool |
| MPM | `MpmElasticCfg`、`MpmElastoPlasticCfg`、`MpmLiquidCfg`、`MpmSandCfg`、`MpmSnowCfg`、`MpmMuscleCfg` | mpm |
| FEM | `FemElasticCfg`、`FemClothCfg`、`FemMuscleCfg` | fem |
| PBD | `PbdElasticCfg`、`PbdClothCfg`、`PbdLiquidCfg`、`PbdParticleCfg` | pbd |
| SPH | `SphLiquidCfg` | sph |
| SF | `SfSmokeCfg` | sf |
| Composite | `HybridMaterialCfg` | 各组成部分的并集 |

`HybridMaterialCfg` 把刚性骨架（`material_rigid`）与柔性外皮（`material_soft`）
耦合在一起；两者本身也是材料配置，会被递归构建。

## 挂载材料

```python
from genelab.configs import InteractiveSceneCfg
from genelab.entity import RigidObjectCfg
from genelab.materials import MpmElasticCfg, MetalCfg

scene = InteractiveSceneCfg(
    entities={
        "blob": RigidObjectCfg(
            morph="box",
            size=(0.1, 0.1, 0.1),
            material=MpmElasticCfg(E=3e5, nu=0.3),
            surface=MetalCfg(metal_type="gold"),
        )
    }
)
```

`material` 优先于 `RigidObjectCfg` 上的 `friction` / `density` 快捷字段；不设置
`material` 时这些快捷字段的行为与以前完全一致。`surface` 与物理正交 —— 它只控制
渲染外观（`gs.surfaces.Glass` / `Metal` / `Plastic` / `Emission` / `BSDF`）。

## 启用求解器

可形变体或流体若未在构造 `gs.Scene` 时配上对应的求解器选项，就是惰性的。
`InteractiveScene` 读取每个实体材料的 `required_solvers()`，仅启用正在使用的
求解器（用 Genesis 默认值），因此纯刚体场景保持不变。

域边界与迭代次数通过 `InteractiveSceneCfg.solvers` 上的 `SolverOptionsCfg` 调节：

```python
from genelab.materials import SolverOptionsCfg, MpmOptionsCfg

scene = InteractiveSceneCfg(
    entities={"blob": RigidObjectCfg(morph="box", material=MpmElasticCfg())},
    solvers=SolverOptionsCfg(mpm=MpmOptionsCfg(lower_bound=(-1, -1, 0), upper_bound=(1, 1, 1))),
)
```

在 `solvers` 下设置了某个求解器字段时，即使没有材料需要它也会被启用，这对在运行
时添加粒子的 emitter 很有用。

!!! warning "MPM / SPH / PBD 的域是有界的"
    MPM、SPH、PBD 在一个有限的盒子内仿真。Genesis 的默认域很小（MPM 大致是一个
    单位立方体）；离开域的物体会被截断。在对应的 options 配置上设置
    `lower_bound` / `upper_bound` 以适配场景。

## CLI 覆盖

材料与求解器字段都是普通的点路径覆盖项，由 `genelab info` 展示，并可在 `play` /
`train` 上设置：

```bash
genelab play TASK --env.scene.entities.blob.material.E 500000
genelab play TASK --env.scene.solvers.mpm.lower_bound -1,-1,0
```

## See also

- [场景与实体](scene.md)
- [配置系统](configs.md)
