# 数据录制与实时绘图

Recording 把选定运行时信号变成实时曲线、文件或视频，而不需要把输出代码嵌入环境循环。

## 心智模型

`RecordingCfg` 描述一个数据源和一个或多个输出 sink。scene 在 build 时把这些配置转换成 Genesis recorder，并在传感器存在后绑定到 live env。

```text
RecordingCfg(source, field, env_idx)
└── PyQtPlotCfg / MPLPlotCfg / MPLImagePlotCfg / NPZFileCfg / CSVFileCfg / VideoFileCfg
```

## 数据源

| Source | 含义 |
|---|---|
| 传感器名 | 读取 `env.sensors[name].data`，可选沿 `field` 取字段。 |
| 无参 callable | 直接调用。 |
| 带 `env` 参数的 callable | 用 live `ManagerBasedRlEnv` 调用。 |

`env_idx` 会压缩 batched tensor 的 env 维，用于单 env 绘图。源已经返回标量，或文件需要捕获完整 batch 时，使用 `env_idx=None`。

## 输出

| Output | 用途 |
|---|---|
| `PyQtPlotCfg` | 实时 PyQtGraph 折线图（时间序列）。 |
| `MPLPlotCfg` | 实时 Matplotlib 折线图（时间序列）。 |
| `MPLImagePlotCfg` | `CameraSensor` 源的实时 Matplotlib 图像窗口——`field="rgb"` 显示画面，`field="depth"` 显示深度图。 |
| `NPZFileCfg` | cleanup/reset 时写压缩数组。 |
| `CSVFileCfg` | 行式流式输出。 |
| `VideoFileCfg` | 相机帧写 `.mp4`。 |

折线图（`PyQtPlotCfg` / `MPLPlotCfg`）消费一维通道数据；`MPLImagePlotCfg` 消费二维帧，且要求相机源。所有绘图窗口都会自动检测显示器，无显示器时静默跳过。

## 继续阅读

- [传感器](sensors.md)
- [调试常见问题](../best-practices/debugging.md)
- [API 参考](../api/reference.md)
