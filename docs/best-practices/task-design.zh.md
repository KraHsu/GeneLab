# 如何设计 GeneLab 任务

这份指南说明如何组织一个容易运行、检查、训练和扩展的 GeneLab task。

## 1. 先定 task id

写代码前先选一个稳定 id：

```text
<Project>-<Robot>-<Objective>-v0
```

示例：

- `GeneLab-Inverted-Pendulum-v0`
- `Genelab-Velocity-Flat-Unitree-G1-v0`
- `MyProject-PickPlace-v0`

`TaskCfg.name`、`TASKS` 注册、日志和 README 命令里都使用同一个 id。

## 2. 分开 `env` 与 `play_env`

`env` 放训练默认值，`play_env` 放人工检查默认值。

| 配置 | 典型默认值 |
|---|---|
| `env` | 多 env、viewer 关闭、课程和扰动开启、只保留训练传感器。 |
| `play_env` | 单 env、viewer 开启、鼠标/bridge 控制开启、随机化较少、可选实时曲线。 |

不要让其他人为了打开 viewer 记一串 override。`genelab play TASK --vis` 应该能直接用任务的
play 配置运行。

## 3. 用 factory 构造机器人

注册 robot factory，而不是注册已经构造好的对象：

```python
register_robot(
    "my-robot",
    get_my_robot_cfg,
    description="My robot.",
    cfg_type=MyRobotCfg,
)
```

factory 能保持导入轻量，也让 CLI 在不启动 Genesis 的情况下列出元信息。

## 4. 按意图命名 manager term

每个 manager 字典都使用描述性名称。这些名称会成为 CLI override 路径和日志 key。

```python
rewards_cfg = {
    "track_lin_vel": RewardTermCfg(...),
    "action_rate": RewardTermCfg(...),
    "feet_slip": RewardTermCfg(...),
}
```

避免 `r1`、`penalty2`、`tmp` 这类名字。好名字应该稳定到可以出现在论文、run config 或
dashboard 中。

## 5. 显式组织 observation

至少提供 `policy` 组。runner 需要特权信息时再加 `critic` 组。

```python
observations_cfg = {
    "policy": ObservationGroupCfg(terms=policy_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic_terms, enable_corruption=False),
}
```

corruption、scale、clip 设置放在 term cfg 中，这样 `genelab info TASK` 能直接展示。

## 6. 用小 rollout 验证

长训练前先跑：

```bash
uv run genelab info TASK_ID
uv run genelab play TASK_ID --steps 32
uv run genelab play TASK_ID --agent random --steps 64
uv run genelab train TASK_ID --num_envs 64 --max_iterations 2
```

这能在长任务前捕获注册、scene 构造、action 维度、observation shape、reward 和 runner 接线问题。

## 预期结果

一个设计良好的 task 应该有稳定 task id、清晰的 `TaskCfg`、适合 viewer 的 `play_env`、
可读的 manager term 名、可查的 override，以及能跑通完整路径的短 smoke 命令。
