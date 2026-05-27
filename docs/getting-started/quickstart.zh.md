# 快速开始

环境已经安装好、只想走最短命令路径时，用这一页。命令统一写作裸 `genelab`，前提是已激活
venv（`source .venv/bin/activate`）；原因以及 `uv run --no-sync` 替代方案见
[安装 → 运行命令](installation.md#_4)。

## 1. 列出已注册内容

```bash
genelab list robots
genelab list envs
genelab list tasks
```

如果缺少某个任务，显式加载对应扩展：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 2. 查看任务

```bash
genelab info GeneLab-Inverted-Pendulum-v0
```

从打印出的配置树复制 override 路径，不要盲猜。

## 3. Play

```bash
genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
genelab play GeneLab-Inverted-Pendulum-v0 --agent random --steps 128
```

## 4. Train

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

更长训练：

```bash
genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 4096 \
  --max_iterations 300
```

## 5. 生成项目

```bash
genelab project new my_robot_project
uv pip install -e my_robot_project
genelab list tasks
```

## 另见

- [教程](../tutorial.md)
- [CLI 参考](../reference/cli.md)
- [运行 RL 实验](../best-practices/rl-experiments.md)
