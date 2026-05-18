# 快速开始

环境已经安装好、只想走最短命令路径时，用这一页。

## 1. 列出已注册内容

```bash
uv run genelab list robots
uv run genelab list envs
uv run genelab list tasks
```

如果缺少某个任务，显式加载对应扩展：

```bash
PYTHONPATH=examples/inverted_pendulum/src \
  uv run genelab --import genelab_inverted_pendulum.tasks list tasks
```

## 2. 查看任务

```bash
uv run genelab info GeneLab-Inverted-Pendulum-v0
```

从打印出的配置树复制 override 路径，不要手猜。

## 3. Play

```bash
uv run genelab play GeneLab-Inverted-Pendulum-v0 --steps 64
uv run genelab play GeneLab-Inverted-Pendulum-v0 --vis --steps 500
uv run genelab play GeneLab-Inverted-Pendulum-v0 --agent random --steps 128
```

## 4. Train

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 64 \
  --max_iterations 2
```

更长训练：

```bash
uv run genelab train GeneLab-Inverted-Pendulum-v0 \
  --num_envs 4096 \
  --max_iterations 300
```

## 5. 生成项目

```bash
uv run genelab project new my_robot_project
uv pip install -e my_robot_project
uv run genelab list tasks
```

## 另见

- [教程](../tutorial.md)
- [CLI 参考](../reference/cli.md)
- [运行 RL 实验](../best-practices/rl-experiments.md)
