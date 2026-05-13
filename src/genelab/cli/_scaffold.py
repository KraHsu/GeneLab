"""Project scaffolding helpers for ``genelab project new``."""

import os
import re
from pathlib import Path
from textwrap import dedent

from genelab.cli._render import render_project_created


def create_project_skeleton(
    name: str,
    *,
    path: Path = Path("."),
    package: str | None = None,
    task_id: str | None = None,
    force: bool = False,
) -> None:
    distribution_name = _normalize_distribution_name(name)
    package_name = _normalize_package_name(package or name)
    resolved_task_id = task_id or _default_task_id(package_name)
    parent = path.expanduser()
    target = (parent / name).resolve()

    if target.exists() and not target.is_dir():
        raise SystemExit(f"target exists and is not a directory: {target}")
    if target.exists() and any(target.iterdir()) and not force:
        raise SystemExit(
            f"target directory is not empty: {target}; use --force to overwrite scaffold files"
        )

    package_dir = Path("src") / package_name
    genelab_source = _relative_path(_default_genelab_source_root(), target)
    robot_name = f"{distribution_name}-robot"
    env_name = f"{distribution_name}-env"
    files = {
        "README.md": _template_readme(name, resolved_task_id),
        "pyproject.toml": _template_pyproject(distribution_name, package_name, genelab_source),
        str(package_dir / "__init__.py"): '"""GeneLab extension project."""\n',
        str(package_dir / "config.py"): _template_config(),
        str(package_dir / "robots.py"): _template_robots(package_name),
        str(package_dir / "envs.py"): _template_envs(package_name),
        str(package_dir / "tasks.py"): _template_tasks(
            package_name,
            task_id=resolved_task_id,
            robot_name=robot_name,
            env_name=env_name,
        ),
    }

    target.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = target / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and file_path.is_dir():
            raise SystemExit(f"cannot overwrite directory with scaffold file: {file_path}")
        if file_path.exists() and not force:
            raise SystemExit(f"scaffold file already exists: {file_path}; use --force to overwrite")
        file_path.write_text(content, encoding="utf-8")

    render_project_created(target, resolved_task_id)


def _normalize_distribution_name(raw: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "-", raw.strip().lower()).strip("-")
    if not name:
        raise SystemExit("project name must contain at least one letter or digit")
    return name


def _normalize_package_name(raw: str) -> str:
    name = re.sub(r"\W+", "_", raw.strip().lower()).strip("_")
    if not name:
        raise SystemExit("package name must contain at least one letter or digit")
    if name[0].isdigit():
        name = f"project_{name}"
    if not name.isidentifier():
        raise SystemExit(f"package name must be a valid Python identifier, got {raw!r}")
    return name


def _default_task_id(package_name: str) -> str:
    title = "".join(part.capitalize() for part in package_name.split("_") if part)
    return f"{title}-Example-v0"


def _default_genelab_source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative_path(source: Path, target: Path) -> str:
    return os.path.relpath(source.resolve(), target.resolve()).replace(os.sep, "/")


def _template_readme(name: str, task_id: str) -> str:
    return dedent(
        f"""\
        # {name}

        This is a GeneLab external project skeleton.

        ```bash
        uv sync
        uv run genelab list tasks
        uv run genelab play {task_id}
        ```

        GeneLab loads this project through the `genelab.extensions` entry point in `pyproject.toml`.
        """
    )


def _template_pyproject(distribution_name: str, package_name: str, genelab_source: str) -> str:
    return dedent(
        f"""\
        [project]
        name = "{distribution_name}"
        version = "0.1.0"
        description = "GeneLab extension project."
        requires-python = ">=3.12"
        dependencies = ["genelab"]

        [project.entry-points."genelab.extensions"]
        {package_name} = "{package_name}.tasks:register"

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{package_name}"]

        [tool.uv.sources]
        genelab = {{ path = "{genelab_source}", editable = true }}
        """
    )


def _template_config() -> str:
    return dedent(
        """\
        \"\"\"Configuration owned by this GeneLab extension project.\"\"\"

        from dataclasses import dataclass, field

        from genelab.configs import ManagerBasedEnvCfg, SceneCfg


        @dataclass
        class RobotCfg:
            asset_path: str = "assets/robot.usd"


        @dataclass
        class EnvCfg(ManagerBasedEnvCfg):
            scene: SceneCfg = field(default_factory=lambda: SceneCfg(steps=128))
            robot: RobotCfg = field(default_factory=RobotCfg)
        """
    )


def _template_robots(package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Robot definitions owned by this GeneLab extension project.\"\"\"

        from dataclasses import dataclass

        from {package_name}.config import RobotCfg


        @dataclass
        class Robot:
            cfg: RobotCfg


        def create_robot(cfg: RobotCfg | None = None) -> Robot:
            return Robot(cfg or RobotCfg())
        """
    )


def _template_envs(package_name: str) -> str:
    return dedent(
        f"""\
        \"\"\"Environment runners owned by this GeneLab extension project.\"\"\"

        from {package_name}.config import EnvCfg


        class ExampleEnv:
            def __init__(self, cfg: EnvCfg | None = None) -> None:
                self.cfg = cfg or EnvCfg()

            def play(self) -> None:
                print(f"Run {{type(self).__name__}} for {{self.cfg.scene.steps}} steps")
        """
    )


def _template_tasks(
    package_name: str,
    *,
    task_id: str,
    robot_name: str,
    env_name: str,
) -> str:
    return dedent(
        f"""\
        \"\"\"GeneLab registration hook for this extension project.\"\"\"

        from genelab.configs import TaskCfg
        from genelab.registry import ENVS, ROBOTS, TASKS, register_env, register_robot, register_task

        from {package_name}.config import EnvCfg, RobotCfg
        from {package_name}.envs import ExampleEnv
        from {package_name}.robots import create_robot


        TASK_ID = "{task_id}"
        ROBOT_NAME = "{robot_name}"
        ENV_NAME = "{env_name}"


        class ExampleTask:
            def __init__(self) -> None:
                self.cfg = TaskCfg(
                    name=TASK_ID,
                    env_name=ENV_NAME,
                    robot_name=ROBOT_NAME,
                    env=EnvCfg(),
                    trainable=False,
                )

            def play(self) -> None:
                if not isinstance(self.cfg.env, EnvCfg):
                    raise TypeError("ExampleTask requires EnvCfg")
                ExampleEnv(self.cfg.env).play()

            def train(self) -> None:
                raise NotImplementedError(f"add your training runner for {{self.cfg.name}}")


        def register() -> None:
            if ROBOT_NAME not in ROBOTS:
                register_robot(
                    ROBOT_NAME,
                    create_robot,
                    description="Robot provided by this GeneLab extension project.",
                    cfg_type=RobotCfg,
                )
            if ENV_NAME not in ENVS:
                register_env(
                    ENV_NAME,
                    ExampleEnv,
                    description="Environment provided by this GeneLab extension project.",
                    cfg_type=EnvCfg,
                )
            if TASK_ID not in TASKS:
                register_task(
                    TASK_ID,
                    ExampleTask,
                    description="Task provided by this GeneLab extension project.",
                    cfg_type=TaskCfg,
                )
        """
    )
