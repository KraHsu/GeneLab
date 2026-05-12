"""Compatibility wrapper for the registered Wuji hand playback task."""

import argparse

from genelab.cli import main as genelab_main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play the bundled Wuji trajectory on the Wuji dexterous hand in Genesis."
    )
    parser.add_argument("--side", choices=("left", "right"), default="right", help="Hand side to load.")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the Genesis viewer.")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend instead of CPU.")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of Genesis steps to run; 0 loops until the viewer closes.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Genesis simulation timestep in seconds.")
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=500,
        help="Hard-reset interval in steps; 0 disables periodic reset.",
    )
    args = parser.parse_args()

    argv = [
        "play",
        "GeneLab-Wuji-Hand-Playback-v0",
        "--steps",
        str(args.steps),
        "--env.robot.side",
        args.side,
        "--env.scene.dt",
        str(args.dt),
        "--env.reset_interval",
        str(args.reset_interval),
    ]
    if args.vis:
        argv[2:2] = ["--vis"]
    if args.gpu:
        argv[2:2] = ["--gpu"]
    genelab_main(["--import", "genelab_examples.tasks", *argv])


if __name__ == "__main__":
    main()
