#!/usr/bin/env python3
"""Rubik's Cube video renderer.

Usage:
    # Single video with explicit moves
    uv run main.py render "R U R' U'"
    uv run main.py render "R B L2 F' D" -o solve.mp4

    # Batch: generate N random videos in parallel
    uv run main.py batch -n 10 --moves 20 --workers 4 -o output_dir/
    uv run main.py batch -n 5 --moves 15 --scramble --workers 8
"""

import argparse
import os
import random
from multiprocessing import Pool

from animator import render_cube_video
from scramble import generate_scramble


def _render_one(task: dict):
    """Worker function for multiprocessing. Each call creates its own EGL context."""
    render_cube_video(
        moves_str=task["moves"],
        output_path=task["output"],
        width=task["width"],
        height=task["height"],
        fps=task["fps"],
        move_duration=task["speed"],
        pause_before=task["pause_before"],
        pause_after=task["pause_after"],
        scramble_str=task.get("scramble"),
        label=task.get("label", ""),
    )
    return task["output"]


def cmd_render(args):
    """Render a single video from explicit moves."""
    width, height = map(int, args.resolution.split("x"))
    render_cube_video(
        moves_str=args.moves,
        output_path=args.output,
        width=width,
        height=height,
        fps=args.fps,
        move_duration=args.speed,
        pause_before=args.pause_before,
        pause_after=args.pause_after,
    )


def cmd_batch(args):
    """Generate N random videos in parallel."""
    width, height = map(int, args.resolution.split("x"))
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for i in range(args.num):
        rng = random.Random(args.seed + i if args.seed is not None else None)

        # Generate random moves to animate
        moves = generate_scramble(args.moves, rng)

        # Optionally scramble the initial state first
        scramble = None
        if args.scramble:
            scramble = generate_scramble(args.scramble_moves, rng)

        filename = f"cube_{i:04d}.mp4"
        output_path = os.path.join(args.output_dir, filename)

        tasks.append(
            {
                "moves": moves,
                "scramble": scramble,
                "output": output_path,
                "width": width,
                "height": height,
                "fps": args.fps,
                "speed": args.speed,
                "pause_before": args.pause_before,
                "pause_after": args.pause_after,
                "label": f"[{i + 1}/{args.num}] ",
            }
        )

    print(
        f"Generating {args.num} videos ({args.moves} moves each, "
        f"{args.workers} workers, {width}x{height}@{args.fps}fps)"
    )
    if args.scramble:
        print(f"  Initial scramble: {args.scramble_moves} random moves")
    if args.seed is not None:
        print(f"  Seed: {args.seed}")
    print()

    if args.workers == 1:
        for task in tasks:
            _render_one(task)
    else:
        with Pool(processes=args.workers) as pool:
            pool.map(_render_one, tasks)

    print(f"\nAll done! {args.num} videos saved to {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Rubik's Cube video renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- render subcommand ---
    p_render = subparsers.add_parser(
        "render",
        help="Render a single video from explicit moves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py render "R U R' U'"
  uv run main.py render "R B L2 F' D" -o solve.mp4
  uv run main.py render "R U R' U' R' F R2 U' R' U' R U R' F'" --speed 0.3

Supported notation:
  Faces:  R, L, U, D, F, B
  Slices: M, E, S
  Whole:  x, y, z
  Wide:   Rw (or r), Lw, Uw, Dw, Fw, Bw
  Modifiers: ' (prime/CCW), 2 (double)
    """,
    )
    p_render.add_argument(
        "moves",
        help="Move sequence (e.g., \"R U R' U'\")",
    )
    p_render.add_argument(
        "-o",
        "--output",
        default="cube.mp4",
        help="Output file (default: cube.mp4)",
    )
    _add_common_args(p_render)

    # --- batch subcommand ---
    p_batch = subparsers.add_parser(
        "batch",
        help="Generate N random videos in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py batch -n 10 --moves 20 -o videos/
  uv run main.py batch -n 100 --moves 25 --scramble --workers 8
  uv run main.py batch -n 5 --moves 10 --scramble --scramble-moves 30 --seed 42
    """,
    )
    p_batch.add_argument(
        "-n",
        "--num",
        type=int,
        required=True,
        help="Number of videos to generate",
    )
    p_batch.add_argument(
        "--moves",
        type=int,
        default=20,
        help="Number of random moves per video (default: 20)",
    )
    p_batch.add_argument(
        "--scramble",
        action="store_true",
        help="Start from a random scrambled state (instead of solved)",
    )
    p_batch.add_argument(
        "--scramble-moves",
        type=int,
        default=25,
        help="Scramble moves for initial state (default: 25)",
    )
    p_batch.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    p_batch.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Output directory (default: output/)",
    )
    p_batch.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    _add_common_args(p_batch)

    args = parser.parse_args()

    if args.command == "render":
        cmd_render(args)
    elif args.command == "batch":
        cmd_batch(args)


def _add_common_args(parser):
    """Add rendering options shared by both subcommands."""
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second (default: 60)",
    )
    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="Resolution WxH (default: 1920x1080)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.4,
        help="Seconds per move (default: 0.4)",
    )
    parser.add_argument(
        "--pause-before",
        type=float,
        default=0.5,
        help="Pause before first move (default: 0.5s)",
    )
    parser.add_argument(
        "--pause-after",
        type=float,
        default=0.5,
        help="Pause after last move (default: 0.5s)",
    )


if __name__ == "__main__":
    main()
