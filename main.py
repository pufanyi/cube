#!/usr/bin/env python3
"""Rubik's Cube video renderer.

Usage:
    uv run main.py "R U R' U'"
    uv run main.py "R B L2 F' D" -o solve.mp4 --fps 60 --resolution 1920x1080
"""

import argparse
from animator import render_cube_video


def main():
    parser = argparse.ArgumentParser(
        description="Render Rubik's Cube move sequences as video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py "R U R' U'"
  uv run main.py "R B L2 F' D" -o solve.mp4
  uv run main.py "R U R' U' R' F R2 U' R' U' R U R' F'" --fps 30 --speed 0.3

Supported notation:
  Faces:  R, L, U, D, F, B
  Slices: M, E, S
  Whole:  x, y, z
  Wide:   Rw (or r), Lw, Uw, Dw, Fw, Bw
  Modifiers: ' (prime/CCW), 2 (double)
        """,
    )
    parser.add_argument("moves", help="Move sequence in standard notation (e.g., \"R U R' U'\")")
    parser.add_argument("-o", "--output", default="cube.mp4", help="Output video file (default: cube.mp4)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
    parser.add_argument("--resolution", default="1920x1080", help="Video resolution WxH (default: 1920x1080)")
    parser.add_argument("--speed", type=float, default=0.4, help="Duration per move in seconds (default: 0.4)")
    parser.add_argument("--pause-before", type=float, default=0.5, help="Pause before first move (default: 0.5s)")
    parser.add_argument("--pause-after", type=float, default=0.5, help="Pause after last move (default: 0.5s)")

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
