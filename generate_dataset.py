#!/usr/bin/env python3
"""Generate a dataset of cube rotation videos.

Creates videos starting from random scrambled states with 1/2/3/4-step rotations.

Usage:
    # Dry run: 10 per step count
    uv run generate_dataset.py --dry-run

    # Full: 10K per step count, 8 workers
    uv run generate_dataset.py -w 8

    # Custom output and seed
    uv run generate_dataset.py -o my_dataset --seed 42 -w 16
"""

import argparse
import contextlib
import io
import os
import random
from multiprocessing import Pool

from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn

from animator import render_cube_video
from scramble import generate_scramble


def _render_one(task: dict) -> str:
    """Worker: render one video, suppressing per-frame output."""
    with contextlib.redirect_stdout(io.StringIO()):
        render_cube_video(
            moves_str=task["moves"],
            output_path=task["output"],
            width=task["width"],
            height=task["height"],
            fps=task["fps"],
            move_duration=task["speed"],
            pause_before=task["pause_before"],
            pause_after=task["pause_after"],
            scramble_str=task["scramble"],
        )
    return task["output"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate cube rotation dataset (1/2/3/4 steps from scrambled states)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 10 per step count instead of 10K",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("-o", "--output-dir", default="dataset", help="Output directory (default: dataset/)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--scramble-moves", type=int, default=25, help="Scramble moves (default: 25)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
    parser.add_argument("--resolution", default="1920x1080", help="Resolution WxH (default: 1920x1080)")
    parser.add_argument("--speed", type=float, default=0.5, help="Seconds per move (default: 0.5)")
    parser.add_argument("--pause-before", type=float, default=0.5, help="Pause before first move (default: 0.5s)")
    parser.add_argument("--pause-after", type=float, default=0.5, help="Pause after last move (default: 0.5s)")
    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    n_per_step = 10 if args.dry_run else 10_000
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32)

    # Build tasks grouped by step count
    step_tasks: dict[int, list[dict]] = {}
    for num_steps in range(1, 5):
        step_dir = os.path.join(args.output_dir, f"{num_steps}_step")
        os.makedirs(step_dir, exist_ok=True)

        step_tasks[num_steps] = []
        for i in range(n_per_step):
            seed = base_seed + num_steps * 100_000 + i
            rng = random.Random(seed)

            scramble = generate_scramble(args.scramble_moves, rng)
            moves = generate_scramble(num_steps, rng)

            step_tasks[num_steps].append({
                "moves": moves,
                "scramble": scramble,
                "output": os.path.join(step_dir, f"{i:05d}.mp4"),
                "width": width,
                "height": height,
                "fps": args.fps,
                "speed": args.speed,
                "pause_before": args.pause_before,
                "pause_after": args.pause_after,
            })

    all_tasks = [t for tasks in step_tasks.values() for t in tasks]
    total = len(all_tasks)
    mode = "dry run" if args.dry_run else "full"

    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    with progress:
        overall = progress.add_task(
            f"[bold]Total ({mode}: {n_per_step} x 4)", total=total
        )
        step_bars = {}
        for num_steps in range(1, 5):
            step_bars[num_steps] = progress.add_task(
                f"  {num_steps}-step", total=len(step_tasks[num_steps])
            )

        if args.workers == 1:
            for task in all_tasks:
                _render_one(task)
                # Figure out which step this belongs to
                for ns, tasks in step_tasks.items():
                    if task in tasks:
                        progress.advance(step_bars[ns])
                        break
                progress.advance(overall)
        else:
            # Tag each task with its step count for progress tracking
            for num_steps, tasks in step_tasks.items():
                for task in tasks:
                    task["_step"] = num_steps

            with Pool(processes=args.workers) as pool:
                for path in pool.imap_unordered(_render_one, all_tasks):
                    # Determine step count from path
                    for ns in range(1, 5):
                        if f"{ns}_step" in path:
                            progress.advance(step_bars[ns])
                            break
                    progress.advance(overall)

    progress.console.print(f"[bold green]Done![/] {total} videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
