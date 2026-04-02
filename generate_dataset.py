#!/usr/bin/env python3
"""Generate cube rotation dataset: render videos, extract first frames, create JSON metadata.

Combines video generation and dataset organization into a single pipeline.

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
import json
import os
import random
import subprocess
from multiprocessing import Pool

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from animator import render_cube_video
from scramble import generate_scramble

PROMPTS = [
    "Rotate the Rubik's cube following the instructions shown in the image.",
    "Apply the move sequence displayed in the image to the Rubik's cube.",
    "Follow the notation in the image to rotate the cube accordingly.",
    "Perform the cube rotations indicated by the instructions in the image.",
    "Execute the moves shown in the image on the Rubik's cube.",
    "Turn the Rubik's cube according to the steps shown in the image.",
    "Use the move notation in the image to manipulate the cube.",
    "Carry out the Rubik's cube operations displayed in the image.",
    "Rotate the cube by following the move sequence in the image.",
    "Apply the indicated rotations from the image to the Rubik's cube.",
    "Perform the Rubik's cube moves as instructed in the image.",
    "Follow the operation sequence in the image to turn the cube.",
    "Execute the rotation steps shown in the image on the cube.",
    "Manipulate the Rubik's cube using the instructions from the image.",
    "Complete the cube rotations based on the notation in the image.",
    "Turn the cube following the formula displayed in the image.",
    "Apply the moves from the image to rotate the Rubik's cube.",
    "Observe the instructions in the image and rotate the cube accordingly.",
    "Perform the indicated cube operations as shown in the image.",
    "Follow the move instructions in the image to transform the cube.",
    "Rotate the Rubik's cube as directed by the image overlay.",
    "Use the notation shown in the image to perform the cube moves.",
    "Carry out the cube rotations according to the image instructions.",
    "Execute the move formula displayed in the image on the Rubik's cube.",
    "Apply the rotation sequence from the image to the cube.",
    "Follow the steps in the image to rotate the Rubik's cube.",
    "Perform the cube transformations indicated in the image.",
    "Turn the Rubik's cube by following the displayed move notation.",
    "Complete the moves shown in the image on the Rubik's cube.",
    "Rotate the cube according to the operation symbols in the image.",
    "Use the image instructions to perform the Rubik's cube rotations.",
    "Execute the indicated moves from the image on the cube.",
    "Follow the rotation instructions in the image to turn the cube.",
    "Apply the cube moves shown in the image step by step.",
    "Perform the rotations on the Rubik's cube as shown in the image.",
    "Manipulate the cube following the move sequence in the image.",
    "Turn the cube according to the rotation formula in the image.",
    "Carry out the Rubik's cube moves indicated in the image.",
    "Rotate the cube based on the instructions displayed in the image.",
    "Execute the cube rotation sequence shown in the image.",
    "Follow the image notation and apply the moves to the Rubik's cube.",
    "Perform the Rubik's cube rotations following the image guide.",
    "Apply the displayed instructions to rotate the Rubik's cube.",
    "Turn the Rubik's cube using the move formula from the image.",
    "Complete the cube operations as indicated by the image.",
    "Rotate the Rubik's cube by executing the moves in the image.",
    "Use the instructions from the image to turn the Rubik's cube.",
    "Follow the move annotations in the image to rotate the cube.",
    "Perform the cube moves according to the image instructions.",
    "Apply the rotation steps displayed in the image to the cube.",
]


def _render_one(task: dict) -> dict:
    """Worker: render one video, return task dict with metadata."""
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
    return task


def _extract_frame(task: dict) -> str:
    """Extract the first frame from a video as JPG."""
    os.makedirs(os.path.dirname(task["image_path"]), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", task["video_path"],
        "-frames:v", "1",
        "-update", "1",
        "-q:v", "2",
        task["image_path"],
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return task["image_path"]


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )


def _run_parallel(func, tasks, workers, progress, bar):
    """Run tasks with a worker pool (or single-threaded) and update progress bar."""
    results = []
    if workers == 1:
        for t in tasks:
            results.append(func(t))
            progress.advance(bar)
    else:
        with Pool(processes=workers) as pool:
            for result in pool.imap_unordered(func, tasks):
                results.append(result)
                progress.advance(bar)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate cube rotation dataset: videos, frames, and JSON metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate 10 per step count instead of full dataset",
    )
    parser.add_argument("-n", "--num-per-step", type=int, default=100_000, help="Samples per step count (default: 100000)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("-o", "--output-dir", default="dataset_3", help="Output directory (default: dataset/)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--scramble-moves", type=int, default=25, help="Scramble moves (default: 25)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
    parser.add_argument("--resolution", default="1920x1080", help="Resolution WxH (default: 1920x1080)")
    parser.add_argument("--speed", type=float, default=0.5, help="Seconds per move (default: 0.5)")
    parser.add_argument("--pause-before", type=float, default=0.5, help="Pause before first move (default: 0.5s)")
    parser.add_argument("--pause-after", type=float, default=0.5, help="Pause after last move (default: 0.5s)")
    parser.add_argument("--min-steps", type=int, default=1, help="Minimum step count (default: 1)")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum step count (default: 4)")
    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    n_per_step = 10 if args.dry_run else args.num_per_step
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**32)
    mode = "dry run" if args.dry_run else "full"

    # Generation config stored in metadata
    generation_config = {
        "base_seed": base_seed,
        "scramble_moves": args.scramble_moves,
        "resolution": args.resolution,
        "fps": args.fps,
        "speed": args.speed,
        "pause_before": args.pause_before,
        "pause_after": args.pause_after,
        "num_per_step": n_per_step,
    }

    for num_steps in range(args.min_steps, args.max_steps + 1):
        step_name = f"{num_steps}_step"
        video_dir = os.path.join(args.output_dir, step_name)
        frames_dir = os.path.join(args.output_dir, f"{step_name}_frames")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        rng_prompts = random.Random(base_seed + num_steps)

        # ── Phase 1: Build render tasks ─────────────────────────────
        render_tasks = []
        for i in range(n_per_step):
            seed = base_seed + num_steps * 1_000_000 + i
            rng = random.Random(seed)

            scramble = generate_scramble(args.scramble_moves, rng)
            moves = generate_scramble(num_steps, rng)

            render_tasks.append({
                "idx": i,
                "moves": moves,
                "scramble": scramble,
                "seed": seed,
                "output": os.path.join(video_dir, f"{i:05d}.mp4"),
                "width": width,
                "height": height,
                "fps": args.fps,
                "speed": args.speed,
                "pause_before": args.pause_before,
                "pause_after": args.pause_after,
                "prompt_base": rng_prompts.choice(PROMPTS),
            })

        # ── Phase 2: Render videos ──────────────────────────────────
        progress = _make_progress()
        with progress:
            bar = progress.add_task(
                f"[bold]{step_name} — Rendering ({mode}: {n_per_step})",
                total=len(render_tasks),
            )
            completed = _run_parallel(_render_one, render_tasks, args.workers, progress, bar)

        completed.sort(key=lambda t: t["idx"])

        # ── Phase 3: Extract first frames ───────────────────────────
        frame_tasks = []
        for t in completed:
            name = f"{t['idx']:05d}"
            frame_tasks.append({
                "video_path": t["output"],
                "image_path": os.path.join(frames_dir, f"{name}.jpg"),
            })

        progress = _make_progress()
        with progress:
            bar = progress.add_task(
                f"[bold]{step_name} — Extracting frames",
                total=len(frame_tasks),
            )
            _run_parallel(_extract_frame, frame_tasks, args.workers, progress, bar)

        # ── Phase 4: Write JSON ─────────────────────────────────────
        entries = []
        for t in completed:
            sample_id = f"{t['idx']:05d}"
            moves = t["moves"].strip()
            prompt = f"{t['prompt_base']} The moves are: {moves}"

            entries.append({
                "video": f"{step_name}/{sample_id}.mp4",
                "image": [f"{step_name}_frames/{sample_id}.jpg"],
                "prompt": prompt,
                "metadata": {
                    "id": sample_id,
                    "num_steps": num_steps,
                    "moves": moves,
                    "scramble": t["scramble"],
                    "seed": t["seed"],
                    "generation": generation_config,
                },
            })

        json_path = os.path.join(args.output_dir, f"{step_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)

        print(f"  Saved {json_path} ({len(entries)} entries)")

    print(f"\nDone! Videos and metadata saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
