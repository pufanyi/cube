#!/usr/bin/env python3
"""Extract missing frames from already-generated videos.

Usage:
    uv run extract_missing_frames.py -o dataset_3 --steps 1 -w 8
"""

import argparse
import os
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


def _extract_frame(task: dict) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Extract missing frames from generated videos")
    parser.add_argument("-o", "--output-dir", default="dataset_3", help="Dataset directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[1], help="Step counts to process (e.g. --steps 1 2 3)")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers (default: 8, keep low!)")
    args = parser.parse_args()

    for num_steps in args.steps:
        step_name = f"{num_steps}_step"
        video_dir = os.path.join(args.output_dir, step_name)
        frames_dir = os.path.join(args.output_dir, f"{step_name}_frames")

        if not os.path.isdir(video_dir):
            print(f"Skipping {step_name}: {video_dir} not found")
            continue

        # Find videos missing their frame
        tasks = []
        videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
        for v in videos:
            name = os.path.splitext(v)[0]
            frame_path = os.path.join(frames_dir, f"{name}.jpg")
            if not os.path.exists(frame_path):
                tasks.append({
                    "video_path": os.path.join(video_dir, v),
                    "image_path": frame_path,
                })

        print(f"{step_name}: {len(videos)} videos, {len(videos) - len(tasks)} frames exist, {len(tasks)} missing")

        if not tasks:
            continue

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
            bar = progress.add_task(f"[bold]{step_name} — Extracting missing frames", total=len(tasks))
            if args.workers == 1:
                for t in tasks:
                    _extract_frame(t)
                    progress.advance(bar)
            else:
                with Pool(processes=args.workers) as pool:
                    for _ in pool.imap_unordered(_extract_frame, tasks):
                        progress.advance(bar)

    print("Done!")


if __name__ == "__main__":
    main()
