#!/usr/bin/env python3
"""Organize cube rotation dataset: extract first frames and create JSON metadata.

Usage:
    uv run organize_dataset.py                # full run
    uv run organize_dataset.py --dry-run      # process 10 per step
    uv run organize_dataset.py -w 16          # 16 parallel workers
"""

import argparse
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


def extract_frame(task: dict) -> dict:
    """Extract the first frame from a video and return metadata entry."""
    video_path = task["video_path"]
    image_path = task["image_path"]

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        "-q:v", "2",
        image_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return {
        "video": task["video_rel"],
        "image": [task["image_rel"]],
        "prompt": task["prompt"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Organize cube dataset: extract frames + create JSONs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Process only 10 per step")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("-i", "--input-dir", default="dataset", help="Input video directory")
    parser.add_argument("-o", "--output-dir", default="dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompts")
    args = parser.parse_args()

    rng = random.Random(args.seed)

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
        for num_steps in range(1, 5):
            step_name = f"{num_steps}_step"
            video_dir = os.path.join(args.input_dir, step_name)

            # Collect video files
            videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
            if args.dry_run:
                videos = videos[:10]

            # Build tasks
            tasks = []
            for v in videos:
                name = os.path.splitext(v)[0]
                image_rel = f"{step_name}_frames/{name}.jpg"
                tasks.append({
                    "video_path": os.path.join(video_dir, v),
                    "image_path": os.path.join(args.output_dir, image_rel),
                    "video_rel": f"{step_name}/{v}",
                    "image_rel": image_rel,
                    "prompt": rng.choice(PROMPTS),
                })

            # Extract frames in parallel
            bar = progress.add_task(f"[bold]{step_name}", total=len(tasks))
            results = []

            if args.workers == 1:
                for t in tasks:
                    results.append(extract_frame(t))
                    progress.advance(bar)
            else:
                with Pool(processes=args.workers) as pool:
                    for entry in pool.imap_unordered(extract_frame, tasks):
                        results.append(entry)
                        progress.advance(bar)

            # Sort by video path for consistent ordering
            results.sort(key=lambda x: x["video"])

            # Write JSON
            json_path = os.path.join(args.output_dir, f"{step_name}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            progress.console.print(f"  [green]Saved {json_path}[/] ({len(results)} entries)")

    progress.console.print("[bold green]Done![/]")


if __name__ == "__main__":
    main()
