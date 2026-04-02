#!/usr/bin/env python3
"""Build JSON and extract frames for dataset_2/1_step (already-generated videos)."""

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

BASE_DIR = "dataset_2"
STEP_NAME = "1_step"
SEED = 42
WORKERS = 16


def _extract_frame(task: dict) -> dict:
    """Returns task dict with 'ok' flag."""
    cmd = [
        "ffmpeg", "-y",
        "-i", task["video_path"],
        "-frames:v", "1",
        "-update", "1",
        "-q:v", "2",
        task["image_path"],
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {**task, "ok": result.returncode == 0}


def main():
    video_dir = os.path.join(BASE_DIR, STEP_NAME)
    frames_dir = os.path.join(BASE_DIR, f"{STEP_NAME}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    rng = random.Random(SEED)

    # Build tasks
    tasks = []
    for v in videos:
        name = os.path.splitext(v)[0]
        tasks.append({
            "video_path": os.path.join(video_dir, v),
            "image_path": os.path.join(frames_dir, f"{name}.jpg"),
            "video_rel": f"{STEP_NAME}/{v}",
            "image_rel": f"{STEP_NAME}_frames/{name}.jpg",
            "prompt": rng.choice(PROMPTS),
        })

    # Extract frames
    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    )
    failed = []
    succeeded = []
    with progress:
        bar = progress.add_task("[bold]Extracting frames", total=len(tasks))
        with Pool(processes=WORKERS) as pool:
            for result in pool.imap_unordered(_extract_frame, tasks):
                if result["ok"]:
                    succeeded.append(result)
                else:
                    failed.append(result["video_path"])
                progress.advance(bar)

    if failed:
        print(f"\nWarning: {len(failed)} videos failed to extract frames (corrupted?):")
        for f in failed[:20]:
            print(f"  {f}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Write JSON (only for successfully extracted frames)
    succeeded.sort(key=lambda t: t["video_rel"])
    entries = []
    for t in succeeded:
        entries.append({
            "video": t["video_rel"],
            "image": [t["image_rel"]],
            "prompt": t["prompt"],
        })

    json_path = os.path.join(BASE_DIR, f"{STEP_NAME}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

    print(f"Done! {len(entries)} entries -> {json_path} ({len(failed)} skipped)")


if __name__ == "__main__":
    main()
