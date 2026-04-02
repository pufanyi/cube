#!/usr/bin/env python3
"""Generate 1-step eval dataset: first frames, GT videos, questions & answers JSONs.

Creates:
  dataset/1_step_eval/           MP4 ground-truth videos
  dataset/1_step_eval_frames/    First-frame JPGs (model input)
  dataset/1_step_eval_questions.json
  dataset/1_step_eval_answers.json

Usage:
    uv run generate_eval_dataset.py --dry-run      # 10 samples
    uv run generate_eval_dataset.py -w 8           # 500 samples, 8 workers
    uv run generate_eval_dataset.py -n 1000 -w 8   # custom count
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

# Human-readable descriptions for every possible 1-step move
MOVE_DESCRIPTIONS = {
    "R": "Rotate the right face of the cube 90 degrees clockwise.",
    "R'": "Rotate the right face of the cube 90 degrees counter-clockwise.",
    "R2": "Rotate the right face of the cube 180 degrees.",
    "L": "Rotate the left face of the cube 90 degrees clockwise.",
    "L'": "Rotate the left face of the cube 90 degrees counter-clockwise.",
    "L2": "Rotate the left face of the cube 180 degrees.",
    "U": "Rotate the upper face of the cube 90 degrees clockwise.",
    "U'": "Rotate the upper face of the cube 90 degrees counter-clockwise.",
    "U2": "Rotate the upper face of the cube 180 degrees.",
    "D": "Rotate the bottom face of the cube 90 degrees clockwise.",
    "D'": "Rotate the bottom face of the cube 90 degrees counter-clockwise.",
    "D2": "Rotate the bottom face of the cube 180 degrees.",
    "F": "Rotate the front face of the cube 90 degrees clockwise.",
    "F'": "Rotate the front face of the cube 90 degrees counter-clockwise.",
    "F2": "Rotate the front face of the cube 180 degrees.",
    "B": "Rotate the back face of the cube 90 degrees clockwise.",
    "B'": "Rotate the back face of the cube 90 degrees counter-clockwise.",
    "B2": "Rotate the back face of the cube 180 degrees.",
}

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


def main():
    parser = argparse.ArgumentParser(description="Generate 1-step eval dataset")
    parser.add_argument("--dry-run", action="store_true", help="Generate 10 samples only")
    parser.add_argument("-n", "--num-samples", type=int, default=500, help="Number of samples (default: 500)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("-o", "--output-dir", default="dataset", help="Output directory (default: dataset/)")
    parser.add_argument("--seed", type=int, default=99999, help="Random seed (default: 99999, avoids overlap with training)")
    parser.add_argument("--scramble-moves", type=int, default=25, help="Scramble moves (default: 25)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second (default: 60)")
    parser.add_argument("--resolution", default="1920x1080", help="Resolution WxH (default: 1920x1080)")
    parser.add_argument("--speed", type=float, default=0.5, help="Seconds per move (default: 0.5)")
    parser.add_argument("--pause-before", type=float, default=0.5, help="Pause before move (default: 0.5s)")
    parser.add_argument("--pause-after", type=float, default=0.5, help="Pause after move (default: 0.5s)")
    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    n_samples = 10 if args.dry_run else args.num_samples
    base_seed = args.seed

    video_dir = os.path.join(args.output_dir, "1_step_eval")
    frames_dir = os.path.join(args.output_dir, "1_step_eval_frames")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    rng_prompts = random.Random(base_seed)

    # ── Phase 1: Build render tasks ──────────────────────────────────
    tasks = []
    for i in range(n_samples):
        seed = base_seed + i
        rng = random.Random(seed)

        scramble = generate_scramble(args.scramble_moves, rng)
        move = generate_scramble(1, rng)  # single move

        tasks.append({
            "idx": i,
            "moves": move,
            "scramble": scramble,
            "output": os.path.join(video_dir, f"{i:05d}.mp4"),
            "width": width,
            "height": height,
            "fps": args.fps,
            "speed": args.speed,
            "pause_before": args.pause_before,
            "pause_after": args.pause_after,
            "prompt": rng_prompts.choice(PROMPTS),
        })

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

    # ── Phase 2: Render videos ───────────────────────────────────────
    completed_tasks = []
    with progress:
        bar = progress.add_task(f"[bold]Rendering videos ({mode}: {n_samples})", total=n_samples)

        if args.workers == 1:
            for t in tasks:
                completed_tasks.append(_render_one(t))
                progress.advance(bar)
        else:
            with Pool(processes=args.workers) as pool:
                for t in pool.imap_unordered(_render_one, tasks):
                    completed_tasks.append(t)
                    progress.advance(bar)

    # Sort by index for consistent ordering
    completed_tasks.sort(key=lambda t: t["idx"])

    # ── Phase 3: Extract first frames ────────────────────────────────
    frame_tasks = []
    for t in completed_tasks:
        name = f"{t['idx']:05d}"
        frame_tasks.append({
            "video_path": t["output"],
            "image_path": os.path.join(frames_dir, f"{name}.jpg"),
        })

    with progress:
        bar = progress.add_task("[bold]Extracting frames", total=len(frame_tasks))

        if args.workers == 1:
            for ft in frame_tasks:
                _extract_frame(ft)
                progress.advance(bar)
        else:
            with Pool(processes=args.workers) as pool:
                for _ in pool.imap_unordered(_extract_frame, frame_tasks):
                    progress.advance(bar)

    # ── Phase 4: Write JSON files ────────────────────────────────────
    questions = []
    answers = []

    for t in completed_tasks:
        sample_id = f"{t['idx']:05d}"
        move = t["moves"].strip()

        questions.append({
            "id": sample_id,
            "image": [f"1_step_eval_frames/{sample_id}.jpg"],
            "prompt": t["prompt"],
        })

        answers.append({
            "id": sample_id,
            "move": move,
            "answer": MOVE_DESCRIPTIONS.get(move, f"Perform the move: {move}"),
            "video": f"1_step_eval/{sample_id}.mp4",
        })

    q_path = os.path.join(args.output_dir, "1_step_eval_questions.json")
    a_path = os.path.join(args.output_dir, "1_step_eval_answers.json")

    with open(q_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)

    with open(a_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)

    print(f"\nDone! {n_samples} eval samples generated.")
    print(f"  Videos:    {video_dir}/")
    print(f"  Frames:    {frames_dir}/")
    print(f"  Questions: {q_path}")
    print(f"  Answers:   {a_path}")


if __name__ == "__main__":
    main()
