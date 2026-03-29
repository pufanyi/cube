"""Animation controller: interpolates moves and drives the renderer + ffmpeg pipeline."""

import subprocess
import sys
import numpy as np
from typing import Optional

from cube_model import CubeState, parse_moves, rotation_matrix_90, get_wide_layers
from renderer import CubeRenderer


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out for smooth rotation animation."""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - (-2 * t + 2) ** 3 / 2


def render_cube_video(
    moves_str: str,
    output_path: str = "cube.mp4",
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
    move_duration: float = 0.4,
    pause_before: float = 0.5,
    pause_after: float = 0.5,
):
    """Render a Rubik's cube move sequence to an MP4 video.

    Args:
        moves_str: Space-separated move notation (e.g., "R U R' U'").
        output_path: Output file path.
        width: Video width in pixels.
        height: Video height in pixels.
        fps: Frames per second.
        move_duration: Duration of each move animation in seconds.
        pause_before: Pause before the first move (seconds).
        pause_after: Pause after the last move (seconds).
    """
    moves = parse_moves(moves_str)
    cube = CubeState()
    renderer = CubeRenderer(width, height)

    # Start ffmpeg process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frames_per_move = max(1, int(fps * move_duration))
    frames_pause_before = int(fps * pause_before)
    frames_pause_after = int(fps * pause_after)
    total_move_frames = sum(frames_per_move * m[4] for m in moves)  # m[4] = reps
    total_frames = frames_pause_before + total_move_frames + frames_pause_after

    frame_count = 0

    def write_frame(frame_bytes: bytes):
        nonlocal frame_count
        ffmpeg_proc.stdin.write(frame_bytes)
        frame_count += 1
        pct = frame_count * 100 // total_frames
        print(f"\rRendering: {frame_count}/{total_frames} frames ({pct}%)", end="", flush=True)

    # Initial pause
    if frames_pause_before > 0:
        static_frame = renderer.render_frame(cube)
        for _ in range(frames_pause_before):
            write_frame(static_frame)

    # Animate each move
    for move_name, axis, layer, clockwise, reps in moves:
        for rep in range(reps):
            _animate_single_move(
                cube, renderer, axis, layer, clockwise,
                frames_per_move, write_frame, move_name,
            )

    # Final pause
    if frames_pause_after > 0:
        static_frame = renderer.render_frame(cube)
        for _ in range(frames_pause_after):
            write_frame(static_frame)

    # Finalize
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read()
        print(f"\nffmpeg error: {stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone! {frame_count} frames written to {output_path}")
    renderer.release()


def _animate_single_move(
    cube: CubeState,
    renderer: CubeRenderer,
    axis: np.ndarray,
    layer: Optional[int],
    clockwise: bool,
    num_frames: int,
    write_frame,
    move_name: str,
):
    """Animate a single 90-degree move over num_frames frames, then commit."""
    # Determine which cubies are affected
    affected_indices = []
    layers_to_move = [layer]

    # Check for wide moves
    base = move_name[0]
    if 'w' in move_name:
        layers_to_move = get_wide_layers(base)

    for l in layers_to_move:
        cubies = cube.get_affected_cubies(axis, l)
        for c in cubies:
            idx = cube.cubies.index(c)
            if idx not in affected_indices:
                affected_indices.append(idx)

    # Total rotation angle: 90 degrees
    # Direction: negative angle for CW (looking from +axis), positive for CCW
    target_angle = -np.pi / 2 if clockwise else np.pi / 2

    # Render interpolated frames
    for frame_i in range(num_frames):
        t = (frame_i + 1) / num_frames
        eased_t = ease_in_out_cubic(t)
        angle = target_angle * eased_t

        frame_bytes = renderer.render_frame(
            cube,
            rotating_cubies=affected_indices,
            rotation_axis=axis,
            rotation_angle=angle,
        )
        write_frame(frame_bytes)

    # Commit the move to cube state
    for l in layers_to_move:
        cube.apply_move(axis, l, clockwise)
