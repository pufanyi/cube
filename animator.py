"""Animation controller: interpolates moves and drives the renderer + ffmpeg pipeline."""

import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cube_model import CubeState, get_wide_layers, parse_moves, tokenize_moves
from renderer import CubeRenderer

# --- Text overlay configuration ---
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_FONT_PATH_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_COLOR_NORMAL = (180, 180, 180)
_COLOR_CURRENT = (255, 220, 50)
_COLOR_DONE = (100, 100, 100)
_COLOR_STEP = (220, 220, 220)


def _add_move_overlay(
    frame_bytes: bytes,
    width: int,
    height: int,
    move_tokens: list[str],
    current_move_idx: int,
    animating: bool,
) -> bytes:
    """Draw the move sequence and step indicator onto a frame.

    Args:
        frame_bytes: Raw RGB24 bytes.
        width, height: Frame dimensions.
        move_tokens: List of move notation strings (e.g. ["R", "U", "R'", "U'"]).
        current_move_idx: Index of the move currently being animated (-1 for pauses).
        animating: True if a move is actively animating (highlights current move).
    """
    img = Image.frombytes("RGB", (width, height), frame_bytes)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Scale font sizes relative to video height
    seq_font_size = max(20, height // 22)
    step_font_size = max(16, height // 28)
    try:
        seq_font = ImageFont.truetype(_FONT_PATH, seq_font_size)
        step_font = ImageFont.truetype(_FONT_PATH_REGULAR, step_font_size)
    except OSError:
        seq_font = ImageFont.load_default()
        step_font = ImageFont.load_default()

    margin = width // 30
    padding = 12

    # --- Measure text for background ---
    space_w = draw.textlength(" ", font=seq_font)
    total_seq_w = 0.0
    for i, token in enumerate(move_tokens):
        total_seq_w += draw.textlength(token, font=seq_font)
        if i < len(move_tokens) - 1:
            total_seq_w += space_w

    total = len(move_tokens)
    if current_move_idx < 0:
        step_text = f"Step 0 / {total}"
    elif current_move_idx >= total:
        step_text = f"Step {total} / {total}  ✓"
    else:
        step_num = current_move_idx + 1 if animating else current_move_idx
        step_text = f"Step {step_num} / {total}"

    step_w = draw.textlength(step_text, font=step_font)
    box_w = max(total_seq_w, step_w) + padding * 2
    box_h = seq_font_size + step_font_size + padding * 2 + 8

    # --- Draw semi-transparent background ---
    y_top = margin
    draw.rounded_rectangle(
        [margin, y_top, margin + box_w, y_top + box_h],
        radius=10,
        fill=(0, 0, 0, 160),
    )

    # --- Draw move sequence ---
    y_seq = y_top + padding
    x = margin + padding
    for i, token in enumerate(move_tokens):
        if i < current_move_idx:
            color = (*_COLOR_DONE, 255)
        elif i == current_move_idx and animating:
            color = (*_COLOR_CURRENT, 255)
        elif i == current_move_idx and not animating:
            color = (*_COLOR_DONE, 255)
        else:
            color = (*_COLOR_NORMAL, 255)
        draw.text((x, y_seq), token, fill=color, font=seq_font)
        tw = draw.textlength(token, font=seq_font)
        x += tw + space_w

    # --- Draw step indicator ---
    y_step = y_seq + seq_font_size + 8
    draw.text((margin + padding, y_step), step_text, fill=(*_COLOR_STEP, 255), font=step_font)

    # Composite overlay onto frame
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB").tobytes()


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
    move_duration: float = 0.5,
    pause_before: float = 0.5,
    pause_after: float = 0.5,
    scramble_str: str | None = None,
    label: str = "",
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
        scramble_str: Optional scramble to apply before animating (not animated).
        label: Label for progress output (e.g., "[3/10]").
    """
    moves = parse_moves(moves_str)
    move_tokens = tokenize_moves(moves_str)
    cube = CubeState()

    # Apply pre-scramble instantly (no animation)
    if scramble_str:
        for _, axis, layer, clockwise, reps in parse_moves(scramble_str):
            for _ in range(reps):
                cube.apply_move(axis, layer, clockwise)
    renderer = CubeRenderer(width, height)

    # Start ffmpeg process
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",  # Read from stdin
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
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
    # Overlay state: current_move_idx tracks which token we're on
    overlay_state = {"move_idx": -1, "animating": False}

    def write_frame(frame_bytes: bytes):
        nonlocal frame_count
        frame_with_overlay = _add_move_overlay(
            frame_bytes,
            width,
            height,
            move_tokens,
            overlay_state["move_idx"],
            overlay_state["animating"],
        )
        ffmpeg_proc.stdin.write(frame_with_overlay)
        frame_count += 1
        pct = frame_count * 100 // total_frames
        msg = f"\r{label}Rendering: {frame_count}/{total_frames} frames ({pct}%)"
        print(msg, end="", flush=True)

    # Initial pause — show sequence before any move
    if frames_pause_before > 0:
        static_frame = renderer.render_frame(cube)
        for _ in range(frames_pause_before):
            write_frame(static_frame)

    # Animate each move, tracking token index
    for token_idx, (move_name, axis, layer, clockwise, reps) in enumerate(moves):
        overlay_state["move_idx"] = token_idx
        overlay_state["animating"] = True
        for _rep in range(reps):
            _animate_single_move(
                cube,
                renderer,
                axis,
                layer,
                clockwise,
                frames_per_move,
                write_frame,
                move_name,
            )

    # Final pause — all moves done
    overlay_state["move_idx"] = len(move_tokens)
    overlay_state["animating"] = False
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

    print(f"\n{label}Done! {frame_count} frames -> {output_path}")
    renderer.release()


def _animate_single_move(
    cube: CubeState,
    renderer: CubeRenderer,
    axis: np.ndarray,
    layer: int | None,
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
    if "w" in move_name:
        layers_to_move = get_wide_layers(base)

    for lyr in layers_to_move:
        cubies = cube.get_affected_cubies(axis, lyr)
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
    for lyr in layers_to_move:
        cube.apply_move(axis, lyr, clockwise)
