# Cube

A 3D Rubik's Cube video renderer. Generates MP4 videos of move sequences with smooth animations, Phong-lit rendering, and mirror stickers for hidden faces.

https://github.com/user-attachments/assets/placeholder

## Features

- **3D rendering** with ModernGL (EGL offscreen) and Phong lighting
- **Mirror stickers** for hidden faces (back, left, bottom), similar to [alg.cubing.net](https://alg.cubing.net)
- **Move overlay** with highlighted current move
- **Full notation support**: R, L, U, D, F, B, M, E, S, x, y, z, wide moves (Rw/r), prime (`'`), double (`2`)
- **Batch generation** with parallel workers
- **WCA-style scrambles** with proper constraints

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- FFmpeg
- EGL support (GPU or software rendering)

## Quick Start

```bash
# Clone and sync dependencies
git clone https://github.com/pufanyi/cube.git
cd cube
uv sync

# Render a single video
uv run main.py render "R U R' U' R' F R2 U' R' U' R U R' F'"

# Custom output and speed
uv run main.py render "R U R' U'" -o solve.mp4 --speed 0.3
```

## Usage

### Single Video

```bash
uv run main.py render <moves> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `cube.mp4` | Output file path |
| `--fps` | `60` | Frames per second |
| `--resolution` | `1920x1080` | Video resolution (WxH) |
| `--speed` | `0.5` | Seconds per move |
| `--pause-before` | `0.5` | Pause before first move (seconds) |
| `--pause-after` | `0.5` | Pause after last move (seconds) |

### Batch Generation

Generate multiple videos with random moves in parallel:

```bash
uv run main.py batch -n <count> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --num` | (required) | Number of videos to generate |
| `--moves` | `20` | Random moves per video |
| `--scramble` | off | Start from a scrambled state |
| `--scramble-moves` | `25` | Number of scramble moves |
| `-w, --workers` | `4` | Parallel workers |
| `-o, --output-dir` | `output/` | Output directory |
| `--seed` | (random) | Random seed for reproducibility |

All rendering options from `render` are also available.

**Examples:**

```bash
# 10 videos, 20 moves each, 4 workers
uv run main.py batch -n 10 --moves 20 --workers 4

# 100 videos from scrambled states, reproducible
uv run main.py batch -n 100 --scramble --seed 42 --workers 8

# Lower resolution for faster rendering
uv run main.py batch -n 50 --resolution 960x540 --workers 8
```

### Move Notation

| Type | Moves | Description |
|------|-------|-------------|
| Face | `R L U D F B` | Single face rotation |
| Slice | `M E S` | Middle layer |
| Whole | `x y z` | Whole cube rotation |
| Wide | `Rw` or `r` | Face + adjacent middle layer |
| Prime | `R'` | Counter-clockwise |
| Double | `R2` | 180-degree turn |

## Project Structure

```
├── main.py          # CLI entry point (render / batch)
├── animator.py      # Animation loop, move overlay, ffmpeg pipeline
├── renderer.py      # ModernGL offscreen renderer, Phong shading
├── cube_model.py    # Cube state, move parsing, cubie management
├── scramble.py      # WCA-style random scramble generator
└── pyproject.toml   # Project config (uv, ruff)
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## License

[MIT](LICENSE)
