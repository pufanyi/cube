#!/bin/bash
set -e

OUTPUT_DIR="dataset_3"

# Clean up old incompatible data (old seed is lost, videos can't match new metadata)
if [ -d "$OUTPUT_DIR/1_step" ]; then
    echo "Removing old $OUTPUT_DIR/1_step and 1_step_frames (incompatible with new seed)..."
    rm -rf "$OUTPUT_DIR/1_step" "$OUTPUT_DIR/1_step_frames" "$OUTPUT_DIR/1_step.json"
fi

uv run generate_dataset.py \
    -n 100000 \
    --min-steps 1 \
    --max-steps 1 \
    --seed 42 \
    -w 32 \
    -o "$OUTPUT_DIR"
