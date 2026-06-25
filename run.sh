#!/bin/bash
set -e
cd "$(dirname "$0")"

# Parse config.txt
CLASSES=$(awk -F'=' '/^\[model\]/{s=1} s && /^classes/{print $2; exit}' config.txt | tr -d ' ')
MASK_MODE=$(awk -F'=' '/^\[model\]/{s=1} s && /^mask_mode/{print $2; exit}' config.txt | tr -d ' ')

# comma-separated → space-separated for argparse
CLASSES_ARGS=$(echo "$CLASSES" | tr ',' ' ')
ENGINE_DIR="sam3_${MASK_MODE}_$(echo "$CLASSES" | tr ',' '_')"

echo "[run] classes=${CLASSES}  mask=${MASK_MODE}  engine=${ENGINE_DIR}"

# Step 1: specialize — skip if engine already exists
if [ -f "${ENGINE_DIR}/sam3.engine" ]; then
    echo "[run] Engine found, skipping specialize.py"
else
    echo "[run] Exporting engine..."
    CUDA_VISIBLE_DEVICES=1 python3 specialize.py \
        --classes ${CLASSES_ARGS} --mask ${MASK_MODE} --device cuda
fi

# Step 2: run pipeline
echo "[run] Starting pipeline..."
CUDA_VISIBLE_DEVICES=1 python3 ds_vis_psm.py
