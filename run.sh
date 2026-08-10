#!/bin/bash
set -e
cd "$(dirname "$0")"

# Parse config.txt
CLASSES=$(awk -F'=' '/^\[model\]/{s=1} s && /^classes/{print $2; exit}' config.txt | tr -d ' ')
MODE=$(awk -F'=' '/^\[model\]/{s=1} s && /^mode/{print $2; exit}' config.txt | tr -d ' ')
IMGSZ=$(awk -F'=' '/^\[model\]/{s=1} s && /^imgsz/{print $2; exit}' config.txt | tr -d ' ')
MODE=${MODE:-seg}
IMGSZ=${IMGSZ:-1008}

# comma-separated → space-separated for argparse
CLASSES_ARGS=$(echo "$CLASSES" | tr ',' ' ')
ENGINE_DIR="sam3_${IMGSZ}_${MODE}_$(echo "$CLASSES" | tr ',' '_')"

echo "[run] classes=${CLASSES}  mode=${MODE}  imgsz=${IMGSZ}  engine=${ENGINE_DIR}"

# Step 1: specialize — skip if engine already exists
if [ -f "${ENGINE_DIR}/sam3.engine" ]; then
    echo "[run] Engine found, skipping specialize.py"
else
    echo "[run] Exporting engine..."
    CUDA_VISIBLE_DEVICES=1 python3 specialize.py \
        --classes ${CLASSES_ARGS} --mode ${MODE} --imgsz ${IMGSZ} --device cuda
fi

# Step 2: run pipeline
echo "[run] Starting pipeline..."
CUDA_VISIBLE_DEVICES=1 python3 ds9_rtsp.py
