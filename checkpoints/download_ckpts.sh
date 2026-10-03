#!/bin/bash

# Downloads the Depth-Anything-V2 metric (Hypersim, Base) checkpoint used by cvision/depth_map.py
cd "$(dirname "$0")"  # always save into checkpoints/, wherever this is run from

BASE_URL="https://huggingface.co/depth-anything"
CKPT="depth_anything_v2_metric_hypersim_vitb.pth"
URL="${BASE_URL}/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/${CKPT}"

echo "Downloading ${CKPT}..."
if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$CKPT" "$URL" || { echo "Failed to download checkpoint from $URL"; exit 1; }
else
    wget -O "$CKPT" "$URL" || { echo "Failed to download checkpoint from $URL"; exit 1; }
fi

echo "Checkpoint downloaded to checkpoints/${CKPT}"
