#!/bin/bash


# Define the URLs for the checkpoints
BASE_URL="https://huggingface.co/depth-anything"
dv2_metric_base_url="${BASE_URL}/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth"


echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
wget $dv2_metric_base_url || { echo "Failed to download checkpoint from $dv2_metric_base_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
