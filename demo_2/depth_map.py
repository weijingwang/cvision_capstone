"""
Depth Anything Initialization
1/17/25 PROBABLY NOT METRIC!!! need to use their metric model (but not video...)
1/30/25 replaced with metric depthanything2
"""
import numpy as np
import cv2
import argparse
import torch
from depth_anything_v2.dpt import DepthAnythingV2

def init_depth():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_hypersim_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    return args, depth_anything

def get_plottable_depth(raw_depth, args, cmap):
    # Calculate min and max depth values and their locations
    min_val, max_val = raw_depth.min(), raw_depth.max()
    min_loc = np.unravel_index(np.argmin(raw_depth, axis=None), raw_depth.shape)
    max_loc = np.unravel_index(np.argmax(raw_depth, axis=None), raw_depth.shape)

    # Normalize depth map for rendering
    depth = ((raw_depth - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

    # Apply grayscale or colormap based on args
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)  # Convert to 3-channel grayscale
    else:
        if cmap is None:
            raise ValueError("A colormap must be provided if grayscale is False.")
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)  # Apply colormap and convert to BGR
    return depth, min_val, min_loc, max_val, max_loc

def get_distance_of_object(depth_masked): 
    # Step 1: Identify non-zero elements
    non_zero_elements = depth_masked[depth_masked != 0]

    # Step 2: Calculate the average of non-zero elements
    average_non_zero = np.mean(non_zero_elements)

    # Step 3: Calculate the percentage of non-zero elements
    # total_elements = depth_masked.size
    # non_zero_count = non_zero_elements.size
    # percentage_non_zero = (non_zero_count / total_elements) * 100

    # # Print the results
    # print(f"Average of non-zero elements: {average_non_zero}")
    # print(f"Percentage of non-zero elements: {percentage_non_zero}%")
    return average_non_zero

def get_depth_map(raw_frame, depth_anything, args, cmap=None):
    # Infer depth map from the raw frame
    raw_depth = depth_anything.infer_image(raw_frame, args.input_size)  # float32, same resolution as webcam

    depth_info = get_plottable_depth(raw_depth, args, cmap)
    depth, min_val, min_loc, max_val, max_loc = depth_info[0], depth_info[1], depth_info[2], depth_info[3], depth_info[4]

    cv2.putText(depth, f'Closest: {min_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (255, 0, 255), -1)  # Closest point
    cv2.putText(depth, f'Closest', (min_loc[1], min_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (0, 0, 255), -1)  # Farthest point
    cv2.putText(depth, f'Farthest', (max_loc[1], max_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return raw_depth, depth, min_val, max_val, min_loc, max_loc

