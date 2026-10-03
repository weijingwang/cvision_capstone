import matplotlib.pyplot as plt
import matplotlib
import cv2
from depth_map import *
from my_constants import *

import numpy as np

# Initialize depth map
depth_init = init_depth()
args = depth_init[0]
depth_anything = depth_init[1]
print("Loaded depthmap...")

# Initialize webcam
cmap = matplotlib.colormaps.get_cmap('gray')
cap = cv2.VideoCapture(1)  # webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

if args.pred_only:
    output_width = frame_width
else:
    output_width = frame_width * 2 + MARGIN_WIDTH

print("Webcam started...")
print("Fully Initialized")

while cap.isOpened():
    # Webcam variables
    ret, raw_frame = cap.read()
    if not ret:
        break
    raw_frame = cv2.flip(raw_frame, 1)

    # Depth math and get depth map to render
    raw_depth = depth_anything.infer_image(raw_frame, args.input_size)
    min_val = raw_depth.min()
    max_val = raw_depth.max()
    min_loc = np.unravel_index(np.argmin(raw_depth, axis=None), raw_depth.shape)
    max_loc = np.unravel_index(np.argmax(raw_depth, axis=None), raw_depth.shape)
    print(min_val, min_loc, max_val, max_loc)

    depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
    depth = depth.astype(np.uint8)
    if args.grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    cv2.putText(depth, f'Closest: {min_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (255, 0, 0), -1)  # Closest point
    cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (0, 0, 255), -1)  # Farthest point

    combined_frame = cv2.hconcat([raw_frame, depth])


    # Display the combined frame
    cv2.imshow('Depth and Raw Frame', combined_frame)

    # Exit the loop if ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
