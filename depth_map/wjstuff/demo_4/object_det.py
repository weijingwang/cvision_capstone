"""
YOLO STUFF
"""

from ultralytics import YOLO
from sort.sort import Sort  # Ensure you have SORT (e.g., from https://github.com/abewley/sort) available
import numpy as np
import cv2
import globals
from my_constants import *


def init_objectDet():
    model = YOLO('yolov8n-seg.pt')  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    # model = YOLO('yolo11n-seg.pt')
    # MODEL_NAMES = list(model.names.values())
    return model

def init_sort():
    mot_tracker = Sort()
    return mot_tracker

def process_SAM_mask(combined_mask):
    combined_mask_resized = cv2.resize(combined_mask, WEBCAM_RESOLUTION)#(raw_frame.shape[1], raw_frame.shape[0]))
    combined_mask_for_show = cv2.cvtColor(combined_mask_resized*255, cv2.COLOR_GRAY2BGR)
    combined_mask_for_show = combined_mask_for_show.astype(np.uint8)
    return combined_mask_resized, combined_mask_for_show


def process_yolo_results(results, raw_frame, raw_depth, names, tracker):
    """
    Processes YOLO detection results, overlays segmentation masks and bounding boxes,
    computes additional information (like depth and danger detection), and integrates
    the SORT tracker to assign consistent IDs to detected objects.

    Args:
        results: A list of detection results from the YOLO model.
        raw_frame: The original color frame (numpy array).
        raw_depth: The depth map corresponding to raw_frame.
        names: A dictionary or list mapping class IDs to class names.
        tracker: An instance of the SORT tracker.

    Returns:
        A tuple with:
          - modified raw_frame (with overlays),
          - combined binary mask,
          - depth_obj (depth value for a specific object, if found),
          - danger_detected (boolean flag),
          - obj_detected (boolean flag for a specific object),
          - x_angle (normalized horizontal position),
          - x_center, y_center (center coordinates for the specific object).
    """

    # Initialize variables
    combined_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)  # Same size as the frame
    objects = []  # List to hold detected objects (class_name, confidence)
    danger_detected = False
    obj_detected = False
    depth_obj = np.inf  
    x_center = 0
    y_center = 0
    # Variables for Will HRTF
    x_angle = 0
    y_angle = 0

    # List for SORT detections; each element is [x1, y1, x2, y2, confidence]
    sort_dets = []

    # Process each detection result
    for result in results:
        masks = result.masks  # Segmentation masks
        boxes = result.boxes  # Bounding boxes

        # Check if masks and boxes are available
        if masks is not None and boxes is not None:
            # Iterate over each detected object
            for i in range(len(boxes)):
                # Get bounding box coordinates in xyxy format
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integers

                # Get confidence score
                confidence = boxes.conf[i].item()

                # Get class ID and class name
                class_id = int(boxes.cls[i])
                class_name = names[class_id]
                objects.append((class_name, confidence))

                # Add detection for SORT tracking
                sort_dets.append([x1, y1, x2, y2, confidence])

                # Draw bounding box on the raw frame (magenta)
                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Draw mask overlay
                mask_points = masks.xy[i].astype(int)  # Convert to integer coordinates
                overlay = raw_frame.copy()
                cv2.fillPoly(overlay, [mask_points], (255, 0, 255))
                raw_frame = cv2.addWeighted(overlay, 0.3, raw_frame, 0.7, 0)

                # Add label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(raw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                # Additional logic for specific objects
                # (Assuming DANGEROUS_OBJECTS is defined globally)
                if class_name in DANGEROUS_OBJECTS:
                    danger_detected = True

                # If the detected object matches the voice command
                if class_name == globals.voice_command:
                    # Get mask for the current person/object
                    mask = masks.xy[i]  # Polygon points for the mask

                    # Convert polygon points to a binary mask
                    mask_pts = np.array(mask, dtype=np.int32)
                    obj_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(obj_mask, [mask_pts], 1)  # Fill polygon with 1s

                    # Combine with the global combined mask
                    combined_mask = cv2.bitwise_or(combined_mask, obj_mask)

                    obj_detected = True
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Get depth value at the center of the bounding box
                    if 0 <= y_center < raw_depth.shape[0] and 0 <= x_center < raw_depth.shape[1]:
                        depth_obj = raw_depth[y_center, x_center]
                    else:
                        print(f"Coordinates ({x_center}, {y_center}) are out of bounds for the depth map.")

                    # Draw a red circle at the center of the bounding box
                    cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

                    # Track the horizontal position of the red circle (normalized)
                    # Variables for will's HRTF
                    x_angle = x_center / raw_frame.shape[1]
                    y_angle = y_center / raw_frame.shape[0]

    # Convert SORT detections to a NumPy array (or an empty array if no detections)
    sort_dets = np.array(sort_dets) if len(sort_dets) > 0 else np.empty((0, 5))
    
    # Update SORT tracker; it returns an array where each row is [x1, y1, x2, y2, object_id]
    tracked_objects = tracker.update(sort_dets)

    # Draw tracking information (red bounding boxes with IDs)
    for d in tracked_objects:
        tx1, ty1, tx2, ty2, obj_id = d.astype(int)
        cv2.rectangle(raw_frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        cv2.putText(raw_frame, f"ID {obj_id}", (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    globals.objects_buffer = objects

    return (raw_frame, combined_mask, depth_obj, danger_detected, 
            obj_detected, x_angle, y_angle, x_center, y_center)
