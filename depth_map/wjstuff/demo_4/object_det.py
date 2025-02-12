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

    # combined_mask_resized, combined_mask_for_show = process_SAM_mask(combined_mask)[0], process_SAM_mask(combined_mask)[1]
    # depth_masked = combined_mask_resized * raw_depth

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def get_distance_of_object(depth_masked): # input is a segment of depth map
    # Step 1: Identify non-zero elements
    non_zero_elements = depth_masked[depth_masked != 0]

    # Step 2: Calculate the average of non-zero elements
    average_non_zero = np.mean(non_zero_elements)
    return average_non_zero

def process_yolo_results(results, raw_frame, raw_depth, names, tracker, object_state=None):
    """
    Processes YOLO detection results and returns a dictionary of tracked objects with their properties.
    Uses YOLO's original detections for visualization while maintaining SORT tracking for ID consistency.

    Args:
        results: A list of detection results from the YOLO model.
        raw_frame: The original color frame (numpy array).
        raw_depth: The depth map corresponding to raw_frame.
        names: A dictionary or list mapping class IDs to class names.
        tracker: An instance of the SORT tracker.
        object_state: Dictionary tracking previous frame's objects. If None, initializes empty dict.

    Returns:
        Dictionary with object IDs as keys, containing object properties
    """
    if object_state is None:
        object_state = {}

    objects = []
    yolo_detections = []  # Store all YOLO detections
    masks_dict = {}
    
    # First pass: Process and store all YOLO detections
    for result in results:
        masks = result.masks
        boxes = result.boxes

        if masks is not None and boxes is not None:
            for i in range(len(boxes)):
                # Get original YOLO bounding box
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                confidence = boxes.conf[i].item()
                class_id = int(boxes.cls[i])
                class_name = names[class_id]
                
                # Store YOLO detection for tracking
                yolo_detections.append([x1, y1, x2, y2, confidence])

                # Create mask and get its bounding box
                if masks is not None:
                    mask_points = masks.xy[i].astype(int)
                    obj_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(obj_mask, [mask_points], 1)
                    


                    # Get the actual bounding box from the mask
                    mask_indices = np.where(obj_mask > 0)
                    if len(mask_indices[0]) > 0:  # If mask is not empty
                        min_y, max_y = np.min(mask_indices[0]), np.max(mask_indices[0])
                        min_x, max_x = np.min(mask_indices[1]), np.max(mask_indices[1])
                        
                        # Add padding to ensure full coverage (adjust padding as needed)
                        padding = 5
                        min_x = max(0, min_x - padding)
                        min_y = max(0, min_y - padding)
                        max_x = min(raw_frame.shape[1], max_x + padding)
                        max_y = min(raw_frame.shape[0], max_y + padding)
                        
                        # Use the larger of YOLO bbox and mask bbox
                        x1 = min(x1, min_x)
                        y1 = min(y1, min_y)
                        x2 = max(x2, max_x)
                        y2 = max(y2, max_y)
                
                # Store detection data
                masks_dict[i] = {
                    'mask': obj_mask if masks is not None else None,
                    'bbox': [x1, y1, x2, y2],
                    'class_name': class_name,
                    'confidence': confidence
                }

    # Update SORT tracker
    yolo_detections = np.array(yolo_detections) if yolo_detections else np.empty((0, 5))
    tracked_objects = tracker.update(yolo_detections)
    
    new_state = {}
    


    # Second pass: Match tracking IDs with detections
    for track in tracked_objects:
        obj_id = int(track[4])
        track_bbox = track[:4]
        
        # Find best matching detection using IoU
        best_iou = 0
        best_match = None
        
        for det_idx, det_data in masks_dict.items():
            iou = calculate_iou(track_bbox, det_data['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = det_data
        
        # If we found a good match
        if best_match and best_iou > 0.3:
            bbox = best_match['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate center points
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            
            # Calculate normalized positions
            x_angle = x_center / raw_frame.shape[1]
            y_angle = y_center / raw_frame.shape[0]
            
            # Get depth at center
            # depth = raw_depth[y_center, x_center] if (0 <= y_center < raw_depth.shape[0] and 
            #                                          0 <= x_center < raw_depth.shape[1]) else np.inf
            obj_mask_processed, obj_mask_for_show = process_SAM_mask(obj_mask)
            depth_masked = obj_mask_processed * raw_depth
            depth = get_distance_of_object(depth_masked)

            # Maintain temporal consistency
            if obj_id in object_state:
                prev_bbox = object_state[obj_id]['bbox']
                # Apply slight smoothing to reduce jitter
                smooth_factor = 0.8  # Adjust this value (0-1) to control smoothing
                bbox = [
                    int(smooth_factor * prev_bbox[0] + (1 - smooth_factor) * bbox[0]),
                    int(smooth_factor * prev_bbox[1] + (1 - smooth_factor) * bbox[1]),
                    int(smooth_factor * prev_bbox[2] + (1 - smooth_factor) * bbox[2]),
                    int(smooth_factor * prev_bbox[3] + (1 - smooth_factor) * bbox[3])
                ]
            
            # Determine if this is a new object
            sound_played = object_state.get(obj_id, {}).get('sound_played', False)
            
            # Store object data
            new_state[obj_id] = {
                'mask': best_match['mask'],
                'bbox': bbox,
                'depth': depth,
                'x_angle': x_angle,
                'y_angle': y_angle,
                'sound_played': sound_played,
                'class_name': best_match['class_name'],
                'confidence': best_match['confidence']
            }
            
            # Draw the bounding box and label

            
            cv2.rectangle(raw_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"ID {obj_id} {best_match['class_name']} {depth:.2f}m"
            cv2.putText(raw_frame, label, 
                       (int(bbox[0]), int(bbox[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)

    globals.objects_buffer = objects
    return raw_frame, new_state