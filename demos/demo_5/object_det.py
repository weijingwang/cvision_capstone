"""
YOLO STUFF
"""

from ultralytics import YOLO
# from sort.sort import Sort  # Ensure you have SORT (e.g., from https://github.com/abewley/sort) available
import numpy as np
import cv2
import globals
from my_constants import *
import supervision as sv

def init_objectDet():
    model = YOLO('yolov8n-seg.pt')  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    # model = YOLO('yolo11n-seg.pt')
    # MODEL_NAMES = list(model.names.values())
    return model


def am_i_dangerous(depth, classname):
    if depth < 1 or classname in DANGEROUS_OBJECTS:
        dangerous = True
    else: dangerous = False
    return dangerous

def process_depth_mask(depth_map, mask, frame_shape):
    """
    Calculate average depth for a given mask
    
    Args:
        depth_map: 2D numpy array of depth values
        mask: Binary mask from YOLOv8
        frame_shape: Shape of the original frame (height, width)
    """
    # Resize mask to match frame dimensions
    mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
    
    # Convert mask to binary uint8 (0 or 255)
    mask_binary = (mask > 0).astype(np.uint8) * 255
    
    # Ensure depth map is in the correct format (float32)
    depth_map = depth_map.astype(np.float32)
    
    # Apply the mask
    masked_depth = cv2.bitwise_and(depth_map, depth_map, mask=mask_binary)
    
    # Calculate average depth of non-zero pixels
    non_zero_depths = masked_depth[masked_depth != 0]
    if len(non_zero_depths) > 0:
        return np.mean(non_zero_depths)
    return 0

def yolo_to_sv_detections(results):
    """Convert YOLO results to supervision Detections format"""
    # Extract boxes, confidence scores, and class IDs
    boxes = results.boxes.xyxy.cpu().numpy()
    conf = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    # Create supervision Detections object
    detections = sv.Detections(
        xyxy=boxes,
        confidence=conf,
        class_id=class_ids,
    )
    return detections

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




def process_yolo_results(frame, model, results, raw_depth, depth_to_plot, tracker):
    # Convert YOLO results to supervision Detections format
    detections = yolo_to_sv_detections(results)
    
    # Update tracks
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
    
    # Store current sounded_already states before clearing
    sounded_states = {
        track_id: obj_data['sounded_already'] 
        for track_id, obj_data in globals.objects_data.items()
    }
    
    # Clear previous objects info
    globals.objects_data.clear()
    
    # Process each detection
    for i in range(len(detections)):
        # Get box coordinates
        box = detections.xyxy[i].astype(int)
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        x_angle = x_center / frame.shape[1]  # Normalize to [0, 1]
        y_angle = y_center / frame.shape[0]  # Normalize to [0, 1]
        
        # Get tracking ID
        track_id = detections.tracker_id[i]
        if track_id is None:
            continue
            
        # Get class information
        class_id = detections.class_id[i]
        class_name = model.names[class_id]
        confidence = detections.confidence[i]
        
        # Get segmentation mask for this object
        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[i].cpu().numpy()
            # Calculate average depth for this object
            avg_depth = process_depth_mask(raw_depth, mask, frame.shape[:2])
        else:
            avg_depth = 0
            mask_vis = None
        
        isDangerous = am_i_dangerous(avg_depth, class_name)


        # DRAWING STUFF MASKS INDIVIDUALLY
        # Create visualization mask
        mask_vis = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_vis = (mask_vis > 0).astype(np.uint8)
        # Draw mask
        colored_mask = np.zeros_like(frame)
        if not isDangerous:
            colored_mask[mask_vis > 0] = [0, 255, 0]  # Green mask
        else:
            colored_mask[mask_vis > 0] = [0, 0, 255]  # red mask BGR
        depth_to_plot = cv2.addWeighted(depth_to_plot, 1, colored_mask, 0.8, 0)
        # Draw bounding box
        cv2.rectangle(depth_to_plot, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Create label with depth
        label = f"{class_name} ({track_id}) {avg_depth:.2f}m"
        cv2.putText(depth_to_plot, label, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store object information, preserving sounded_already state if it exists
        globals.objects_data[track_id] = {
            'class': class_name,
            'depth': float(avg_depth),
            'sounded_already': sounded_states.get(track_id, False),  # Get previous state or False if new
            'confidence': float(confidence),
            'mask_vis': mask_vis,
            'x_angle': float(x_angle),
            'y_angle': float(y_angle),
            'isDangerous' : isDangerous
        }
    
    return depth_to_plot