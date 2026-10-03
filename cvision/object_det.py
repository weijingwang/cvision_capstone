"""
object_det.py
YOLO AND ARUCO DETECTION
"""

from ultralytics import YOLO
import numpy as np
import cv2
import globals
from my_constants import *
import supervision as sv

def init_objectDet():
    model = YOLO('yolov8x-seg.pt')
    return model

def init_aruco_detector():
    """Initialize ArUco marker detector based on OpenCV version"""
    opencv_major_ver = int(cv2.__version__.split('.')[0])
    
    if opencv_major_ver >= 4:
        try:
            # Try newer API first (OpenCV 4.7+)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            def detect_func(img):
                return detector.detectMarkers(img)
                
        except AttributeError:
            # Fall back to older API (OpenCV 4.0-4.6)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            
            def detect_func(img):
                return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    else:
        # Very old OpenCV 3.x
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        
        def detect_func(img):
            return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    
    return detect_func

def am_i_dangerous(depth, classname):
    if depth < DANGER_METER or classname in DANGEROUS_OBJECTS:
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

def detect_aruco_markers(frame, raw_depth, aruco_detector, depth_to_plot, cached_aruco_ids=None):
    """
    Detect ArUco markers in the frame and add them to globals.objects_data
    
    Args:
        frame: Input color frame
        raw_depth: Raw depth map
        aruco_detector: Function to detect ArUco markers
        depth_to_plot: Visualization frame to draw on
        cached_aruco_ids: List of previously detected ArUco IDs (optional)
    
    Returns:
        depth_to_plot: Updated visualization frame
        detected_aruco_ids: List of detected ArUco IDs in this frame
    """
    # Keep track of detected markers in this frame
    detected_aruco_ids = []
    
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector(gray)
    
    # Process detected markers
    if ids is not None and len(ids) > 0:
        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(depth_to_plot, corners, ids)
        
        # Process each detected marker
        for i in range(len(ids)):
            # Get marker ID
            marker_id = ids[i][0]
            track_id = f"aruco_{marker_id}"
            detected_aruco_ids.append(track_id)  # Track which markers are visible
            
            # Get the corners of the marker
            corner = corners[i][0]
            corner = corner.astype(np.int32)
            
            # Get the bounding box
            x_min = int(min(corner[:, 0]))
            y_min = int(min(corner[:, 1]))
            x_max = int(max(corner[:, 0]))
            y_max = int(max(corner[:, 1]))
            
            # Calculate center
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            
            # Normalize to [0, 1]
            x_angle = x_center / frame.shape[1]
            y_angle = y_center / frame.shape[0]
            
            # Create a mask for this marker
            marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(marker_mask, [corner], 255)
            
            # Calculate depth
            avg_depth = process_depth_mask(raw_depth, marker_mask, frame.shape[:2])
            
            # Check if this ArUco marker is dangerous (using same criteria as other objects)
            class_name = f"ArUco_{marker_id}"
            is_dangerous = am_i_dangerous(avg_depth, class_name)
            
            # Set border color based on danger status
            border_color = (0, 0, 255) if is_dangerous else (0, 255, 255)
            
            # Draw bounding box with appropriate color
            cv2.rectangle(depth_to_plot, (x_min, y_min), (x_max, y_max), border_color, 2)
            
            # Add marker ID and depth info
            label = f"ArUco ID: {marker_id} {avg_depth:.2f}m"
            cv2.putText(depth_to_plot, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
            
            # Draw the mask with appropriate color
            colored_mask = np.zeros_like(frame)
            if is_dangerous:
                colored_mask[marker_mask > 0] = [0, 0, 255]  # Red mask for dangerous markers
            else:
                colored_mask[marker_mask > 0] = [0, 255, 255]  # Original yellow mask
            
            depth_to_plot = cv2.addWeighted(depth_to_plot, 1, colored_mask, 0.8, 0)
            
            # Store object information
            globals.objects_data[track_id] = {
                'class': class_name,
                'depth': float(avg_depth),
                'sounded_already': globals.objects_data.get(track_id, {}).get('sounded_already', False),
                'confidence': 1.0,  # ArUco markers are deterministic
                'mask_vis': marker_mask,
                'x_angle': float(x_angle),
                'y_angle': float(y_angle),
                'isDangerous': is_dangerous,  # Apply danger criteria to ArUco markers
                'marker_id': int(marker_id)  # Store the marker ID explicitly
            }
    
    # Only remove ArUco markers that haven't been detected for a while
    # (In this case, we're using cached_aruco_ids to determine which ones to keep)
    if cached_aruco_ids is not None:
        for track_id in list(globals.objects_data.keys()):
            if (isinstance(track_id, str) and track_id.startswith("aruco_") and 
                track_id not in detected_aruco_ids and track_id not in cached_aruco_ids):
                globals.objects_data.pop(track_id)
    
    # Return the updated visualization and the list of detected ArUco IDs
    return depth_to_plot, detected_aruco_ids

def process_yolo_results(frame, model, results, raw_depth, depth_to_plot, tracker, aruco_detector=None):
    
    # Process ArUco markers if detector is provided
    if aruco_detector is not None:
        # Create a fresh copy for ArUco visualization to avoid accumulating boxes
        depth_to_plot, _ = detect_aruco_markers(frame, raw_depth, aruco_detector, depth_to_plot.copy())
    
    # Convert YOLO results to supervision Detections format
    detections = yolo_to_sv_detections(results)
    
    # Update tracks
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
    
    # Store current sounded_already states before clearing
    sounded_states = {
        track_id: obj_data['sounded_already'] 
        for track_id, obj_data in globals.objects_data.items()
        if not (isinstance(track_id, str) and track_id.startswith("aruco_"))  # Check if it's a string first
    }

    # Clear previous objects info (except ArUco markers)
    for track_id in list(globals.objects_data.keys()):
        if not (isinstance(track_id, str) and track_id.startswith("aruco_")):
            globals.objects_data.pop(track_id)

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
        
        # Always skip objects in ALWAYS_IGNORE list regardless of danger level
        if class_name in ALWAYS_IGNORE:
            continue
        
        # Get segmentation mask for this object
        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[i].cpu().numpy()
            # Calculate average depth for this object
            avg_depth = process_depth_mask(raw_depth, mask, frame.shape[:2])
        else:
            avg_depth = 0
            mask_vis = None
        
        isDangerous = am_i_dangerous(avg_depth, class_name)
        
        # Skip objects in IGNORE_OBJECTS list unless they're dangerous
        if class_name in IGNORE_OBJECTS and not isDangerous:
            continue

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