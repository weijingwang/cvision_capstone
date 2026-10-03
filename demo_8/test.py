import cv2
import numpy as np
from ultralytics import YOLO
import torch
import supervision as sv
from depth_map import *
import matplotlib

cmap = matplotlib.colormaps.get_cmap('gray')

# Initialize YOLOv8 model with segmentation
model = YOLO('yolov8n-seg.pt')

# Load depth map
args, depth_anything = init_depth()
print("Loaded depthmap...")

# Initialize tracker
tracker = sv.ByteTrack()

# Initialize video capture
cap = cv2.VideoCapture('person_walk_test_low.MOV')

# Initialize dictionary to store object information
objects_info = {}

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Simulate depth map generation (replace this with your actual depth map generation)
    # This is just a placeholder - you should integrate your actual depth map here
    depth_map = np.random.rand(*frame.shape[:2]) * 10  # Random depths between 0-10 meters
    depth_map, depth_to_plot = get_depth_map(frame, depth_anything, args, cmap)
    # Run YOLOv8 inference with segmentation
    results = model(frame, conf=0.25)[0]
    
    # Convert YOLO results to supervision Detections format
    detections = yolo_to_sv_detections(results)
    
    # Update tracks
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)

    # Clear previous objects info
    objects_info.clear()

    # Process each detection
    for i in range(len(detections)):
        # Get box coordinates
        box = detections.xyxy[i].astype(int)
        
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
            avg_depth = process_depth_mask(depth_map, mask, frame.shape[:2])
            
            # Create visualization mask
            mask_vis = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_vis = (mask_vis > 0).astype(np.uint8)
            
            # Draw mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask_vis > 0] = [0, 255, 0]  # Green mask
            depth_to_plot = cv2.addWeighted(depth_to_plot, 1, colored_mask, 0.5, 0)
        else:
            avg_depth = 0

        # Draw bounding box
        cv2.rectangle(depth_to_plot, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Create label with depth
        label = f"{class_name} ({track_id}) {avg_depth:.2f}m"
        cv2.putText(depth_to_plot, label, (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store object information
        objects_info[track_id] = {
            'class': class_name,
            'confidence': float(confidence),
            'depth': float(avg_depth)
        }
    
    # Display the objects_info dictionary
    print("\nObjects Info:", objects_info)
    
    # Show the frame
    cv2.imshow('YOLOv8 Segmentation with Depth', depth_to_plot)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()