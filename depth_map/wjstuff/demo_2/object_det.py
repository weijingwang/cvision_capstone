"""
YOLO STUFF
"""

from ultralytics import YOLO
import numpy as np
import cv2
import globals
from my_constants import *

def init_objectDet():
    model = YOLO('yolov8n-seg.pt')  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    return model

def process_SAM_mask(combined_mask):
    combined_mask_resized = cv2.resize(combined_mask, WEBCAM_RESOLUTION)#(raw_frame.shape[1], raw_frame.shape[0]))
    combined_mask_for_show = cv2.cvtColor(combined_mask_resized*255, cv2.COLOR_GRAY2BGR)
    combined_mask_for_show = combined_mask_for_show.astype(np.uint8)
    return combined_mask_resized, combined_mask_for_show


def process_yolo_results(results, raw_frame, raw_depth, names):
    # Initialize variables
    combined_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)  # Same size as the frame
    objects = []
    apple_detected = False
    person_detected = False
    red_circle_position = 0
    depth_person = np.inf  
    x_center = 0
    y_center = 0

    # Process each detection result
    for result in results:
        masks = result.masks  # Segmentation masks
        boxes = result.boxes  # Bounding boxes
       # Class names

        # Check if masks and boxes are available
        if masks is not None and boxes is not None:
            # Iterate over each detected object
            for i in range(len(boxes)):
                # Get bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integers

                # Get confidence score
                confidence = boxes.conf[i].item()

                # Get class ID and class name
                class_id = int(boxes.cls[i])
                class_name = names[class_id]

                # Add detected object to the list
                objects.append((class_name, confidence))

                # Get mask for the current object
                mask_points = masks.xy[i].astype(int)  # Convert to integers

                # Draw bounding box
                cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw mask overlay
                overlay = raw_frame.copy()
                cv2.fillPoly(overlay, [mask_points], (0, 255, 0))
                raw_frame = cv2.addWeighted(overlay, 0.3, raw_frame, 0.7, 0)

                # Add label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(raw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Additional logic for specific objects
                if class_name == "apple":
                    apple_detected = True

                if class_name == "cell phone":
                    # Get mask for the current person
                    mask = masks.xy[i]  # Polygon points for the mask

                    # Convert polygon points to a binary mask
                    mask_pts = np.array(mask, dtype=np.int32)
                    person_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(person_mask, [mask_pts], 1)  # Fill the polygon with 1s

                    # Combine the person mask with the combined mask using logical OR
                    combined_mask = cv2.bitwise_or(combined_mask, person_mask)

                    person_detected = True
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Get depth value at the center of the bounding box
                    if 0 <= y_center < raw_depth.shape[0] and 0 <= x_center < raw_depth.shape[1]:
                        depth_person = raw_depth[y_center, x_center]
                    else:
                        print(f"Coordinates ({x_center}, {y_center}) are out of bounds for the depth map.")

                    # Draw a red circle at the center of the bounding box
                    cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

                    # Track the horizontal position of the red circle (panning position)
                    red_circle_position = x_center / raw_frame.shape[1]  # Normalize to [0, 1]
    globals.objects_buffer = objects
    return raw_frame, combined_mask, depth_person, apple_detected, person_detected, red_circle_position, x_center, y_center