import cv2
import torch
import numpy as np
from ultralytics import YOLO
import random

# Constants
WEBCAM_RESOLUTION = (640, 480)  # (1280, 720)
FRAME_SKIP = 5  # Process every 5th frame (adjust as needed)

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Use 'yolov8m-seg.pt' or 'yolov8l-seg.pt' for better accuracy

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RESOLUTION[1])

# Ensure webcam opens
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Generate random colors for each class
num_classes = len(model.names)
colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}

# Variables to store the last results
last_results = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only run YOLO inference every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        # Run YOLOv8 segmentation on the current frame
        last_results = model(frame)

    # If we have results from a previous inference, use them
    if last_results is not None:
        for result in last_results:
            masks = result.masks  # Extract masks
            boxes = result.boxes  # Bounding boxes
            names = model.names   # Class names

            if masks is not None:
                for i, mask in enumerate(masks.xy):  # Get polygon coordinates
                    class_id = int(boxes.cls[i])  # Get class ID
                    conf = boxes.conf[i].item()  # Confidence score
                    label = f"{names[class_id]} {conf:.2f}"

                    # Convert polygon mask to OpenCV format
                    mask_pts = np.array(mask, dtype=np.int32)

                    # Assign a unique color per class
                    color = colors[class_id]

                    # Draw mask overlay
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [mask_pts], color=color)  # Unique class color
                    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                    # Compute centroid of the mask
                    M = cv2.moments(mask_pts)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = mask_pts.mean(axis=0).astype(int)  # Fallback to mean point

                    # Draw text label in the center of the mask
                    font_scale = 1.2
                    font_thickness = 3
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    text_x = cx - text_size[0] // 2
                    text_y = cy + text_size[1] // 2

                    # Draw text with contrasting color
                    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)  # White text for dark masks, black for light
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                font_scale, text_color, font_thickness, cv2.LINE_AA)

    frame_count += 1

    # Show output
    cv2.imshow("YOLOv8 Segmentation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()