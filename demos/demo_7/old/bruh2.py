import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Use 'yolov8m-seg.pt' or 'yolov8l-seg.pt' for better accuracy

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Main loop for processing webcam frames
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 segmentation on the frame
    results = model(frame, verbose=False)  # Disable verbose output for cleaner logs

    # Initialize an empty combined mask for persons
    combined_person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Same size as the frame

    # Process results
    for result in results:
        masks = result.masks  # Segmentation masks
        boxes = result.boxes  # Bounding boxes
        names = model.names   # Class names

        # Check if masks and boxes are available
        if masks is not None and boxes is not None:
            # Iterate over each detected object
            for i in range(len(boxes)):
                # Get class ID and class name
                class_id = int(boxes.cls[i])
                class_name = names[class_id]

                # Check if the detected object is a person
                if class_name == "person":
                    # Get mask for the current person
                    mask = masks.xy[i]  # Polygon points for the mask

                    # Convert polygon points to a binary mask
                    mask_pts = np.array(mask, dtype=np.int32)
                    person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(person_mask, [mask_pts], 1)  # Fill the polygon with 1s

                    # Combine the person mask with the combined mask using logical OR
                    combined_person_mask = cv2.bitwise_or(combined_person_mask, person_mask)

    # Convert the combined mask to a 3-channel image for visualization
    combined_mask_visual = cv2.cvtColor(combined_person_mask * 255, cv2.COLOR_GRAY2BGR)

    # Overlay the combined mask on the original frame
    overlay = cv2.addWeighted(frame, 0.7, combined_mask_visual, 0.3, 0)

    # Display the frame with the combined mask
    cv2.imshow("Combined Person Mask", overlay)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()