import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (segmentation version)
model = YOLO("yolov8n-seg.pt")  # You can use yolov8s-seg.pt, yolov8m-seg.pt, etc.

# Class index for 'person' in the COCO dataset
PERSON_CLASS = 0

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, replace with the video source if needed

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Predict with YOLOv8
    results = model(frame)

    # Get masks and class IDs
    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Get masks
            class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs

            # Create an empty mask for the "person" class
            person_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Combine all person masks
            for mask, class_id in zip(masks, class_ids):
                if int(class_id) == PERSON_CLASS:  # Check if the detected class is "person"
                    # Resize the mask to match the frame size (if needed)
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0-255
                    person_mask = cv2.bitwise_or(person_mask, mask_resized)

            # Apply the mask to the original frame (grayscale mask to 3 channels)
            person_mask_3channel = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
            person_video = cv2.bitwise_and(frame, person_mask_3channel)

            # Show the segmented person video
            cv2.imshow("Person Segmentation", person_video)

    # Show the original frame for reference
    cv2.imshow("Original Video", person_video)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
