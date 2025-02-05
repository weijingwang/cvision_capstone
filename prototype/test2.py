import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # Ensure you have SORT (e.g., from https://github.com/abewley/sort) available

# -----------------------------
# Initialize YOLO Model and SORT Tracker
# -----------------------------
# Load the YOLO model from your .pt file
model = YOLO('./misc/yolov5su.pt')  # Update with the correct model file path

# Create an instance of the SORT tracker
tracker = Sort()

# -----------------------------
# Start Video Capture
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # Run YOLO Inference
    # -----------------------------
    results = model(frame)
    
    # Prepare detections for SORT:
    # SORT expects detections as a numpy array of shape [N, 5] with columns:
    # [x1, y1, x2, y2, confidence]
    detections = []
    
    # Check if there are detections in the result.
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            # Get the bounding box coordinates as integers.
            # box.xyxy is a tensor of shape (1, 4).
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            # Get the detection confidence. If not available, default to 0.9.
            conf = box.conf[0].cpu().numpy() if hasattr(box, 'conf') else 0.9

            # Append the detection for SORT tracking.
            detections.append([x1, y1, x2, y2, conf])
            
            # Optionally, draw the detection (green box) and label on the frame.
            cls = int(box.cls[0].cpu().numpy())
            label = model.names.get(cls, "object")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert detections to a numpy array; if none, create an empty array.
    dets = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
    
    # -----------------------------
    # Update SORT Tracker
    # -----------------------------
    # tracker.update() returns an array of tracked objects with shape [N, 5],
    # where each row is: [x1, y1, x2, y2, object_id]
    tracked_objects = tracker.update(dets)

    # Loop over tracked objects and draw the tracking info (red box and ID).
    for d in tracked_objects:
        x1, y1, x2, y2, obj_id = d.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # -----------------------------
    # Display the Frame
    # -----------------------------
    cv2.imshow("SORT Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
