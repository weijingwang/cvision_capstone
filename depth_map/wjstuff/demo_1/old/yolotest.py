import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)

# Open the video file or webcam (use 0 for the default webcam)
video_source = "sitar video - 1st class - 1-7-25.mov"  # Replace with 0 for a webcam feed
cap = cv2.VideoCapture(0)  # Replace with `video_source` if using a file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Timer initialization
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Get frame dimensions
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Run YOLOv8 inference on the frame
    results = model(frame, verbose=False)

    # Collect detected objects
    objects = []
    for detection in results[0].boxes.data:
        x_min, y_min, x_max, y_max = map(float, detection[:4])  # Bounding box coordinates
        class_id = int(detection[5])  # Class ID
        confidence = float(detection[4])  # Confidence score
        x_center = (x_min + x_max) / 2  # X-coordinate of the center
        y_center = (y_min + y_max) / 2  # Y-coordinate of the center

        # Normalize x_center to range [-50, 50]
        normalized_x = ((x_center / frame_width) * 100) - 50

        objects.append((model.names[class_id], confidence, normalized_x))

        # If the detected object is a "person", draw a red circle at the center of the bounding box
        if model.names[class_id] == "person":
            center = (int(x_center), int(y_center))
            cv2.circle(frame, center, radius=5, color=(0, 0, 255), thickness=-1)  # Red circle

    # Print detected objects every second
    current_time = time.time()
    if current_time - start_time >= 1.0:
        print("Detected objects:")
        for obj_name, confidence, normalized_x in objects:
            print(f"- {obj_name}: {confidence:.2f}, Normalized X-Center: {normalized_x:.2f}")
        start_time = current_time  # Reset the timer

    # Display the frame with annotations
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
