"""
SAM example
"""

import cv2
from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO('yolov8n-seg.pt')  # Replace with a larger model if needed (e.g., yolov8m-seg.pt)

# Use GPU if available
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model.to(device)

# Open webcam
video_path = "video2.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Set webcam resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # # Downscale the frame for faster processing
    # resized_frame = cv2.resize(frame, (640, 480))

    # Perform segmentation
    results = model(frame)

    # Draw the results on the original frame for display
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Real-Time Segmentation", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
