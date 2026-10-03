import cv2

# Open webcam
cap = cv2.VideoCapture(0)  # Change index if multiple cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get default FPS
default_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Default FPS: {default_fps}")

# Test different FPS settings
for test_fps in [15, 30, 60, 120, 240]:  # Adjust based on your webcam
    cap.set(cv2.CAP_PROP_FPS, test_fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Requested: {test_fps} FPS -> Actual: {actual_fps:.2f} FPS")

cap.release()
