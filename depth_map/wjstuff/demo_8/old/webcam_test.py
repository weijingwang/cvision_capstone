import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# List of common resolutions
resolutions = [
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
    (640, 480),    # Standard
    (320, 240),    # Low resolution
    (160, 120)     # Very low resolution
]

# Check supported resolutions
print("Checking supported resolutions:")
for width, height in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read the actual width and height set
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (actual_width, actual_height) == (width, height):
        print(f"Resolution {width}x{height} is supported.")
    else:
        print(f"Resolution {width}x{height} is NOT supported.")

# Release the webcam
cap.release()
