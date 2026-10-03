import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model (segmentation model)
model = YOLO("yolov8s-seg.pt")  # Use the appropriate model path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change to the correct webcam index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Get the masks from the results
    combined_mask = None
    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data  # Masks as binary numpy arrays
            # Combine all masks into a single mask
            for mask in masks:
                mask = mask.cpu().numpy().astype(np.uint8) * 255  # Convert to binary mask
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Ensure the combined mask is not None
    if combined_mask is None:
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Convert the combined mask to BGR for display
    combined_mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    # Resize the combined mask to match the original frame's dimensions
    combined_mask_resized = cv2.resize(combined_mask_bgr, (frame.shape[1], frame.shape[0]))

    # Concatenate the original frame (left) and the resized mask (right)
    concatenated_display = np.hstack((frame, combined_mask_resized))

    # Show the combined view
    cv2.imshow("YOLOv8 Segmentation - Webcam (Left) | Masks (Right)", concatenated_display)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
