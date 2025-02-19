import cv2

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define text properties
top_text = "First time singing on stream"
bottom_text = "Be Nice :)"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_thickness = 8
text_color = (255, 255, 255)  # White text
bg_color = (0, 0, 0)  # Black background

while True:
    ret, raw_frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Get text size
    (top_text_w, top_text_h), _ = cv2.getTextSize(top_text, font, font_scale, font_thickness)
    (bottom_text_w, bottom_text_h), _ = cv2.getTextSize(bottom_text, font, font_scale, font_thickness)

    # Draw background for text
    cv2.rectangle(raw_frame, (0, 0), (raw_frame.shape[1], top_text_h + 10), bg_color, -1)  # Top
    cv2.rectangle(raw_frame, (0, raw_frame.shape[0] - bottom_text_h - 10), (bottom_text_w + 20, raw_frame.shape[0]), bg_color, -1)  # Bottom

    # Draw text
    cv2.putText(raw_frame, top_text, (5, top_text_h + 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(raw_frame, bottom_text, (5, raw_frame.shape[0] - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Webcam Stream", raw_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
