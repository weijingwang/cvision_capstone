import cv2
import numpy as np

# Example data
cycle_time = 0.05
inference_time = 0.02
depth_time = 0.015

# Compute times
performance_text = [
    f"Tot: {int(cycle_time*1000)}ms",
    f"YOLO: {int(inference_time*1000)}ms",
    f"Depth: {int(depth_time*1000)}ms",
    f"Other: {int((cycle_time-inference_time-depth_time)*1000)}ms"
]

# Create a blank image (black background)
height, width, _ = raw_frame.shape
text_height = 150  # Adjust based on number of lines
performance_image = np.zeros((text_height, width, 3), dtype=np.uint8)  # Black background

# Text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
text_color = (255, 255, 255)  # White
bg_color = (128, 0, 128)  # Purple
padding = 5

# Text position
start_x, start_y = 10, 30
line_spacing = 10  # Space between lines

# Draw text with background
for i, line in enumerate(performance_text):
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
    text_x = start_x
    text_y = start_y + i * (text_height + line_spacing)

    # Draw purple background rectangle
    cv2.rectangle(performance_image, 
                  (text_x - padding, text_y - text_height - padding), 
                  (text_x + text_width + padding, text_y + baseline + padding), 
                  bg_color, 
                  thickness=cv2.FILLED)

    # Draw text over the rectangle
    cv2.putText(performance_image, line, (text_x, text_y), font, font_scale, text_color, font_thickness)

# Plotting
if args.pred_only:
    cv2.imshow('depth only', depth_to_plot)
else:
    top_row_frame = cv2.hconcat([raw_frame, depth_to_plot])
    bottom_row_frame = performance_image  # Now an actual image with text + background
    final_output = cv2.vconcat([top_row_frame, bottom_row_frame])
    cv2.imshow("wjdemo2", final_output)
