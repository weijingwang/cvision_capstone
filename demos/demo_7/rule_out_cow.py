from depth_map import *
import cv2
import matplotlib
import numpy as np

# Load image in BGR format for openCV
image = cv2.imread("cow2.jpg")
# Convert to NumPy array
image_array = np.array(image)
args, depth_anything = init_depth()
print("Loaded depthmap...")
cmap = matplotlib.colormaps.get_cmap('gray')
raw_depth, depth_to_plot = get_depth_map(image, depth_anything, args, cmap)

print(depth_to_plot.shape)

# Get min and max depth values
min_depth = np.min(raw_depth)
max_depth = np.max(raw_depth)
print(f"Depth range: {min_depth:.2f} to {max_depth:.2f}")

raw_depth[raw_depth > 5] = 5
depth_to_plot = get_plottable_depth(raw_depth, args, cmap)[0]
print(depth_to_plot.shape)

# Convert raw_depth to a compatible type
raw_depth_float32 = raw_depth.astype(np.float32)

# Calculate the gradient (Sobel operators for X and Y directions)
sobelx = cv2.Sobel(raw_depth_float32, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(raw_depth_float32, cv2.CV_32F, 0, 1, ksize=3)

# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

# Create binary gradient image: 1 where magnitude > 0, otherwise 0
binary_gradient = np.zeros_like(gradient_magnitude)
binary_gradient[gradient_magnitude > 0.1] = 1

# Convert binary gradient to uint8 (0-255) for visualization
binary_gradient_vis = (binary_gradient * 255).astype(np.uint8)

# Normal gradient for comparison
gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Convert the grayscale images to BGR for display
gradient_bgr = cv2.cvtColor(gradient_norm, cv2.COLOR_GRAY2BGR)
binary_gradient_bgr = cv2.cvtColor(binary_gradient_vis, cv2.COLOR_GRAY2BGR)

# Create a copy of the binary gradient for drawing bounding boxes
boxed_gradient = binary_gradient_bgr.copy()

# Find contours in the binary image
contours, hierarchy = cv2.findContours(binary_gradient_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size to avoid tiny boxes (adjust min_area as needed)
min_area = 10000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

print(f"Found {len(contours)} contours, {len(filtered_contours)} after filtering")

# Draw bounding boxes on the binary gradient image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(boxed_gradient, (x, y), (x + w, y + h), (0, 255, 0), 20)

# Create the visualization
x1, y1, x2, y2 = 2500, 800, 3000, 1400 # COW
x1, y1, x2, y2 = 1100, 800, 1700, 1400
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 50)
cv2.rectangle(depth_to_plot, (x1, y1), (x2, y2), (255, 0, 0), 50)
cv2.rectangle(gradient_bgr, (x1, y1), (x2, y2), (255, 0, 0), 50)
cv2.rectangle(boxed_gradient, (x1, y1), (x2, y2), (255, 0, 0), 50)


top_row_frame = cv2.hconcat([image, depth_to_plot])
bottom_row_frame = cv2.hconcat([gradient_bgr, boxed_gradient])
final_output = cv2.vconcat([top_row_frame, bottom_row_frame])



# Extract the gradient magnitude in the bounding box
roi_gradient = gradient_magnitude[y1:y2, x1:x2]

# Compute variance and standard deviation
variance = np.var(roi_gradient)
std_dev = np.std(roi_gradient)

# Print results
print(f"Gradient Variance in bounding box: {variance:.4f}")
print(f"Gradient Std Dev in bounding box: {std_dev:.4f}")
# Set a threshold for detecting low gradient variation
low_variation_threshold = 0.5  # Adjust based on empirical testing

if variance < low_variation_threshold:
    print("Gradient magnitude does not change significantly in this region.")
else:
    print("Gradient magnitude varies significantly in this region.")


# Display the results
cv2.imshow("Depth Analysis with Bounding Boxes", final_output)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Close the window