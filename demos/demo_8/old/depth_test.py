import matplotlib.pyplot as plt
import matplotlib
import cv2
from depth_map import *

# Initialize depth map
depth_init = init_depth()
args = depth_init[0]
depth_anything = depth_init[1]
print("Loaded depthmap...")

# Initialize webcam
cmap = matplotlib.colormaps.get_cmap('gray')

raw_image = cv2.imread('./images/IMG_7018.jpg')
# Depth math and get depth map to render
raw_depth = depth_anything.infer_image(raw_image, args.input_size)

print(raw_depth)

# Plotting the raw depth map alongside the original image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot original image
axes[0].imshow(raw_image)  # Convert BGR to RGB
axes[0].set_title("Original Image")
axes[0].axis("off")  # Hide axes

# Plot depth map
axes[1].imshow(raw_depth, cmap=cmap)
axes[1].set_title("Raw Depth Map")
axes[1].axis("off")  # Hide axes

# Show the plot
plt.tight_layout()
plt.show()
