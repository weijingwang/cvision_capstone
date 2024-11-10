import torch
from fastsam import FastSAM
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


print(sys.path)

os.chdir("/home/jordanprescott/shiv_capstone/depth_map/depth_of_object")

# Load the image you want to segment
image_path = "./misc/test_car_street.png"
image = Image.open(image_path)

# Initialize the FastSAM model
model = FastSAM()

# Convert image to a tensor and make it compatible with the model
image_tensor = np.array(image)
image_tensor = torch.tensor(image_tensor).unsqueeze(0).permute(0, 3, 1, 2).float()

# Run segmentation
segmentation = model(image_tensor)

# Get the segmentation masks
masks = segmentation['masks']

# Visualize the segmentation result
plt.imshow(masks[0].cpu().numpy(), cmap='jet')  # Displaying the first mask
plt.show()
