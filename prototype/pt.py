import sys
import os
from depth_first_depth import get_oda
from new_audio import text_to_speech_proximity_spatial

# Paths to the image and depth map
image_path = "./misc/smaller_cars.png"
depth_map_path = "./misc/resized_out.npz"

# thresholds
distance_threshold = 5 # 5m
angle_threshold = 180 # angle from center that is desired
normalized_angle_threshold = angle_threshold / 180

# Get objects, distances, angles, and importance
objects, distances, angles = get_oda(image_path, distance_threshold, normalized_angle_threshold)

# Pass the output to the text-to-speech function
text_to_speech_proximity_spatial(objects, distances, angles)
