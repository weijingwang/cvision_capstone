from PIL import Image
import depth_pro
import get_npz_output
import numpy as np
import os
import torch

def get_depth_map(name):
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px, = depth_pro.load_rgb(f'/home/vikramiyer/ml-depth-pro/images/{name}.jpg')
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    npz_filename = f'/home/vikramiyer/ml-depth-pro/output/phelps_1210/{name}_depth_map.npz'
    np.savez(npz_filename, depth=depth, focallength_px=focallength_px)

    print(f"Depth map saved as {npz_filename}")
