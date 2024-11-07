import json
import numpy as np

def get_npz_output(name):
    # Load the .npz file
    npz_file = f'/home/vikramiyer/ml-depth-pro/output/phelps_1210/{name}_depth_map.npz'
    data = np.load(npz_file)
    image_array = data['depth']  # Use the array key (e.g., 'arr_0') if you're unsure

    return image_array

