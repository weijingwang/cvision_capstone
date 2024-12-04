from PIL import Image
import numpy as np
import json
from get_yolo_json import get_json
import torch
import ml_depth_pro.src.depth_pro as depth_pro

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches




def plot_filtered_depth_map_with_bboxes(filtered_depth_map, filtered_objects, filtered_positions, filtered_distances, bboxes):
    """
    Plots the filtered depth map with bounding boxes overlaid for filtered objects.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(filtered_depth_map, cmap='inferno')
    plt.colorbar(label="Depth (m)")

    # Add bounding boxes and labels
    for obj, dist, angle, (label, bbox) in zip(filtered_objects, filtered_distances, filtered_positions, bboxes):
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='cyan', facecolor='none'
        )
        plt.gca().add_patch(rect)

        # Annotate the bounding box with the object label and distance
        label_text = f"{obj}\nDist: {dist:.2f}m\nAngle: {angle:.2f}"
        plt.text(x_min, y_min - 10, label_text, color='cyan', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    plt.title("Filtered Depth Map with Bounding Boxes")
    plt.axis('off')
    plt.show()



def plot_depth_map(depth_map, threshold=None):
    """
    Plots the depth map using matplotlib with inverted colormap.
    """
    inverted_cmap = cm.get_cmap('inferno').reversed()

    filtered_depth_map = depth_map.copy()
    filtered_depth_map[depth_map >= threshold] = 0

    plt.figure(figsize=(10, 10))
    plt.imshow(filtered_depth_map, cmap='inferno')
    plt.colorbar()
    plt.title("Filtered Depth Map")
    plt.show()


def get_depth_map(image_path, model, transform):
    """
    Retrieves the depth map and focal length using the preloaded model and transform.
    """
    print("Starting depth map retrieval...")

    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    focallength_px = prediction["focallength_px"]  # Focal length in pixels
    depth_array = depth.cpu().numpy()

    print("Depth map retrieval successful.")

    return depth_array, focallength_px


def get_map_of_specific_depth(depth_map, specific_depth):
    filter_depth_map = depth_map.copy()
    filter_depth_map[depth_map > specific_depth] = 0
    return filter_depth_map


def average_depth_over_bbox(specific_depth_map: np.ndarray, bbox: tuple):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_sdm = specific_depth_map[y_min:y_max, x_min:x_max]
    nonzero = cropped_sdm[cropped_sdm != 0]

    avg_depth = np.mean(nonzero) if nonzero.size > 0 else np.nan
    return avg_depth


def get_bboxes(yolo_output_json, im_shape):
    if isinstance(yolo_output_json, str):
        yolo_output_json = json.loads(yolo_output_json)

    bboxes = []
    for detection in yolo_output_json:
        label = detection["label"]
        x_min, y_min, x_max, y_max = map(int, detection['bbox'])
        relative_angle = ((x_max - x_min) / 2 + x_min) / im_shape[1]  # Relative position in image width
        relative_angle = 2 * relative_angle - 1
        bboxes.append((label, (x_min, y_min, x_max, y_max), relative_angle))
    return bboxes


def filter_results(objects, distances, positions, distance_threshold, angle_threshold):
    """
    Filters the objects, distances, and positions based on thresholds.
    """
    filtered_objects = []
    filtered_distances = []
    filtered_positions = []

    for obj, dist, angle in zip(objects, distances, positions):
        if dist <= distance_threshold and abs(angle) <= angle_threshold:
            filtered_objects.append(obj)
            filtered_distances.append(dist)
            filtered_positions.append(angle)

    return filtered_objects, filtered_distances, filtered_positions


def get_oda(im_path: str, distance_threshold: float, normalized_angle_threshold: float, model, transform):
    """
    Main function to get objects, distances, and angles based on depth and YOLO detections.
    """
    # Display the image as a figure
    plt.imshow(plt.imread(im_path))
    plt.axis('off')
    plt.show()


    # Get YOLO detections
    print("Getting YOLO detections...")
    yolo_output_json = get_json(im_path)

    # Get bounding boxes
    image = Image.open(im_path).convert("RGB")
    bounding_boxes = get_bboxes(yolo_output_json, np.array(image).shape)

    print(f"Number of detected objects: {len(bounding_boxes)}")


    # Get depth map
    print("Getting depth map...")
    depth, _ = get_depth_map(im_path, model, transform)

    # Create a filtered depth map based on the distance threshold
    filtered_depth_map = depth.copy()
    filtered_depth_map[depth >= distance_threshold] = 0

    # Plot the filtered depth map
    plot_depth_map(filtered_depth_map, distance_threshold)

    
    specific_depths = [1, 5, 10, 100]
    results = []

    for sd in specific_depths:
        print(f"Depth threshold: {sd}")
        for label, bbox, angle in bounding_boxes[:]:
            sdm = get_map_of_specific_depth(depth, sd)
            avg_depth = average_depth_over_bbox(sdm, bbox)
            if not np.isnan(avg_depth):
                results.append((label, avg_depth, angle, (label, bbox)))
                bounding_boxes.remove((label, bbox, angle))
                print(f"Object: {label}, Distance: {avg_depth:.2f}, Angle: {angle}")
            
    print(f"Remaining unprocessed objects: {len(bounding_boxes)}")

    objects = [obj for obj, _, _, _ in results]
    distances = [distance for _, distance, _, _ in results]
    angles = [angle for _, _, angle, _ in results]
    result_bboxes = [bbox for _, _, _, bbox in results]

    filtered_objects, filtered_distances, filtered_positions = filter_results(objects, distances, angles, distance_threshold, normalized_angle_threshold)

    # Plot the filtered depth map with filtered bounding boxes
    plot_filtered_depth_map_with_bboxes(filtered_depth_map, filtered_objects, filtered_positions, filtered_distances, result_bboxes)

    return filtered_objects, filtered_distances, filtered_positions
