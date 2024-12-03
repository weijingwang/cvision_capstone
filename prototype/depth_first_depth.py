from PIL import Image
import numpy as np
import json
from get_yolo_json import get_json
import ml_depth_pro.src.depth_pro as depth_pro



def get_depth_map(image_path):

    print("starting depth map retrieval")

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px, = depth_pro.load_rgb(image_path)
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    depth_array = np.array(depth)

    print("success")

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
        relative_angle = ((x_max - x_min)/2 + x_min) / im_shape[1]  # Relative position in image width
        relative_angle = 2 * relative_angle - 1
        bboxes.append((label, (x_min, y_min, x_max, y_max), relative_angle))
    return bboxes



def filter_results(objects, distances, positions, distance_threshold, angle_threshold):
    """
    Filters the objects, distances, positions, and importance based on given thresholds.

    Parameters:
    - objects: List of detected object labels.
    - distances: List of distances corresponding to each object.
    - positions: List of relative angles (positions) corresponding to each object.
    - importance: List of importance values corresponding to each object.
    - distance_threshold: Maximum allowed distance for filtering.
    - angle_threshold: Maximum allowed angle deviation for filtering.
    - importance_threshold: Minimum importance value for filtering.

    Returns:
    - Filtered objects, distances, positions, and importance as lists.
    """
    
    obj_list = ["car", "person", "tree"]
    
    filtered_objects = []
    filtered_distances = []
    filtered_positions = []

    for obj, dist, angle in zip(objects, distances, positions):
        # if dist <= distance_threshold and abs(angle) <= angle_threshold and obj in obj_list:
        if dist <= distance_threshold and abs(angle) <= angle_threshold:
            filtered_objects.append(obj)
            filtered_distances.append(dist)
            filtered_positions.append(angle)

    return filtered_objects, filtered_distances, filtered_positions




def get_oda(im_path: str, distance_threshold: float, normalized_angle_threshold: float):

    # Get depth map
    print("Getting depth map...")
    depth, _ = get_depth_map(im_path)

    # Get YOLO detections
    print("Getting YOLO detections...")
    yolo_output_json = get_json(im_path)

    # Load the image as grayscale
    image = Image.open(im_path).convert("L")
    # Convert YOLO output to bounding boxes
    bounding_boxes = get_bboxes(yolo_output_json, np.array(image).shape)

    print(f"Number of detected objects: {len(bounding_boxes)}")

    # List of specific depth thresholds
    specific_depths = [1, 5, 10, 100]
    results = []

    for sd in specific_depths:
        print(f"Depth threshold: {sd}")
        for label, bbox, angle in bounding_boxes[:]:
            sdm = get_map_of_specific_depth(depth, sd)
            avg_depth = average_depth_over_bbox(sdm, bbox)
            if not np.isnan(avg_depth):
                results.append((label, avg_depth, angle))
                bounding_boxes.remove((label, bbox, angle))
                print(f"Object: {label}, Distance: {avg_depth:.2f}, Angle: {angle}")
                
    print(f"Remaining unprocessed objects: {len(bounding_boxes)}")

    # Prepare data for text-to-speech function
    objects = [obj for obj, _, _ in results]
    distances = [distance for _, distance, _ in results]
    angles = [angle for _, _, angle in results]
    
    filtered_objects, filtered_distances, filtered_positions = filter_results(objects, distances, angles, distance_threshold, normalized_angle_threshold)

    return filtered_objects, filtered_distances, filtered_positions
