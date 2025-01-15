from PIL import Image
import numpy as np
import json
import torch
import ml_depth_pro.src.depth_pro as depth_pro
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from ultralytics import YOLO



def plot_yolo_results(image_path, detections):
    # Load the image
    image = Image.open(image_path)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the image
    ax.imshow(image)

    # Iterate over the detections and draw bounding boxes
    for detection in detections:
        label = detection["label"]
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        
        # Add the patch to the axes
        ax.add_patch(rect)
        
        # Add a label and confidence text
        ax.text(
            bbox[0], bbox[1] - 10, f'{label}: {confidence:.2f}',
            fontsize=12, color='red', weight='bold'
        )

    # Set title and axis off
    ax.set_title('YOLO Detection Results')
    ax.axis('off')

    # Show the plot
    plt.show()


def get_yolo_json(image_path):
    # Load a pretrained YOLOv5 model
    model = YOLO('./misc/yolov5su.pt')  # Specify the correct model file path

    # Perform inference on the image
    results = model(image_path)  # This returns a list of Results objects for each image

    # Extract the detected objects and their bounding boxes
    detections = []

    # Get the results for the first image (index 0)
    result = results[0]  # Access the first result (you can loop through if multiple images)

    # Extract bounding boxes (boxes) and labels (names)
    boxes = result.boxes.xyxy  # Bounding boxes in xyxy format (xmin, ymin, xmax, ymax)
    labels = result.names  # Mapping of class IDs to class names
    confidences = result.boxes.conf  # Confidence scores for each detection

    # Iterate through the boxes and create a list of detections
    for i, box in enumerate(boxes):
        label = labels[int(result.boxes.cls[i])]  # Get the class label for the current detection
        confidence = confidences[i]  # Get the confidence score for the current detection
        xmin, ymin, xmax, ymax = box  # Extract the bounding box coordinates

        # Create a dictionary with label and bounding box for each detected object
        detections.append({
            "label": label,
            "confidence": float(confidence),  # Convert confidence to float for JSON compatibility
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
        })
    
    # Plot the results
    plot_yolo_results(image_path, detections)

    # Return the detections as JSON
    return json.dumps(detections)


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
    yolo_output_json = get_yolo_json(im_path)

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

    # Specific depth for which we will see if there are any objects
    specific_depths = [1, 5, 10, 100]
    results = []

    # for each specifc depth we want to check
    for sd in specific_depths:
        print(f"Depth threshold: {sd}")
        # for each object that we detected in the normal image
        for label, bbox, angle in bounding_boxes[:]:
            sdm = get_map_of_specific_depth(depth, sd) # specific depth map
            avg_depth = average_depth_over_bbox(sdm, bbox) # average depth over the box
            if not np.isnan(avg_depth):
                # if there is any depth of the object within the box
                # remove the object so it is not detected twice
                results.append((label, avg_depth, angle, (label, bbox)))
                bounding_boxes.remove((label, bbox, angle))
                print(f"Object: {label}, Distance: {avg_depth:.2f}, Angle: {angle}")
            
    print(f"Remaining unprocessed objects: {len(bounding_boxes)}")

    objects = [obj for obj, _, _, _ in results]
    distances = [distance for _, distance, _, _ in results]
    angles = [angle for _, _, angle, _ in results]
    result_bboxes = [bbox for _, _, _, bbox in results]

    # use constant filter parameters passed from pt.py
    filtered_objects, filtered_distances, filtered_positions = filter_results(objects, distances, angles, distance_threshold, normalized_angle_threshold)

    # Plot the filtered depth map with filtered bounding boxes
    plot_filtered_depth_map_with_bboxes(filtered_depth_map, filtered_objects, filtered_positions, filtered_distances, result_bboxes)

    return filtered_objects, filtered_distances, filtered_positions
