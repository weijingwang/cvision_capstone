from PIL import Image
import numpy as np
import os
import json
from get_yolo_json import get_json
# from sam_tests.FastSAM.fastsam import FastSAM, FastSAMPrompt
# from sam_tests.FastSAM.fastsam_test import test


os.chdir("/home/jordanprescott/shiv_capstone/depth_map/depth_of_object")



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
        # print(detection)
        label = detection["label"]
        x_min, y_min, x_max, y_max = map(int, detection['bbox'])
        relative_angle = ((x_max - x_min)/2 + x_min) / im_shape[1] # use [1] because that is the number of cols in the image, which is x
        bboxes.append((label, (x_min, y_min, x_max, y_max), relative_angle))
    return bboxes




def main():
    input_path = "./misc/smaller_cars.png"
    image = Image.open(input_path).convert("L") # do this so its a grayscale image and only has two dimensions

    # Load depth map
    data = np.load('./misc/resized_out.npz')
    depth = data[data.files[0]]

    # Get YOLO detections
    yolo_output_json = get_json(input_path)

    # Convert YOLO output to bounding boxes
    bounding_boxes = get_bboxes(yolo_output_json, np.array(image).shape)
    
    print('boxes')
    # print(bounding_boxes)

    # Print the number of detected objects
    print(f"Number of detected objects: {len(bounding_boxes)}")

    # List of specific depth thresholds
    specific_depths = [1, 5, 10, 100]
    results = []

    for sd in specific_depths:
        print(f"Depth threshold: {sd}")
        # Use a copy of the bounding_boxes list to avoid modifying the list while iterating
        for label, bbox, angle in bounding_boxes[:]:  # Use slicing to iterate over a copy of the list
            sdm = get_map_of_specific_depth(depth, sd)
            avg_depth = average_depth_over_bbox(sdm, bbox)
            if not np.isnan(avg_depth):
                results.append((label, avg_depth, angle))
                bounding_boxes.remove((label, bbox, angle))  # Remove the processed object
                print(f"Object: {label}, Distance: {avg_depth:.2f}, Angle: {angle}")
                
    
    # Print remaining objects that did not get a depth associated
    print(f"Remaining unprocessed objects: {len(bounding_boxes)}")
    
    # Convert the list of tuples into a list of dictionaries
    data_dict = [{"object": str(obj), "distance": float(distance), "angle": float(angle)} for obj, distance, angle in results]

    # Convert the list of dictionaries to a JSON string
    json_data = json.dumps(data_dict, indent=4)

    # Print the JSON formatted string
    print(json_data)
    
    return json_data


main()