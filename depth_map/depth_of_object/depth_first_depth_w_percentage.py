from PIL import Image
from rembg import remove
import numpy as np
from numpy import load



def get_map_of_specific_depth(depth_map, specific_depth):
	filter_depth_map = depth_map.copy()
	filter_depth_map[depth_map > specific_depth] = 0
	return filter_depth_map




def average_depth_over_bbox(specific_depth_map: np.ndarray, bbox: tuple, percent_fill_min: float):
	# Extract bounding box region
	x_min, y_min, x_max, y_max = bbox
	cropped_sdm = specific_depth_map[y_min:y_max, x_min:x_max]
	nonzero = cropped_sdm[cropped_sdm != 0]
	# get the number of nonzero pixels and the number of total pixels,
	# if the percentage that the box is filled is greater than the threshold
	# then return the depth, else return None
	print(len(nonzero))
	print(len(nonzero) / ((x_max - x_min) * (y_max - y_min)))
	if len(nonzero) / ((x_max - x_min) * (y_max - y_min)) > percent_fill_min:

		avg_depth = np.mean(nonzero)
	else:
		print("did not clear ")
		avg_depth = None

	print(avg_depth)
	return avg_depth



def yolo_json_to_bboxes(yolo_output_json):
	"""
	Converts YOLO JSON output into a list of bounding boxes.

	Args:
		yolo_output_json (list): List of YOLO detections in JSON format.
			Each detection should contain 'bounding_box' with 'x', 'y', 'width', and 'height' fields.

	Returns:
		list: List of bounding boxes in the form (x_min, y_min, x_max, y_max).
	"""
	bboxes = []

	for detection in yolo_output_json:
		# Extract bounding box information
		x_min = detection['bounding_box']['x']
		y_min = detection['bounding_box']['y']
		width = detection['bounding_box']['width']
		height = detection['bounding_box']['height']

		# Calculate x_max and y_max
		x_max = x_min + width
		y_max = y_min + height

		# Append bounding box in (x_min, y_min, x_max, y_max) format
		bboxes.append((x_min, y_min, x_max, y_max))

	return bboxes








# Example usage:
if __name__ == "__main__":
	# Load the input image
	input_path = "test_car_street.png"  # The path of the uploaded image
	image = Image.open(input_path)

	data = np.load('out.npz')
	lst = data.files
	depth = data[lst[0]]
	# Image.fromarray(depth).show()

	# Example YOLO JSON output for the yellow car
	yolo_output_json = [
		{
			"class_id": 2,
			"class_label": "car",
			"confidence": 0.99,
			"bounding_box": {
				"x": 380,
				"y": 225,
				"width": 450,
				"height": 450
			}
		}
	]
	# Convert YOLO output to bounding boxes
	bounding_boxes = yolo_json_to_bboxes(yolo_output_json)

	specific_depths = [0, 7]
	for sd in specific_depths:
		for bbox in bounding_boxes:
			sdm = get_map_of_specific_depth(depth, sd)
			avg_dep = average_depth_over_bbox(sdm, bbox, 0.7)
			if avg_dep:
				bounding_boxes.pop(bbox)







