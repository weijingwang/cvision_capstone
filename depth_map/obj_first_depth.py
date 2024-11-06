from PIL import Image
from rembg import remove
import numpy as np
from numpy import load




def average_depth_over_bbox(binary_map, depth_map):
	masked = np.multiply(depth_map,binary_map)
	nonzero = masked[masked != 0]
	print("Approximate depth of object:", np.mean(nonzero))






def process_image_with_bbox(image: Image.Image, bbox: tuple, remove_background=True):
	"""
	Processes an image by extracting a region using a bounding box,
	removing the background, then zero-padding it to match the original size,
	and generates a binary bitmap.

	Args:
		image (Image.Image): Input image.
		bbox (tuple): Bounding box (x_min, y_min, x_max, y_max).
		remove_background (bool): Whether to apply background removal or not.

	Returns:
		np.ndarray: Binary bitmap (numpy array).
	"""

	# Extract bounding box region
	x_min, y_min, x_max, y_max = bbox
	cropped_image = image.crop((x_min, y_min, x_max, y_max))

	# Remove the background from the cropped image
	if remove_background:
		cropped_image = remove(cropped_image)


	# Create a zero-padded image with the same size as the original
	padded_image = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Transparent background
	padded_image.paste(cropped_image, (x_min, y_min))


	# Convert the image to grayscale
	grayscale_image = padded_image.convert("L")

	# Convert the image to a numpy array
	grayscale_array = np.array(grayscale_image)

	# Set all non-zero pixels to 255 (binary map preparation)
	grayscale_array[grayscale_array > 0] = 255

	# Create the binary map (bit map)
	binary_map = (grayscale_array / 255).astype(bool)

	# Return the binary map
	return binary_map


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

	# Process each bounding box through the image processing function
	for bbox in bounding_boxes:
		bitmap = process_image_with_bbox(image, bbox)
		average_depth_over_bbox(bitmap, depth)

