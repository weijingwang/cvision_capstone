import yaml
import json
from ultralytics import YOLO
from torchvision.datasets import CocoDetection
from get_npz_output import get_npz_output

def output_yolo_json(name):
	with open('/home/vikramiyer/ultralytics/ultralytics/cfg/datasets/coco.yaml', 'r') as file:
		coco_data = yaml.safe_load(file)

	class_labels = coco_data['names']


	model = YOLO('yolo11n.pt')
	result = model(f'/home/vikramiyer/ml-depth-pro/images/{name}.jpg')

	output = []
	for res in result:
		for box in res.boxes:
			class_index = int(box.cls)
			data = {
				"class": class_labels[class_index],
				"confidence": float(box.conf),
				"box": [float(coord) for coord in box.xyxy[0].tolist()]
			}
			output.append(data)

	with open(f'/home/vikramiyer/runoutput/{name}_output_with_labels.json', 'w') as f:
		json.dump(output, f, indent=4)

	print("Output saved to my directory") # will later edit this to work on anyone's local directory


def find_centers(name):
	with open(f'/home/vikramiyer/runoutput/{name}_output_with_labels.json') as f:
		data = json.load(f)

	centers = []
	for item in data:
		box = item['box']
		center_x = int((box[0] + box[2]) / 2)
		center_y = int((box[1] + box[3]) / 2)
		centers.append({"center_x": center_x, "center_y": center_y})

	# Save the centers to a new JSON file
	output_file = f'/home/vikramiyer/runoutput/{name}_output_with_labels.json'.replace('.json', '_centers.json')
	with open(output_file, 'w') as f:
		json.dump(centers, f, indent=4)

	print(f"Centers saved to {output_file}")

def extract_depth(json_name):
    with open(f'/home/vikramiyer/runoutput/{json_name}_output_with_labels_centers.json') as f:
        centers = json.load(f)

    depth_map = get_npz_output(json_name)

    depth_values = []
    for center in centers:
        center_x = center['center_x']
        center_y = center['center_y']
        
        # Use center coordinates as indices in the depth map
        depth = float(depth_map[center_y, center_x])
        depth_values.append({
            "center_x": center_x,
            "center_y": center_y,
            "depth": depth
        })
    
    # Save the extracted depth values to a new JSON file
    output_file = f'/home/vikramiyer/runoutput/{json_name}_output_with_labels_centers.json'.replace('_centers.json', '_depth_values.json')
    with open(output_file, 'w') as f:
        json.dump(depth_values, f, indent=4)
    
    print(f"Depth values saved to {output_file}")