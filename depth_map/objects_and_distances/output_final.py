import json


def output_final(name):
    # Load both JSON files
    with open(f'/home/vikramiyer/runoutput/{name}_output_with_labels.json', 'r') as f:
        objects_data = json.load(f)

    with open(f'/home/vikramiyer/runoutput/{name}_output_with_labels_depth_values.json', 'r') as f:
        depth_data = json.load(f)

    with open(f'/home/vikramiyer/runoutput/{name}_objects_and_distances.txt', 'w') as output_file:
        for obj, depth_info in zip(objects_data, depth_data):
            class_name = obj['class']
            depth = depth_info['depth']
            
            output_file.write(f"There is a {class_name} {depth} meters away\n")

    print("Output saved to output.txt")