from ultralytics import YOLO
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.chdir("/home/jordanprescott/shiv_capstone/depth_map/depth_of_object")

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

def get_json(image_path):
    # Load a pretrained YOLOv5 model
    model = YOLO('./misc/yolov5s.pt')  # Specify the correct model file path

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
    # plot_yolo_results(image_path, detections)

    # Return the detections as JSON
    return json.dumps(detections)
