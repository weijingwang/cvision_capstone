import os
from ml_depth_pro.src.depth_pro import depth_pro
from functions import get_oda
from new_audio import text_to_speech_proximity_spatial
from PIL import Image

# Constants
MISC_DIR = "./misc"  # Directory containing the images
DISTANCE_THRESHOLD = 10  # 10 meters
ANGLE_THRESHOLD = 180  # Angle from the center that is desired
NORMALIZED_ANGLE_THRESHOLD = ANGLE_THRESHOLD / 180

# Load the model and transform
print("Loading model...")
model, transform = depth_pro.create_model_and_transforms(device="cuda")
model.eval()
print("Model loaded successfully.")

def process_image(image_name):
    """
    Process the image and generate text-to-speech output.

    Args:
        image_name (str): Name of the image file.
    """
    image_path = os.path.join(MISC_DIR, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print(f"Processing image: {image_path}")

    # Get objects, distances, and angles
    objects, distances, angles = get_oda(
        image_path,
        DISTANCE_THRESHOLD,
        NORMALIZED_ANGLE_THRESHOLD,
        model,
        transform
    )
    
    # Pass the output to the text-to-speech function
    text_to_speech_proximity_spatial(objects, distances, angles)
    print("Processing completed.\n")

# Interactive loop for processing images
print("Enter image names to process (from 'misc' folder), or type 'exit' to quit.")
while True:
    image_name = input("Image name: ").strip()
    if image_name.lower() == 'exit':
        print("Exiting...")
        break
    if not image_name:
        print("Please provide a valid image name.")
        continue
    try:
        process_image(image_name)
    except Exception as e:
        print(f"Error processing image '{image_name}': {e}")
