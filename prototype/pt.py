import os
import cv2
import time
import tempfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Enable interactive mode
# plt.ion()

# Your existing module imports
from ml_depth_pro.src.depth_pro import depth_pro
from functions import get_oda
from new_audio import text_to_speech_proximity_spatial
import imageio
import os
import imageio
from PIL import Image

# Constants
MISC_DIR = "./misc"  # Directory containing images or where you store temp frames, etc.
DISTANCE_THRESHOLD = 100  # in meters
ANGLE_THRESHOLD = 180    # Angle from the center that is desired
NORMALIZED_ANGLE_THRESHOLD = ANGLE_THRESHOLD / 180.0

# Load the model and transform (only once)
print("Loading model...")
model, transform = depth_pro.create_model_and_transforms(device="cuda")
model.eval()
print("Model loaded successfully.\n")

def process_image(image_name):
    """
    Process a single image and generate text-to-speech output.
    """
    image_path = os.path.join(MISC_DIR, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    try:
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

    except Exception as e:
        print(f"Error processing image '{image_name}': {e}")

def process_video(video_name):
    """
    Process a video by reading frames in 'real-time' style:
    - Read a frame
    - Perform inference (depth + object detection)
    - Calculate how long inference took
    - Skip the appropriate number of frames based on that processing time
    - Continue until video ends or user interrupts
    """
    video_path = os.path.join(MISC_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video file: {video_name}")
        return
    
    # Get frame rate from the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Could not read FPS from video, defaulting to 30.")
        fps = 30.0
    frame_duration = 1.0 / fps  # time per frame (sec)

    current_frame_index = 0

    while True:
        start_time = time.time()

        # Jump to the frame we want to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            print("No more frames or unable to read the frame.")
            break

        # Convert the frame (BGR from OpenCV) to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # We'll temporarily save to a file, since `get_oda` expects a path
        # (Alternatively, you could refactor `get_oda` to accept in-memory images.)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            # Save the current frame to disk
            Image.fromarray(frame_rgb).save(tmp_file.name)
            temp_frame_path = tmp_file.name
        
        save_path = os.path.join(MISC_DIR, "gif", f"frame_{current_frame_index}.jpg")
        print(save_path)

        # Now run the usual pipeline on this temp image file
        try:
            objects, distances, angles = get_oda(
                temp_frame_path,
                DISTANCE_THRESHOLD,
                NORMALIZED_ANGLE_THRESHOLD,
                model,
                transform,
                save_path,
            )
            # text_to_speech_proximity_spatial(objects, distances, angles) # commented for debugging
        except Exception as e:
            print(f"Error processing video frame {current_frame_index}: {e}")
        
        # Clean up the temp file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

        # Measure how long processing took
        processing_time = time.time() - start_time

        # Calculate how many frames correspond to that processing time
        frames_to_skip = int(processing_time // frame_duration)
        # Move to the next frame index
        current_frame_index += (1 + frames_to_skip)

        # If we have gone beyond the total frames, break
        if current_frame_index >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

        # Optional: Print progress
        print(f"Processed frame {current_frame_index - (1 + frames_to_skip)}; skipping {frames_to_skip} frames.")

    cap.release()

    

    def create_gif():
        gif_folder = os.path.join(MISC_DIR, "gif")
        gif_images = [image for image in os.listdir(gif_folder) if image.endswith(".jpg")]
        gif_images.sort()  # Sort the images in ascending order

        gif_path = os.path.join(MISC_DIR, "output.gif")

        # Create temporary GIF without loop (ImageIO)
        temp_gif_path = os.path.join(MISC_DIR, "temp.gif")
        
        with imageio.get_writer(temp_gif_path, mode="I") as writer:
            for image_name in gif_images:
                image_path = os.path.join(gif_folder, image_name)
                image = imageio.imread(image_path)
                
                # Append multiple times to slow down the GIF
                for _ in range(12):  # Repeat each frame 12 times
                    writer.append_data(image)

        # Use PIL to add infinite looping
        images = [Image.open(os.path.join(gif_folder, img)) for img in gif_images]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],  # Append all other frames
            duration=200,  # Adjust speed (200ms per frame)
            loop=0  # This ensures an infinite loop
        )

        print(f"GIF created successfully at: {gif_path}")



    # Call the create_gif function after processing the video
    create_gif()
    
    print("Finished processing video.\n")

# Interactive loop
def main():
    print("Enter 'image <filename>' to process an image, 'video <filename>' to process a video, or 'exit' to quit.")
    while True:
        user_input = input("Command: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        tokens = user_input.split(maxsplit=1)
        if len(tokens) < 2:
            print("Invalid command. Please use 'image <filename>' or 'video <filename>'.")
            continue

        command, filename = tokens[0], tokens[1]
        
        if command.lower() == "image":
            process_image(filename)
        elif command.lower() == "video":
            process_video(filename)
        else:
            print("Unrecognized command. Please type 'image <filename>', 'video <filename>', or 'exit'.")

if __name__ == "__main__":
    main()
