import pygame
import numpy as np
import threading

import cv2
import time
from ultralytics import YOLO

# Initialize global variables for frequency, volume, and panning
frequency = 440.0  # Default frequency in Hz (A4)
volume = 0.5       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)
sample_rate = 44100
duration = 1  # Short buffer duration for real-time updates
sound = None

# Initialize Pygame mixer
pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)

# Init YOLO stuff
# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)

# Open the video file or webcam (use 0 for the default webcam)
video_source = "sitar video - 1st class - 1-7-25.mov"  # Replace with 0 for a webcam feed
cap = cv2.VideoCapture(0)  # Replace with `video_source` if using a file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()


def generate_sine_wave(frequency, volume, panning, duration=0.1):
    """Generate a stereo sine wave with the given parameters."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t) * volume

    # Apply panning
    left = wave * (1 - panning)
    right = wave * panning

    # Combine into stereo
    stereo_wave = np.column_stack((left, right))
    return (stereo_wave * 32767).astype(np.int16)


def input_listener(): # like an interrupt for key detection
    """Thread to listen for user input and update parameters."""
    global frequency, volume, panning
    while True:
        try:
            user_input = input("Enter 'f <freq>' for frequency, 'v <vol>' for volume, 'p <pan>' for panning: ")
            cmd, value = user_input.split()
            value = float(value)
            if cmd == 'f':
                frequency = max(20.0, min(value, 20000.0))  # Limit frequency range
                print(f"Frequency set to {frequency} Hz")
            elif cmd == 'v':
                volume = max(0.0, min(value, 1.0))  # Limit volume range
                print(f"Volume set to {volume}")
            elif cmd == 'p':
                panning = max(0.0, min(value, 1.0))  # Limit panning range
                print(f"Panning set to {panning} (0.0 = left, 1.0 = right)")
            else:
                print("Invalid command. Use 'f', 'v', or 'p'.")
        except Exception as e:
            print(f"Error: {e}. Use format 'f <freq>', 'v <vol>', or 'p <pan>'.")

# Start the input listener thread
thread = threading.Thread(target=input_listener, daemon=True)
thread.start() # like interrupt, run key detection while loop in background

print("Sine wave generator started. Adjust frequency, volume, and panning in real time.")
print("Commands: 'f <freq>' (frequency), 'v <vol>' (volume), 'p <pan>' (panning)")

# FIX
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break
    


    # Run YOLOv8 inference on the frame
    results = model(frame, verbose=False)

    # Track if a person is detected and find the position of the red circle
    person_detected = False
    red_circle_position = 0  # Default position if no person is detected

    for detection in results[0].boxes.data:
        x_min, y_min, x_max, y_max = map(float, detection[:4])  # Bounding box coordinates
        class_id = int(detection[5])  # Class ID

        if model.names[class_id] == "person":
            person_detected = True
            # Calculate the center of the bounding box
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)

            # Draw a red circle at the center of the bounding box
            cv2.circle(frame, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=-1)

            # Track the horizontal position of the red circle (panning position)
            red_circle_position = x_center / frame.shape[1]  # Normalize to [0, 1]

    panning = max(0.0, min(red_circle_position, 1.0))  # Limit panning range
    print(f"Panning set to {panning} (0.0 = left, 1.0 = right)")


    # Sound generate
    wave = generate_sine_wave(frequency, volume, panning, duration)
    
    #if there is a person on screen, play sound
    if person_detected:

        if sound is None:
            sound = pygame.sndarray.make_sound(wave)
        else:
            sound.stop()
            sound = pygame.sndarray.make_sound(wave)
        
        sound.play(loops=0)


    # Map red_circle_position to the panning range [0, 1]
    # 0 is left, 1 is right, and values in between correspond to positions in between
    # if person_detected:
    #     tone.play(red_circle_position)  # Pan based on the position of the red circle
    # else:
    #     tone.stop()  # Stop the tone if no person is detected

    # Display the frame with annotations
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pygame.time.wait(int(duration * 1000))

# Release resources
cap.release()
cv2.destroyAllWindows()
# tone.stop()  # Ensure the tone stops when the program exits



# # Play the sine wave continuously
# try:
#     play_sine_wave()
# except KeyboardInterrupt:
#     print("Exiting...")
#     pygame.quit()
