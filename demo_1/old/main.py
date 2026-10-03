import cv2
import time
import numpy as np
import simpleaudio as sa
from threading import Thread, Event
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)

# Open the video file or webcam (use 0 for the default webcam)
video_source = "sitar video - 1st class - 1-7-25.mov"  # Replace with 0 for a webcam feed
cap = cv2.VideoCapture(0)  # Replace with `video_source` if using a file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Function to play a continuous sine tone with panning
class ContinuousSineTone:
    def __init__(self, frequency=440, sample_rate=44100, amplitude=0.1):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.amplitude = amplitude
        self.stop_event = Event()
        self.thread = None

    def generate_tone(self, pan_position):
        # Create a sine wave
        t = np.linspace(0, 1, self.sample_rate, endpoint=False)  # 1-second buffer
        wave = (32767 * self.amplitude * np.sin(2 * np.pi * self.frequency * t)).astype(np.int16)

        # Pan the tone by adjusting the left and right channels
        left_channel = wave * (1 - pan_position)  # 1 - pan_position gives the left strength
        right_channel = wave * pan_position  # pan_position gives the right strength

        # Combine both channels into stereo
        stereo_wave = np.vstack((left_channel, right_channel)).T.astype(np.int16)
        return stereo_wave

    def play(self, pan_position):
        if self.thread is None or not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = Thread(target=self._play_loop, args=(pan_position,))
            self.thread.start()

    def _play_loop(self, pan_position):
        stereo_wave = self.generate_tone(pan_position)
        while not self.stop_event.is_set():
            audio = sa.play_buffer(stereo_wave, 2, 2, self.sample_rate)  # 2 channels (stereo)
            audio.wait_done()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()

# Create a ContinuousSineTone instance
tone = ContinuousSineTone(frequency=440, amplitude=0.1)  # Lower amplitude for quieter tone

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

    # Map red_circle_position to the panning range [0, 1]
    # 0 is left, 1 is right, and values in between correspond to positions in between
    if person_detected:
        tone.play(red_circle_position)  # Pan based on the position of the red circle
    else:
        tone.stop()  # Stop the tone if no person is detected

    # Display the frame with annotations
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
tone.stop()  # Ensure the tone stops when the program exits
