import argparse
import cv2
import matplotlib
import numpy as np
import torch
import pygame
import threading
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    

    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + margin_width
    

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

    # # Open the video file or webcam (use 0 for the default webcam)
    # video_source = "sitar video - 1st class - 1-7-25.mov"  # Replace with 0 for a webcam feed
    # cap = cv2.VideoCapture(0)  # Replace with `video_source` if using a file

    # Start the input listener thread
    thread = threading.Thread(target=input_listener, daemon=True)
    thread.start() # like interrupt, run key detection while loop in background

    print("Sine wave generator started. Adjust frequency, volume, and panning in real time.")
    print("Commands: 'f <freq>' (frequency), 'v <vol>' (volume), 'p <pan>' (panning)")


    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_width = raw_frame.shape[1]

        raw_frame = cv2.flip(raw_frame,1)
        raw_depth = depth_anything.infer_image(raw_frame, args.input_size)
        min_val = raw_depth.min()
        max_val = raw_depth.max()
        min_loc = np.unravel_index(np.argmin(raw_depth, axis=None), raw_depth.shape)
        max_loc = np.unravel_index(np.argmax(raw_depth, axis=None), raw_depth.shape)
        depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
        depth = depth.astype(np.uint8)
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        cv2.putText(depth, f'Closest: {min_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (0, 255, 0), -1)  # Closest point
        cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (255, 0, 255), -1)  # Farthest point

        cv2.putText(depth, f'Closest', (max_loc[1], max_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    

        # Run YOLOv8 inference on the frame
        results = model(raw_frame, verbose=False)
        objects = []

        # Track if a person is detected and find the position of the red circle
        person_detected = False
        red_circle_position = 0  # Default position if no person is detected

        for detection in results[0].boxes.data:
            confidence = float(detection[4])  # Confidence score

            x_min, y_min, x_max, y_max = map(float, detection[:4])  # Bounding box coordinates
            class_id = int(detection[5])  # Class ID
            objects.append((model.names[class_id], confidence))

            if model.names[class_id] == "person":
                person_detected = True
                depth_person = np.min(raw_depth[int(y_min):int(y_max), int(x_min):int(x_max)])
                
                if depth_person <= 0.3:
                    volume = 1
                elif depth_person >= 5:
                    volume = 0
                else:
                    # Linearly scale the volume between 0.3 meters (volume = 1) and 5 meters (volume = 0)
                    volume = 1 - (depth_person - 0.3) / (5 - 0.3)
                # print(f"Volume set to {volume}")

                # Calculate the center of the bounding box
                x_center = int((x_min + x_max) / 2)
                y_center = int((y_min + y_max) / 2)

                # Draw a red circle at the center of the bounding box
                cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

                # Track the horizontal position of the red circle (panning position)
                red_circle_position = x_center / raw_frame.shape[1]  # Normalize to [0, 1]

        raw_frame = results[0].plot()




        volume = max(0.0, min(volume, 1.0))  # Limit volume range
        panning = max(0.0, min(red_circle_position, 1.0))  # Limit panning range
        # print(f"Panning set to {panning} (0.0 = left, 1.0 = right)")
        # Sound generate
        wave = generate_sine_wave(frequency, volume, panning, duration)
        # print(min_val)
        #if there is a person on screen, play sound
        if person_detected:
            cv2.rectangle(raw_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            print(f"person. Pan: {panning}, Vol: {volume}, depth: {depth_person}")

            if sound is None:
                sound = pygame.sndarray.make_sound(wave)
            else:
                sound.stop()
                sound = pygame.sndarray.make_sound(wave)
            
            sound.play(loops=0)

        if args.pred_only:
            cv2.imshow('depth only ', depth)
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])
            cv2.imshow('depth together ', combined_frame)
            
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


