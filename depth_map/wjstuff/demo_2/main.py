"""
WJ DEMO1

Enter 0 for main state.

Enter 1 to enter the voice activation state
    ->then press enter to list all objects
    ->or enter a specific object to "navigate to it" (just a print for now)
    /!\automatically returns to main state 0 after action in state 1 voice.


1/17/25 Demo Prototype ok. Depthmap and volume not working. Need to replace with depthpro maybe -wj
"""
import time
import pygame
import threading
from depth_map import *
from object_det import *
from sound_gen import *
from input_handler import *
from my_constants import *
import globals
from webcam import *
from gui import *
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
pygame.init()

# Demo variables
frequency = 440.0  # Default frequency in Hz (A4)
volume = MAX_SINE_VOLUME       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)
sound = None
depth_person = np.inf
person_detected = False
apple_detected = False
red_circle_position = 0 
warning_sound = pygame.mixer.Sound('warning.ogg')
warning_sound.set_volume(1)  # Set volume to 10%
warning_channel = pygame.mixer.Channel(3)  # Use channel 0 for playing this sound

# Pygame GUI vars
clock = pygame.time.Clock()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Your phone - FPS: 0")
font = pygame.font.Font(None, 72)  # Default font with size 36
square_rect = pygame.Rect(SQUARE_X, SQUARE_Y, SQUARE_SIZE, SQUARE_SIZE)
text_surface = font.render("PRESS", True, BLACK).convert_alpha()  # Render text
text_rect = text_surface.get_rect(center=square_rect.center)  # Center text on square

button_is_pressed = False
last_click_time = 0
is_double_clicked = False
is_held = False  # Tracks if the button is being held


# Variables for timing
total_cycle_time = 0
total_inference_time = 0
frame_count = 0

def quit_app():
    pygame.quit()
    quit()

if __name__ == '__main__':
    # Initialize depth map 
    depth_init = init_depth()
    args = depth_init[0]
    depth_anything = depth_init[1]
    print("Loaded depthmap...")

    # Initialize YOLOv8
    model = init_objectDet()  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    print("Loaded YOLO...")

    # Start the input listener thread
    thread = threading.Thread(target=input_listener, daemon=True)
    thread.start() # like interrupt, run key detection while loop in background
    print("Loaded threads...")

    # Initialize webcam
    webcam_data = webcam_init()
    cap, cmap = webcam_data[0], webcam_data[1]
    print("Webcam started...")
    print("Fully Initialized")

    #Program "Grand loop"
    objects = []
    while cap.isOpened():
        # start timing one loop
        cycle_start_time = time.time()

        # reset detection vars each loop
        objects = []
        person_detected = False
        apple_detected = False

        # Get new webcam frame
        ret, raw_frame = cap.read() #raw_frame is dtype uint8!!!
        if not ret: break
        raw_frame = cv2.flip(raw_frame,1)
        
        # GUI handle events and render
        button_is_pressed, is_held, is_double_clicked = handle_gui_events(square_rect)
        render_gui(screen, square_rect, text_surface, text_rect, objects, button_is_pressed, is_double_clicked, clock)

        # YOLO inference and time it
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)
        inference_time = time.time() - inference_start_time

        # Depth map and time it
        depth_start_time = time.time()
        depth_data = get_depth_map(raw_frame, depth_anything, args, cmap)
        raw_depth, depth = depth_data[0], depth_data[1]
        depth_time = time.time() - depth_start_time



        # YOLO what do with each object detected
        combined_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)  # Same size as the frame        
        for result in results:
            masks = result.masks  # Segmentation masks
            boxes = result.boxes  # Bounding boxes
            names = model.names   # Class names

            # Check if masks and boxes are available
            if masks is not None and boxes is not None:
                # Iterate over each detected object
                for i in range(len(boxes)):
                    
                    # Get bounding box coordinates
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers

                    # Get confidence score
                    confidence = boxes.conf[i].item()

                    # Get class ID and class name
                    class_id = int(boxes.cls[i])
                    class_name = names[class_id]

                    objects.append((names[class_id], confidence))

                    # Get mask for the current object
                    mask_points = masks.xy[i].astype(int)  # Convert to integers


                    # Draw bounding box
                    cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw mask overlay
                    overlay = raw_frame.copy()
                    cv2.fillPoly(overlay, [mask_points], (0, 255, 0))
                    raw_frame = cv2.addWeighted(overlay, 0.3, raw_frame, 0.7, 0)

                    # Add label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(raw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # LOGIC
                    if names[class_id] == "apple":
                        apple_detected = True

                    # Check if the detected object is a person
                    if class_name == "cell phone":
                        # Get mask for the current person
                        mask = masks.xy[i]  # Polygon points for the mask

                        # Convert polygon points to a binary mask
                        mask_pts = np.array(mask, dtype=np.int32)
                        person_mask = np.zeros(raw_frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(person_mask, [mask_pts], 1)  # Fill the polygon with 1s

                        # Combine the person mask with the combined mask using logical OR
                        combined_mask = cv2.bitwise_or(combined_mask, person_mask)


                        person_detected = True
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        # Infer depth map from the raw_frame
                        # Get the depth value at the specified location (100, 100)
                        x, y = x_center, y_center  # Coordinates of the pixel
                        
                        if 0 <= y < raw_depth.shape[0] and 0 <= x < raw_depth.shape[1]:  # Check bounds
                            depth_person = raw_depth[y, x]  # Access depth at (row=y, col=x)
                            # print(f"Depth value at ({x}, {y}): {depth_person}")
                        else:
                            print(f"Coordinates ({x}, {y}) are out of bounds for the depth map with shape {raw_depth.shape}.")

                        volume = calculate_volume(depth_person)
                        # Draw a red circle at the center of the bounding box
                        cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

                        # Track the horizontal position of the red circle (panning position)
                        red_circle_position = x_center / raw_frame.shape[1]  # Normalize to [0, 1]


        globals.objects_buffer = objects

        combined_mask_resized = cv2.resize(combined_mask, (raw_frame.shape[1], raw_frame.shape[0]))
        combined_mask_for_show = cv2.cvtColor(combined_mask_resized*255, cv2.COLOR_GRAY2BGR)
        combined_mask_for_show = combined_mask_for_show.astype(np.uint8)

        depth_masked = combined_mask_resized * raw_depth

        # Step 1: Identify non-zero elements
        non_zero_elements = depth_masked[depth_masked != 0]

        # Step 2: Calculate the average of non-zero elements
        average_non_zero = np.mean(non_zero_elements)

        # Step 3: Calculate the percentage of non-zero elements
        total_elements = depth_masked.size
        non_zero_count = non_zero_elements.size
        percentage_non_zero = (non_zero_count / total_elements) * 100

        # Print the results
        print(f"Average of non-zero elements: {average_non_zero}")
        print(f"Percentage of non-zero elements: {percentage_non_zero}%")


        dm_min_val = depth_masked.min()
        dm_max_val = depth_masked.max()
        depth_masked = (depth_masked - dm_min_val) / (dm_max_val - dm_min_val) * 255.0
        depth_masked = depth_masked.astype(np.uint8)
        if args.grayscale:
            depth_masked = np.repeat(depth_masked[..., np.newaxis], 3, axis=-1)
        else:
            depth_masked = (cmap(depth_masked)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Update sound
        # volume = max(0.0, min(volume, MAX_SINE_VOLUME))  # Limit volume range
        volume = calculate_volume(depth_person)
        panning = max(0.0, min(red_circle_position, 1.0))  # Limit panning range
        wave = generate_sound_wave(frequency, SAMPLE_RATE, volume, panning, DURATION, squarewave=apple_detected)
 
        #if there is a person on screen, play sound
        if person_detected:
            # print(f"person. Pan: {panning}, Vol: {volume}, depth: {depth_person}")
            # Rendering the text on top
            cv2.putText(depth, f'Pan: {panning:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(depth, f'Vol: {volume:.2f}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(depth, f'DepthPerson: {depth_person:.2f}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), 3, cv2.LINE_AA)

            if sound is None:
                sound = pygame.sndarray.make_sound(wave)
            else:
                sound.stop()
                sound = pygame.sndarray.make_sound(wave)
            sound.play(loops=0)

        if apple_detected:
            if not warning_channel.get_busy():  # Check if the channel is not currently playing a sound
                warning_channel.play(warning_sound)
        else:
            if warning_channel.get_busy():  # Check if the channel is not currently playing a sound
                warning_channel.fadeout(500)

        if person_detected:
            cv2.putText(depth_masked, f'Avg_dist {average_non_zero:.2f}', (x_center-10, y_center), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE-0.5, (255, 0, 255), 2, cv2.LINE_AA)

        if args.pred_only:
            cv2.imshow('depth only ', depth)
        else:
            blank_image = np.zeros_like(raw_frame)
            top_row_frame = cv2.hconcat([raw_frame, depth])
            bottom_row_frame = cv2.hconcat([combined_mask_for_show, depth_masked])
            final_output = cv2.vconcat([top_row_frame, bottom_row_frame])

            cv2.imshow("wjdemo2", final_output)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # End timing the entire cycle
        cycle_time = time.time() - cycle_start_time
        total_cycle_time += cycle_time
        frame_count += 1
        print(f"Tot: {int(cycle_time*1000)}ms, YOLO: {int(inference_time*1000)}ms, Depth: {int(depth_time*1000)}ms, Other: {int((cycle_time-inference_time-depth_time)*1000)}ms")

    
    quit_app()




