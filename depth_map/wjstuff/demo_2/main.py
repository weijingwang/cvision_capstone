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
import cv2
import matplotlib
import pygame
import threading
from depth_map import *
from object_det import *
from sound_gen import *
from input_handler import *
from my_constants import *
import globals
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
pygame.init()

# Demo variables
frequency = 440.0  # Default frequency in Hz (A4)
volume = MAX_SINE_VOLUME       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)
sound = None
person_detected = False
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



def quit_app():
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
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
    cmap = matplotlib.colormaps.get_cmap('gray')
    cap = cv2.VideoCapture(0) # webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RESOLUTION[1])
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + MARGIN_WIDTH
    print("Webcam started...")
    print("Fully Initialized")


    # Variables for timing
    total_cycle_time = 0
    total_inference_time = 0
    frame_count = 0

    #Program "Grand loop"
    while cap.isOpened():
        # Start timing the entire cycle
        cycle_start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_app()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if square_rect.collidepoint(event.pos):
                    current_time = time.time()
                    if current_time - last_click_time <= DOUBLE_CLICK_THRESHOLD:
                        # print("double pressed")
                        is_double_clicked = True
                    # else:
                    #     # print("pressed")
                    last_click_time = current_time
                    button_is_pressed = True
                    is_held = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if square_rect.collidepoint(event.pos):
                    button_is_pressed = False
                    is_held = False
        if is_double_clicked:
            print("LISTEN FOR VOICE INSTRUCTIONS")
            is_double_clicked = False
        if button_is_pressed:
            print(f"{objects}")
        color = DARK_GREEN if button_is_pressed else GREEN
        screen.fill(WHITE)  # Clear screen
        pygame.draw.rect(screen, color, square_rect)  # Draw green square
        screen.blit(text_surface, text_rect)  # Draw text on the screen
        pygame.display.flip()
        # Limit FPS to 60
        clock.tick(PYGAME_FPS)  # Returns the time passed since the last frame in milliseconds
        pygame_fps = clock.get_fps()  # Get the current frames per second
        # Update window title with FPS
        pygame.display.set_caption(f"Your phone - FPS: {pygame_fps:.3f}")


        person_detected = False
        apple_detected = False
        # Webcam variables
        ret, raw_frame = cap.read() #raw_frame is dtype uint8!!!
        if not ret:
            break

        raw_frame = cv2.flip(raw_frame,1)

        # Run YOLOv8 inference on the frame
        # Start timing the inference step
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)
        # End timing the inference step
        inference_time = time.time() - inference_start_time
        total_inference_time += inference_time

        depth_start_time = time.time()
        # Depth math and get depth map to render
        raw_depth = depth_anything.infer_image(raw_frame, args.input_size) #float32, same resolution as webcam
        min_val = raw_depth.min()
        max_val = raw_depth.max()
        min_loc = np.unravel_index(np.argmin(raw_depth, axis=None), raw_depth.shape)
        max_loc = np.unravel_index(np.argmax(raw_depth, axis=None), raw_depth.shape)
        depth = (raw_depth - min_val) / (max_val - min_val) * 255.0
        depth = depth.astype(np.uint8) #make the depth to show uint8 as well
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        cv2.putText(depth, f'Closest: {min_val:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (255, 0, 255), -1)  # Closest point
        cv2.putText(depth, f'Closest', (min_loc[1], min_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        
        cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (0, 0, 255), -1)  # Farthest point
        cv2.putText(depth, f'Farthest', (max_loc[1], max_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        depth_time = time.time() - depth_start_time



        # YOLO what do with each object detected
        objects = []
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
                    if class_name == "person":
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

                        if depth_person >= 4:
                            volume = MAX_SINE_VOLUME
                        elif depth_person <= 3:
                            volume = 0.0
                        else:
                            # Linear interpolation between 3 and 4
                            volume = MAX_SINE_VOLUME * ((depth_person - 3) / (4 - 3))

                        # Draw a red circle at the center of the bounding box
                        cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

                        # Track the horizontal position of the red circle (panning position)
                        red_circle_position = x_center / raw_frame.shape[1]  # Normalize to [0, 1]


        globals.objects_buffer = objects

        # # At this point, combined_mask contains the combined mask for the "person" class
        # Ensure the combined mask is not None
        # Resize the combined mask
        combined_mask_resized = cv2.resize(combined_mask, (raw_frame.shape[1], raw_frame.shape[0]))

        # Convert the combined mask to BGR for display
        combined_mask_for_show = cv2.cvtColor(combined_mask_resized*255, cv2.COLOR_GRAY2BGR)
        combined_mask_for_show = combined_mask_for_show.astype(np.uint8)

        print(combined_mask.shape, combined_mask_resized.shape, combined_mask_for_show.shape)



        # # Access masks directly from results[0].masks
        # if hasattr(results[0], "masks") and results[0].masks is not None:
        #     masks = results[0].masks.data  # Masks as binary numpy arrays
        #     classes = results[0].boxes.cls  # Class indices for the masks
        #     class_names = results[0].names  # Class name mapping (index to name)

        #     combined_mask = None

        #     # Combine masks only for "person" class
        #     for mask, cls in zip(masks, classes):
        #         if class_names[int(cls)] == "person":
        #             mask = mask.cpu().numpy().astype(np.float32)  # Keep binary mask as 0s and 1s
        #             if combined_mask is None:
        #                 combined_mask = mask
        #             else:
        #                 combined_mask = cv2.bitwise_or(combined_mask, mask)

        # # At this point, combined_mask contains the combined mask for the "person" class

        # # Ensure the combined mask is not None
        # if combined_mask is None:
        #     combined_mask = np.zeros((raw_frame.shape[0], raw_frame.shape[1]), dtype=np.float32)

        # # Resize the combined mask
        # combined_mask_resized = cv2.resize(combined_mask, (raw_frame.shape[1], raw_frame.shape[0]))

        # # Convert the combined mask to BGR for display
        # combined_mask_for_show = cv2.cvtColor(combined_mask_resized*255, cv2.COLOR_GRAY2BGR)
        # combined_mask_for_show = combined_mask_for_show.astype(np.uint8)

        # print(combined_mask.shape, combined_mask_resized.shape, combined_mask_for_show.shape)

        # for result in results:
        #     masks = result.masks  # Extract masks
        #     boxes = result.boxes  # Bounding boxes
        #     names = model.names   # Class names

        # for detection in results[0].boxes.data:
        #     # YOLO save data for object
        #     confidence = float(detection[4])  # Confidence score
        #     x_min, y_min, x_max, y_max = map(float, detection[:4])  # Bounding box coordinates
        #     class_id = int(detection[5])  # Class ID
        #     objects.append((model.names[class_id], confidence))
        #     label = f"{model.names[class_id]}: {confidence:.2f}"
        #     # Draw the bounding box on the combined mask
        #     cv2.rectangle(combined_mask_for_show, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        #     cv2.putText(combined_mask_for_show, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 2)
            


        #     # LOGIC
        #     if model.names[class_id] == "apple":
        #         apple_detected = True

        #     if model.names[class_id] == "person":
        #         person_detected = True
        #         x_center = int((x_min + x_max) / 2)
        #         y_center = int((y_min + y_max) / 2)
        #         # Infer depth map from the raw_frame
        #         # Get the depth value at the specified location (100, 100)
        #         x, y = x_center, y_center  # Coordinates of the pixel
                
        #         if 0 <= y < raw_depth.shape[0] and 0 <= x < raw_depth.shape[1]:  # Check bounds
        #             depth_person = raw_depth[y, x]  # Access depth at (row=y, col=x)
        #             # print(f"Depth value at ({x}, {y}): {depth_person}")
        #         else:
        #             print(f"Coordinates ({x}, {y}) are out of bounds for the depth map with shape {raw_depth.shape}.")

        #         if depth_person >= 4:
        #             volume = MAX_SINE_VOLUME
        #         elif depth_person <= 3:
        #             volume = 0.0
        #         else:
        #             # Linear interpolation between 3 and 4
        #             volume = MAX_SINE_VOLUME * ((depth_person - 3) / (4 - 3))

        #         # Draw a red circle at the center of the bounding box
        #         cv2.circle(raw_frame, (x_center, y_center), radius=50, color=(0, 0, 255), thickness=-1)

        #         # Track the horizontal position of the red circle (panning position)
        #         red_circle_position = x_center / raw_frame.shape[1]  # Normalize to [0, 1]


        # globals.objects_buffer = objects



        # print(combined_mask_resized.shape, raw_depth.shape)

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
        volume = max(0.0, min(volume, MAX_SINE_VOLUME))  # Limit volume range
        panning = max(0.0, min(red_circle_position, 1.0))  # Limit panning range
        wave = generate_sine_wave(frequency, SAMPLE_RATE, volume, panning, DURATION)

        #if there is a person on screen, play sound
        if person_detected:
            # print(f"person. Pan: {panning}, Vol: {volume}, depth: {depth_person}")
            # Rendering the text on top
            cv2.putText(depth, f'Pan: {panning:.2f}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(depth, f'Vol: {volume:.2f}', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(depth, f'DepthPerson: ', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), 3, cv2.LINE_AA)
#{depth_person:.2f}
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

        cv2.putText(depth_masked, f'Avg_dist {average_non_zero:.2f}', (x_center-10, y_center), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE-0.5, (255, 0, 255), 2, cv2.LINE_AA)

        # raw_frame = results[0].plot()
        # print(raw_frame.dtype, combined_mask_for_show.dtype, depth.dtype, depth_masked.dtype)

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




