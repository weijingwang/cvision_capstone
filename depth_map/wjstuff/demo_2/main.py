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
# Initialize global variables for frequency, volume, and panning
frequency = 440.0  # Default frequency in Hz (A4)
volume = MAX_SINE_VOLUME       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)
sound = None
person_detected = False
red_circle_position = 0 


# Initialize Pygame mixer
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Your phone - FPS: 0")
done = False
# Square properties
square_size = 200
square_x = (SCREEN_WIDTH - square_size) // 2
square_y = (SCREEN_HEIGHT - square_size) // 2
square_rect = pygame.Rect(square_x, square_y, square_size, square_size)
button_is_pressed = False
font = pygame.font.Font(None, 72)  # Default font with size 36
text_surface = font.render("PRESS", True, BLACK).convert_alpha()  # Render text
text_rect = text_surface.get_rect(center=square_rect.center)  # Center text on square
clock = pygame.time.Clock()
last_click_time = 0
double_click_threshold = 0.3  # 300 ms for a double click
is_double_clicked = False
is_held = False  # Tracks if the button is being held
warning_sound = pygame.mixer.Sound('warning.ogg')
warning_sound.set_volume(1)  # Set volume to 10%
warning_channel = pygame.mixer.Channel(3)  # Use channel 0 for playing this sound


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
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    if args.pred_only: 
        output_width = frame_width
    else: 
        output_width = frame_width * 2 + MARGIN_WIDTH
    print("Webcam started...")
    print("Fully Initialized")

    #Program "Grand loop"
    while cap.isOpened():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_app()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if square_rect.collidepoint(event.pos):
                    current_time = time.time()
                    if current_time - last_click_time <= double_click_threshold:
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
        pygame_fps = int(clock.get_fps())  # Get the current frames per second
        # Update window title with FPS
        pygame.display.set_caption(f"Your phone - FPS: {pygame_fps}")


        person_detected = False
        apple_detected = False
        # Webcam variables
        ret, raw_frame = cap.read()
        if not ret:
            break
        frame_width = raw_frame.shape[1]
        raw_frame = cv2.flip(raw_frame,1)

        # Run YOLOv8 inference on the frame
        results = model(raw_frame, verbose=False)
        


        # Depth math and get depth map to render
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
        cv2.circle(depth, (min_loc[1], min_loc[0]), 5, (255, 0, 255), -1)  # Closest point
        cv2.putText(depth, f'Closest', (min_loc[1], min_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        
        cv2.putText(depth, f'Farthest: {max_val:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(depth, (max_loc[1], max_loc[0]), 5, (0, 0, 255), -1)  # Farthest point
        cv2.putText(depth, f'Farthest', (max_loc[1], max_loc[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



        # YOLO what do with each object detected
        objects = []
        for detection in results[0].boxes.data:
            # YOLO save data for object
            confidence = float(detection[4])  # Confidence score
            x_min, y_min, x_max, y_max = map(float, detection[:4])  # Bounding box coordinates
            class_id = int(detection[5])  # Class ID
            objects.append((model.names[class_id], confidence))

            # LOGIC
            if model.names[class_id] == "apple":
                apple_detected = True

            if model.names[class_id] == "person":
                person_detected = True
                x_center = int((x_min + x_max) / 2)
                y_center = int((y_min + y_max) / 2)
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
            cv2.putText(depth, f'DepthPerson: {depth_person:.2f}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), 3, cv2.LINE_AA)

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

        raw_frame = results[0].plot()

        if args.pred_only:
            cv2.imshow('depth only ', depth)
        else:
            split_region = np.ones((frame_height, MARGIN_WIDTH, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth])
            cv2.imshow('depth together ', combined_frame)
            
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            quit_app()
            # break
    
    quit_app()




