"""
WJ DEMO1 -> WJ DEMO2

Enter 0 for main state.

Enter 1 to enter the voice activation state
    ->then press enter to list all objects
    ->or enter a specific object to "navigate to it" (just a print for now)
    /!\automatically returns to main state 0 after action in state 1 voice.


1/17/25 Demo Prototype ok. Depthmap and volume not working. Need to replace with depthpro maybe -wj
1/30/25 Refactored code
"""
import soundfile as sf

import time
import pygame
import threading
from depth_map import *
from object_det import *
from sound_gen import *
from input_handler import *
from my_constants import *
# import globals
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
danger_detected = False
red_circle_position = 0 
warning_sound = pygame.mixer.Sound('warning.ogg')
warning_sound.set_volume(1)  # Set volume to 10%
warning_channel = pygame.mixer.Channel(3)  # Use channel 0 for playing this sound
danger_detected = False
important_detected = False


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
    args, depth_anything = init_depth()
    print("Loaded depthmap...")

    # Initialize YOLOv8
    model = init_objectDet()  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    print("Loaded YOLO...")

    # Initialize SORT
    mot_tracker = init_sort()
    print("Loaded SORT...")

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
    while cap.isOpened():
        # start timing one loop
        cycle_start_time = time.time()

        # reset detection vars each loop
        person_detected = False
        danger_detected = False
        danger_detected = False
        important_detected = False

        # Get new webcam frame
        ret, raw_frame = cap.read() #raw_frame is dtype uint8!!!
        if not ret: break
        raw_frame = cv2.flip(raw_frame,1)
        
        # GUI handle events and render
        button_is_pressed, is_held, is_double_clicked = handle_gui_events(square_rect, last_click_time)
        render_gui(screen, square_rect, text_surface, text_rect, globals.objects_buffer, button_is_pressed, is_double_clicked, clock)

        # YOLO inference and time it
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)
        inference_time = time.time() - inference_start_time

        # Depth map and time it
        depth_start_time = time.time()
        raw_depth, depth, _, _, _, _ = get_depth_map(raw_frame, depth_anything, args, cmap)
        depth_time = time.time() - depth_start_time

        # process the yolo stuff
        raw_frame, combined_mask, depth_person, danger_detected, person_detected, x_angle, y_angle, x_center, y_center = process_yolo_results(results, raw_frame, raw_depth, model.names, mot_tracker) 
        combined_mask_resized, combined_mask_for_show = process_SAM_mask(combined_mask)


        depth_masked = combined_mask_resized * raw_depth
        
        # get the depth of your target object
        target_distance = get_distance_of_object(depth_masked)

        depth_masked = get_plottable_depth(depth_masked, args, cmap)[0]

        # update sound based on camera input and processing
        wave = update_sound(depth_person, red_circle_position, frequency, danger_detected)

        # LOGIC
        # print(globals.objects_buffer)
        objects_only = [item[0] for item in globals.objects_buffer]
        # print(objects_only)
        for element in objects_only:
            if element in DANGEROUS_OBJECTS:
                # globals.announce_state = 2
                danger_detected = True
                # print("DANGER:", element, "detected!!!")
            else: globals.announce_state = 0
        for element in objects_only:
            if element in IMPORTANT_OBJECTS:
                # if globals.announce_state != 2:
                important_detected = True
                # globals.announce_state = 2
                # print("IMPORTANT:", element, "detected!!!")
            else: globals.announce_state = 0

        if danger_detected:
            globals.announce_state = 2
        else:
            if important_detected:
                globals.announce_state = 1
            else:
                globals.announce_state = 0


        # SOUND LOGIC
        #if there is a person on screen, play sound
        if person_detected:
            play_sound(sound, wave)
            cv2.putText(depth, f'Pan: {panning:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(depth, f'Vol: {volume:.2f}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(depth, f'DepthPerson: {depth_person:.2f}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(depth_masked, f'Avg_dist {target_distance:.2f}', (x_center-10, y_center), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE-0.5, (255, 0, 255), 2, cv2.LINE_AA)
        if danger_detected:
            # print('bruh')
            play_sound(sound, wave)
            if not warning_channel.get_busy():  # Check if the channel is not currently playing a sound
                warning_channel.play(warning_sound)
        else:
            if warning_channel.get_busy():  # Check if the channel is not currently playing a sound
                warning_channel.fadeout(500)


        # Plotting
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
            cap.release()
            cv2.destroyAllWindows()
        # End timing the entire cycle
        cycle_time = time.time() - cycle_start_time
        total_cycle_time += cycle_time
        frame_count += 1
        # print(f"Tot: {int(cycle_time*1000)}ms, YOLO: {int(inference_time*1000)}ms, Depth: {int(depth_time*1000)}ms, Other: {int((cycle_time-inference_time-depth_time)*1000)}ms")

    # quit pygame stuff and in general
    quit_app()




