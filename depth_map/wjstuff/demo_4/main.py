"""
WJDEMO 4 heavy rewriting in progress 2/10/2025
"""
import pygame, threading, time, supervision
# globals are imported in input_handler
from depth_map import *
from object_det import *
from sound_gen import *
from input_handler import *
from my_constants import *
from webcam import *
from gui import *
from hrtf import *

pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Your phone - FPS: 0")
clock = pygame.time.Clock()
warning_sound = pygame.mixer.Sound('warning.ogg')
warning_sound.set_volume(1)  # Set volume to 10%
warning_channel = pygame.mixer.Channel(3)  # Use channel 0 for playing this sound
font = pygame.font.Font(None, 72)  # Default font with size 36
square_rect = pygame.Rect(SQUARE_X, SQUARE_Y, SQUARE_SIZE, SQUARE_SIZE)
text_surface = font.render("PRESS", True, BLACK).convert_alpha()  # Render text
text_rect = text_surface.get_rect(center=square_rect.center)  # Center text on square







if __name__ == '__main__':
    # Initialize depth map
    args, depth_anything = init_depth()
    print("Loaded depthmap...")

    # Initialize YOLOv8
    model = init_objectDet()  # Use the appropriate YOLOv8 model variant (n, s, m, l, x)
    print("Loaded YOLO...")

    # Initialize tracker
    tracker = supervision.ByteTrack()
    print("Loaded tracker")

    # Start the input listener thread
    thread = threading.Thread(target=input_listener, daemon=True)
    thread.start() # like interrupt, run key detection while loop in background
    print("Loaded threads...")

    """Simulates a busy main loop where the frequency changes based on some variable."""
    target_sound_data = [440, 100, 0.5, 0.5]  # Start with 440 Hz (A4)
    frequency_event = threading.Event()  # Event to trigger tone playback

    # Start the sine tone thread
    tone_thread = threading.Thread(target=play_sine_tone, args=(frequency_event, target_sound_data))
    tone_thread.daemon = True  # Daemon thread will exit when the main program exits
    tone_thread.start()


    # Initialize webcam
    webcam_data = webcam_init()
    cap, cmap = webcam_data[0], webcam_data[1]
    print("Webcam started...")
    print("Fully Initialized")  
    print_logo()
    print_block_letter_art("CviSion")
    print_menu()
    
    #Program "Grand loop"
    while cap.isOpened():
        # start timing one loop
        cycle_start_time = time.time()

        # reset detection vars each loop
        danger_detected = False
        important_detected = False

        # Get new webcam frame
        ret, raw_frame = cap.read() #raw_frame is dtype uint8!!!
        if not ret: break
        raw_frame = cv2.flip(raw_frame,1)
        
        # GUI handle events and render
        globals.button_is_pressed, globals.is_held, globals.is_double_clicked = handle_gui_events(square_rect, globals.last_click_time)
        render_gui(screen, square_rect, text_surface, text_rect, globals.objects_buffer, globals.button_is_pressed, globals.is_double_clicked, clock)

        # Depth map and time it
        depth_start_time = time.time()
        raw_depth, depth_to_plot = get_depth_map(raw_frame, depth_anything, args, cmap)
        depth_time = time.time() - depth_start_time

        # YOLO inference and time it
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)[0] #, conf=0.25
        depth_to_plot = process_yolo_results(raw_frame, model, results, raw_depth, depth_to_plot, tracker)
        inference_time = time.time() - inference_start_time



        # Logic here could be simplified
        if globals.is_guiding:
            if globals.current_target_to_guide is not None and is_key_in_dict(globals.current_target_to_guide, globals.objects_data):
                obj_data = globals.objects_data[globals.current_target_to_guide]
                target_mask_vis,target_class_name, target_depth, target_x_angle, target_y_angle= obj_data['mask_vis'], obj_data['class'], obj_data['depth'], obj_data['x_angle'], obj_data['y_angle']
                print('guiding...')
                if target_depth < ARRIVAL_METERS:
                    globals.current_target_to_guide = None
            else:
                print('lost!')
                globals.is_guiding = False
                
                if globals.state == 2: # wating for the guide to finish 2
                    # if not globals.is_guiding:
                    globals.current_target_to_guide = None
                    globals.state = 0
                    print_notification('finished guiding you to target! returnung to mainstate 0!')
                    print_menu()


            
    # sound test
        if globals.is_guiding:
            # Simulate some busy loop, and change the frequency based on a condition
            # target_sound_data[0] = (target_sound_data[0] + 10) % 1000  # Increase frequency by 10 Hz every loop
            target_sound_data[1] = min(1.0, 1.0 / (target_depth ** 2)) # depth to volume
            target_sound_data[2] = target_x_angle
            target_sound_data[3] = target_y_angle
            # print(f"Current Frequency: {target_sound_data[0]} Hz")
            # print(target_sound_data)
            # Trigger the sine tone to play with the updated frequency
            frequency_event.set()




            
        # End timing the entire cycle
        cycle_time = time.time() - cycle_start_time
        globals.total_cycle_time += cycle_time

        blank_img = np.zeros_like(raw_frame)

        if globals.is_guiding and globals.current_target_to_guide is not None:
            # Get object info

            # Create colored mask
            colored_mask = np.zeros_like(raw_frame)
            colored_mask[target_mask_vis > 0] = WHITE  # White mask
            
            # Find center of the mask
            y_coords, x_coords = np.where(target_mask_vis > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # Format text
                text = f"{target_class_name} {target_depth:.2f}m"
                
                # Get text size for centering
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = center_x - text_width // 2
                text_y = center_y + text_height // 2
                
                # Draw text with background for better visibility
                cv2.putText(colored_mask, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # Thick black outline
                cv2.putText(colored_mask, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)  # White text
            
            # Combine with blank image
            blank_img = cv2.addWeighted(blank_img, 1, colored_mask, 1, 0)
            depth_to_plot = blank_img

        # Create a wider objects screen by doubling the width and half the height
        objects_screen = np.zeros((raw_frame.shape[0]//2, raw_frame.shape[1] * 2, 3), dtype=np.uint8)
        display_dict_info(objects_screen, globals.objects_data, excluding = ['mask_vis'])


        performance_text = [
            f"FPS: {1/cycle_time:.2f}fps",
            f"Tot: {int(cycle_time*1000)}ms",
            f"YOLO: {int(inference_time*1000)}ms",
            f"Depth: {int(depth_time*1000)}ms",
            f"Other: {int((cycle_time-inference_time-depth_time)*1000)}ms"
        ]
        raw_frame = add_performance_text(raw_frame, performance_text)
        
        # Plotting
        if args.pred_only:
            cv2.imshow('depth only ', depth_to_plot)
        else:
            # Create the top row by concatenating raw frame and depth plot
            top_row_frame = cv2.hconcat([raw_frame, depth_to_plot])
            
            # Bottom row is just the single wide objects screen
            bottom_row_frame = objects_screen  # No need to concatenate with itself anymore
            
            final_output = cv2.vconcat([top_row_frame, bottom_row_frame])
            cv2.imshow("wjdemo4", final_output)

        # Break the loop on 'q' key press
        if (cv2.waitKey(1) & 0xFF == ord('q')) or globals.quit == True:
            print("quitting...")
            print_logo()
            print_block_letter_art("Bye guys")
            if cap.isOpened():  # Check if the capture object is open
                cap.release()  # Release the video capture
            quit_app()

# end of program :)




