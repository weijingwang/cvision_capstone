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

    # Initialize tracker
    tracker = supervision.ByteTrack()
    print("Loaded tracker")

    # Start the input listener thread
    thread = threading.Thread(target=input_listener, daemon=True)
    thread.start() # like interrupt, run key detection while loop in background
    print("Loaded threads...")

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
        # results = model(raw_frame, verbose=False)
        results = model(raw_frame, verbose=False)[0] #, conf=0.25

        depth_to_plot = process_yolo_results(raw_frame, model, results, raw_depth, depth_to_plot, tracker)

        inference_time = time.time() - inference_start_time

        if globals.is_guiding:
            if globals.current_target_to_guide is not None and is_key_in_dict(globals.current_target_to_guide, globals.objects_data):
                print('guiding...')
            else:
                print('lost!')
                globals.is_guiding = False
                
                if globals.state == 2: # wating for the guide to finish 2
                    if not globals.is_guiding:
                        globals.current_target_to_guide = None
                        globals.state = 0
                        print_notification('finished guiding you to target! returnung to mainstate 0!')
                        print_menu()


            

        
        # # combined_mask_resized, combined_mask_for_show = process_SAM_mask(combined_mask)


        # # depth_masked = combined_mask_resized * raw_depth
        
        # # get the depth of your target object
        # # target_distance = get_distance_of_object(depth_masked)

        # # depth_masked = get_plottable_depth(depth_masked, args, cmap)[0]


        # # HRTF stuff test
        # """
        # [WARNING!!!!] CHECKS HRFT EVERY TIME EVEN WHEN NOT TRACKING! WHEN NOT TRACKING, x_angle, y_angle = 0!!!!
        # """
        # # hrtf_file, sound_is_flipped = get_HRTF_params(y_angle, x_angle, HRTF_DIR)
        # # print(hrtf_file, sound_is_flipped, x_angle, y_angle)
        
        # # update sound based on camera input and processing
        # # wave = update_sound(depth_person, red_circle_position, frequency, danger_detected)

        # # LOGIC
        # # print(globals.objects_buffer)
        # objects_only = [item[0] for item in globals.objects_buffer]
        # # print(objects_only)
        # for element in objects_only:
        #     if element in DANGEROUS_OBJECTS:
        #         # globals.announce_state = 2
        #         danger_detected = True
        #         # print("DANGER:", element, "detected!!!")
        #     else: globals.announce_state = 0
        # for element in objects_only:
        #     if element in IMPORTANT_OBJECTS:
        #         # if globals.announce_state != 2:
        #         important_detected = True
        #         # globals.announce_state = 2
        #         # print("IMPORTANT:", element, "detected!!!")
        #     else: globals.announce_state = 0

        # if danger_detected:
        #     globals.announce_state = 2
        # else:
        #     if important_detected:
        #         globals.announce_state = 1
        #     else:
        #         globals.announce_state = 0

        """
        If tracking then play sound and the visuals on bottom row.
        """


        # # SOUND LOGIC
        # #if there is a person on screen, play sound
        # if person_detected:
        #     play_sound(sound, wave)
        #     cv2.putText(depth, f'Pan: {panning:.2f}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 0, 255), 3, cv2.LINE_AA)
        #     cv2.putText(depth, f'Vol: {volume:.2f}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), 3, cv2.LINE_AA)
        #     cv2.putText(depth, f'DepthPerson: {depth_person:.2f}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), 3, cv2.LINE_AA)
        #     cv2.putText(depth_masked, f'Avg_dist {target_distance:.2f}', (x_center-10, y_center), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE-0.5, (255, 0, 255), 2, cv2.LINE_AA)
        # if danger_detected:
        #     # print('bruh')
        #     play_sound(sound, wave)
        #     if not warning_channel.get_busy():  # Check if the channel is not currently playing a sound
        #         warning_channel.play(warning_sound)
        # else:
        #     if warning_channel.get_busy():  # Check if the channel is not currently playing a sound
        #         warning_channel.fadeout(500)

            
        # End timing the entire cycle
        cycle_time = time.time() - cycle_start_time
        globals.total_cycle_time += cycle_time



        # Create a wider objects screen by doubling the width and half the height
        objects_screen = np.zeros((raw_frame.shape[0]//2, raw_frame.shape[1] * 2, 3), dtype=np.uint8)
        display_dict_info(objects_screen, globals.objects_data)


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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

    # quit pygame stuff and in general
    quit_app()




