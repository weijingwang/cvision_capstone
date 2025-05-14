# main.py
import pygame, threading, time, supervision
# globals are imported in input_handler
from depth_map import *
from object_det import *
from sound_gen import *
from input_handler import *
from my_constants import *  # Make sure DEPTH_MAP_FRAME_SKIP is defined here
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

    # Initialize ArUco detector
    aruco_detector = init_aruco_detector()
    print("Loaded ArUco detector...")

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
    print("Loaded targeting audio threat...")

    MODEL_NAMES_AUDIO = create_audio_dictionary('classnames_audio')

    # Initialize webcam
    webcam_data = webcam_init()
    cap, cmap = webcam_data[0], webcam_data[1]
    print("Webcam started...")
    print("Fully Initialized")  
    print_logo()
    print_block_letter_art("CviSion")
    print_menu()
    
    # Initialize variables for depth map caching
    frame_counter = 0
    aruco_frame_counter = 0
    cached_raw_depth = None
    cached_depth_to_plot = None
    
    # Initialize ArUco tracking
    aruco_marker_history = {}  # Dictionary to track ArUco markers {marker_id: frames_since_last_seen}
    
    #Program "Grand loop"
    test = 0
    while cap.isOpened():
        # start timing one loop
        cycle_start_time = time.time()

        # reset detection vars each loop
        danger_detected = False
        important_detected = False

        # Get new webcam frame
        ret, raw_frame = cap.read() #raw_frame is dtype uint8!!!
        if not ret: break
        # raw_frame = cv2.flip(raw_frame,1)
        
        # GUI handle events and render
        globals.button_is_pressed, globals.is_held, globals.is_double_clicked = handle_gui_events(square_rect, globals.last_click_time)
        render_gui(screen, square_rect, text_surface, text_rect, globals.objects_buffer, globals.button_is_pressed, globals.is_double_clicked, clock)

        # Depth map calculation
        depth_start_time = time.time()

        # Only run depth map every DEPTH_MAP_FRAME_SKIP frames
        if frame_counter % DEPTH_MAP_FRAME_SKIP == 0:
            # Calculate new depth map
            cached_raw_depth, cached_depth_to_plot = get_depth_map(raw_frame, depth_anything, args, cmap)

        # Use the cached depth map (either just calculated or from previous frames)
        raw_depth, depth_to_plot = cached_raw_depth, cached_depth_to_plot.copy()  # Make a copy to avoid modifying the original
        depth_time = time.time() - depth_start_time

        # YOLO inference and time it
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)[0] 

        # Process ArUco markers only on specific frames
        detected_aruco_ids = []
        if aruco_frame_counter % ARUCO_FRAME_SKIP == 0:
            # Process ArUco markers and get list of currently detected IDs
            depth_to_plot, detected_aruco_ids = detect_aruco_markers(raw_frame, raw_depth, aruco_detector, depth_to_plot)
            
            # Update marker history - reset counter for detected markers
            for aruco_id in detected_aruco_ids:
                aruco_marker_history[aruco_id] = 0
            
            # Increment counter for all markers not detected in this frame
            for aruco_id in list(aruco_marker_history.keys()):
                if aruco_id not in detected_aruco_ids:
                    aruco_marker_history[aruco_id] += 1
                    
                    # If marker hasn't been seen for ARUCO_PERSISTENCE_FRAMES, remove it
                    if aruco_marker_history[aruco_id] >= ARUCO_PERSISTENCE_FRAMES:
                        if aruco_id in globals.objects_data:
                            globals.objects_data.pop(aruco_id)
                        aruco_marker_history.pop(aruco_id)
        
        # Process YOLO results
        depth_to_plot = process_yolo_results(raw_frame, model, results, raw_depth, depth_to_plot, tracker)
        inference_time = time.time() - inference_start_time

        if has_dangerous_items(globals.objects_data):
            globals.is_warning = True
        else: 
            globals.is_warning = False

        # Determine volume multiplier based on whether guiding is active
        volume_multiplier = 0.5 if globals.is_guiding else 1.0

        if globals.is_warning:
            # print_dangerous_objects(globals.objects_data)
            test+=1
            print(test)
            for track_id, obj_data in globals.objects_data.items():
                if obj_data['isDangerous']:
                    print(f"ID: {track_id}, Class: {obj_data['class']}, Depth: {obj_data['depth']}, Danger: {obj_data['isDangerous']}")
                    
                    cls = obj_data['class'].strip().lower()
                    audio_data, samplerate = (
                        MODEL_NAMES_AUDIO.get(cls) or
                        (MODEL_NAMES_AUDIO['aruco_generic'] if cls.startswith('aruco_') else None)
                    )

                    if audio_data is None:
                        raise ValueError(f"No audio found for class '{cls}', and no fallback available.")

                    audio_data = resample_audio(audio_data, samplerate, SAMPLE_RATE)
                    hrtf_file, sound_is_flipped = get_HRTF_params(obj_data['y_angle'], obj_data['x_angle'], HRTF_DIR)
                    hrtf_input, hrtf_fs = sf.read(hrtf_file)  # Use soundfile to read the HRTF WAV file
                    audio_data = apply_hrtf(audio_data, SAMPLE_RATE, hrtf_input, hrtf_fs, sound_is_flipped, distance=1)
                    
                    # Apply volume adjustment based on depth and guiding state
                    base_volume = sigmoid_volume(obj_data['depth'], steepness=SIG_STEEP, midpoint=SIG_MID)
                    audio_data *= base_volume * 1# Reduce volume if guiding
                    
                    pygame_audio = convert_audio_format_to_pygame(audio_data, SAMPLE_RATE, SAMPLE_RATE)
                    pygame_audio = np.ascontiguousarray(pygame_audio, dtype=np.int16)
                    sound = pygame.sndarray.make_sound(pygame_audio)
                    
                    sound.play()
                    obj_data['sounded_already'] = True

        else:
            # This announces objects upon detection
            for track_id, obj_data in globals.objects_data.items():
                if not obj_data['sounded_already']:
                    # For ArUco markers, show marker ID
                    if isinstance(track_id, str) and track_id.startswith("aruco_"):
                        marker_id = obj_data.get('marker_id', 0)
                        print(f"ArUco Marker ID: {marker_id}, Depth: {obj_data['depth']}, "
                              f"X Angle: {obj_data['x_angle']}, Y Angle: {obj_data['y_angle']}")
                    else:
                        print(f"ID: {track_id}, Class: {obj_data['class']}, Depth: {obj_data['depth']}, "
                            f"Confidence: {obj_data['confidence']}, X Angle: {obj_data['x_angle']}, "
                            f"Y Angle: {obj_data['y_angle']}")
                    
                    class_name = obj_data['class'].strip().lower()
                    if class_name in MODEL_NAMES_AUDIO:

                        cls = obj_data['class'].strip().lower()
                        audio_data, samplerate = (
                            MODEL_NAMES_AUDIO.get(cls) or
                            (MODEL_NAMES_AUDIO['aruco_generic'] if cls.startswith('aruco_') else None)
                        )

                        if audio_data is None:
                            raise ValueError(f"No audio found for class '{cls}', and no fallback available.")

                        audio_data = resample_audio(audio_data, samplerate, SAMPLE_RATE)
                        hrtf_file, sound_is_flipped = get_HRTF_params(obj_data['y_angle'], obj_data['x_angle'], HRTF_DIR)
                        hrtf_input, hrtf_fs = sf.read(hrtf_file)  # Use soundfile to read the HRTF WAV file
                        audio_data = apply_hrtf(audio_data, SAMPLE_RATE, hrtf_input, hrtf_fs, sound_is_flipped, distance=1)
                        
                        # Apply volume adjustment based on depth and guiding state
                        base_volume = sigmoid_volume(obj_data['depth'], steepness=SIG_STEEP, midpoint=SIG_MID)
                        audio_data *= base_volume * volume_multiplier # Reduce volume if guiding
                        
                        pygame_audio = convert_audio_format_to_pygame(audio_data, SAMPLE_RATE, SAMPLE_RATE)
                        pygame_audio = np.ascontiguousarray(pygame_audio, dtype=np.int16)
                        sound = pygame.sndarray.make_sound(pygame_audio)
                        
                        sound.play()
                    
                    obj_data['sounded_already'] = True

            # Logic here could be simplified
            # Find the target to be tracked if in tracking mode
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

                target_sound_data[1] = min(1.0, 1.0 / (target_depth ** 2)) # depth to volume
                target_sound_data[2] = target_x_angle
                target_sound_data[3] = target_y_angle
                frequency_event.set()
            
        # End timing the entire cycle
        cycle_time = time.time() - cycle_start_time
        globals.total_cycle_time += cycle_time

        # Plotting stuff
        blank_img = np.zeros_like(raw_frame)

        # Tracking mask
        if globals.is_guiding and globals.current_target_to_guide is not None:
            colored_mask = np.zeros_like(raw_frame)
            colored_mask[target_mask_vis > 0] = WHITE  # White mask
            
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
        display_dict_info(objects_screen, globals.objects_data, excluding = ['mask_vis', 'sounded_already', 'y_angle'])


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

        # Increment frame counters for depth map and ArUco skipping
        frame_counter += 1
        aruco_frame_counter += 1

        # Break the loop on 'q' key press
        if (cv2.waitKey(1) & 0xFF == ord('q')) or globals.quit == True:
            print("quitting...")
            print_logo()
            print_block_letter_art("Bye guys")
            if cap.isOpened():  # Check if the capture object is open
                cap.release()  # Release the video capture
            pygame.quit()
            quit_app()

# end of program :)