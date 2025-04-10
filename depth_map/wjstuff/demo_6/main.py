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

# Add to my_constants.py if not already there
DEPTH_MAP_FRAME_SKIP = 3  # Process depth map every N frames
ARUCO_FRAME_SKIP = 2  # Process ArUco detection every N frames (can be different from depth map skip)

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

def init_aruco_detector():
    """Initialize ArUco marker detector based on OpenCV version"""
    opencv_major_ver = int(cv2.__version__.split('.')[0])
    
    if opencv_major_ver >= 4:
        try:
            # Try newer API first (OpenCV 4.7+)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            def detect_func(img):
                return detector.detectMarkers(img)
                
        except AttributeError:
            # Fall back to older API (OpenCV 4.0-4.6)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            
            def detect_func(img):
                return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    else:
        # Very old OpenCV 3.x
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        
        def detect_func(img):
            return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    
    return detect_func

def detect_aruco_markers(frame, raw_depth, aruco_detector, depth_to_plot):
    """
    Detect ArUco markers in the frame and add them to globals.objects_data
    
    Args:
        frame: Input color frame
        raw_depth: Raw depth map
        aruco_detector: Function to detect ArUco markers
        depth_to_plot: Visualization frame to draw on
    """
    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, rejected = aruco_detector(gray)
    
    # Process detected markers
    if ids is not None and len(ids) > 0:
        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(depth_to_plot, corners, ids)
        
        # Process each detected marker
        for i in range(len(ids)):
            # Get marker ID
            marker_id = ids[i][0]
            
            # Get the corners of the marker
            corner = corners[i][0]
            corner = corner.astype(np.int32)
            
            # Get the bounding box
            x_min = int(min(corner[:, 0]))
            y_min = int(min(corner[:, 1]))
            x_max = int(max(corner[:, 0]))
            y_max = int(max(corner[:, 1]))
            
            # Calculate center
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            
            # Normalize to [0, 1]
            x_angle = x_center / frame.shape[1]
            y_angle = y_center / frame.shape[0]
            
            # Create a mask for this marker
            marker_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(marker_mask, [corner], 255)
            
            # Calculate depth
            avg_depth = process_depth_mask(raw_depth, marker_mask, frame.shape[:2])
            
            # Draw bounding box
            cv2.rectangle(depth_to_plot, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            # Add marker ID and depth info
            label = f"ArUco ID: {marker_id} {avg_depth:.2f}m"
            cv2.putText(depth_to_plot, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Store in globals with a special prefix to distinguish from regular objects
            track_id = f"aruco_{marker_id}"
            
            # Store object information
            globals.objects_data[track_id] = {
                'class': f"aruco_marker",  # Use a consistent class name for audio lookup
                'depth': float(avg_depth),
                'sounded_already': globals.objects_data.get(track_id, {}).get('sounded_already', False),
                'confidence': 1.0,  # ArUco markers are deterministic
                'mask_vis': marker_mask,
                'x_angle': float(x_angle),
                'y_angle': float(y_angle),
                'isDangerous': False,  # By default, ArUco markers are not considered dangerous
                'marker_id': int(marker_id)  # Store the marker ID separately
            }
    
    return depth_to_plot

# Modified process_yolo_results function to handle ArUco markers
def process_yolo_results(frame, model, results, raw_depth, depth_to_plot, tracker, aruco_detector=None):
    # Process ArUco markers if detector is provided
    if aruco_detector is not None:
        depth_to_plot = detect_aruco_markers(frame, raw_depth, aruco_detector, depth_to_plot)
    
    # Convert YOLO results to supervision Detections format
    detections = yolo_to_sv_detections(results)
    
    # Update tracks
    if len(detections) > 0:
        detections = tracker.update_with_detections(detections)
    
    # Store current sounded_already states before clearing
    sounded_states = {}
    for track_id, obj_data in globals.objects_data.items():
        if not (isinstance(track_id, str) and track_id.startswith("aruco_")):  # Skip ArUco markers
            sounded_states[track_id] = obj_data['sounded_already']
    
    # Clear previous objects info (except ArUco markers)
    for track_id in list(globals.objects_data.keys()):
        if not (isinstance(track_id, str) and track_id.startswith("aruco_")):
            globals.objects_data.pop(track_id)
    
    # Process each detection
    for i in range(len(detections)):
        # Get box coordinates
        box = detections.xyxy[i].astype(int)
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        x_angle = x_center / frame.shape[1]  # Normalize to [0, 1]
        y_angle = y_center / frame.shape[0]  # Normalize to [0, 1]
        
        # Get tracking ID
        track_id = detections.tracker_id[i]
        if track_id is None:
            continue
            
        # Get class information
        class_id = detections.class_id[i]
        class_name = model.names[class_id]
        confidence = detections.confidence[i]
        
        # Get segmentation mask for this object
        if hasattr(results, 'masks') and results.masks is not None:
            mask = results.masks.data[i].cpu().numpy()
            # Calculate average depth for this object
            avg_depth = process_depth_mask(raw_depth, mask, frame.shape[:2])
        else:
            avg_depth = 0
            mask_vis = None
        
        isDangerous = am_i_dangerous(avg_depth, class_name)

        # DRAWING STUFF MASKS INDIVIDUALLY
        # Create visualization mask
        mask_vis = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_vis = (mask_vis > 0).astype(np.uint8)
        # Draw mask
        colored_mask = np.zeros_like(frame)
        if not isDangerous:
            colored_mask[mask_vis > 0] = [0, 255, 0]  # Green mask
        else:
            colored_mask[mask_vis > 0] = [0, 0, 255]  # red mask BGR
        depth_to_plot = cv2.addWeighted(depth_to_plot, 1, colored_mask, 0.8, 0)
        # Draw bounding box
        cv2.rectangle(depth_to_plot, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Create label with depth
        label = f"{class_name} ({track_id}) {avg_depth:.2f}m"
        cv2.putText(depth_to_plot, label, (box[0], box[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store object information, preserving sounded_already state if it exists
        globals.objects_data[track_id] = {
            'class': class_name,
            'depth': float(avg_depth),
            'sounded_already': sounded_states.get(track_id, False),  # Get previous state or False if new
            'confidence': float(confidence),
            'mask_vis': mask_vis,
            'x_angle': float(x_angle),
            'y_angle': float(y_angle),
            'isDangerous' : isDangerous
        }
    
    return depth_to_plot


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
    
    # Make sure you have an audio file for ArUco markers
    # If not already in your audio dictionary, add a default sound for ArUco markers
    if 'aruco_marker' not in MODEL_NAMES_AUDIO:
        # Use a default sound or create one specifically for ArUco markers
        # For example, you could use the sound for a similar object or create a new one
        # This is just an example - replace with an appropriate sound
        if 'marker' in MODEL_NAMES_AUDIO:
            MODEL_NAMES_AUDIO['aruco_marker'] = MODEL_NAMES_AUDIO['marker']
        else:
            # Use any existing sound as a fallback - replace with something appropriate
            first_key = next(iter(MODEL_NAMES_AUDIO))
            MODEL_NAMES_AUDIO['aruco_marker'] = MODEL_NAMES_AUDIO[first_key]
    
    data, samplerate = MODEL_NAMES_AUDIO['bowl']
    print(data, samplerate)

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
        raw_depth, depth_to_plot = cached_raw_depth, cached_depth_to_plot
        depth_time = time.time() - depth_start_time

        # YOLO inference and time it
        inference_start_time = time.time()
        results = model(raw_frame, verbose=False)[0] #, conf=0.25
        
        # Only run ArUco detection every ARUCO_FRAME_SKIP frames to save processing time
        current_aruco_detector = None
        if aruco_frame_counter % ARUCO_FRAME_SKIP == 0:
            current_aruco_detector = aruco_detector
        
        depth_to_plot = process_yolo_results(raw_frame, model, results, raw_depth, depth_to_plot, tracker, current_aruco_detector)
        inference_time = time.time() - inference_start_time

        if has_dangerous_items(globals.objects_data):
            globals.is_warning = True
        else: 
            globals.is_warning = False

        if globals.is_warning:
            # print_dangerous_objects(globals.objects_data)
            test+=1
            print(test)
            for track_id, obj_data in globals.objects_data.items():
                if obj_data['isDangerous']:
                    print(f"ID: {track_id}, Class: {obj_data['class']}, Depth: {obj_data['depth']}, Danger: {obj_data['isDangerous']}")
                    
                    audio_data, samplerate = MODEL_NAMES_AUDIO[obj_data['class'].strip().lower()]
                    audio_data = resample_audio(audio_data, samplerate, SAMPLE_RATE)
                    hrtf_file, sound_is_flipped = get_HRTF_params(obj_data['y_angle'], obj_data['x_angle'], HRTF_DIR)
                    hrtf_input, hrtf_fs = sf.read(hrtf_file)  # Use soundfile to read the HRTF WAV file
                    audio_data = apply_hrtf(audio_data, SAMPLE_RATE, hrtf_input, hrtf_fs, sound_is_flipped, distance=1)
                    audio_data *= min(1.0, 1.0 / (obj_data['depth'] ** 2))
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
                        audio_data, samplerate = MODEL_NAMES_AUDIO[class_name]
                        audio_data = resample_audio(audio_data, samplerate, SAMPLE_RATE)
                        hrtf_file, sound_is_flipped = get_HRTF_params(obj_data['y_angle'], obj_data['x_angle'], HRTF_DIR)
                        hrtf_input, hrtf_fs = sf.read(hrtf_file)  # Use soundfile to read the HRTF WAV file
                        audio_data = apply_hrtf(audio_data, SAMPLE_RATE, hrtf_input, hrtf_fs, sound_is_flipped, distance=1)
                        audio_data *= min(1.0, 1.0 / (obj_data['depth'] ** 2))
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
            quit_app()

# end of program :)