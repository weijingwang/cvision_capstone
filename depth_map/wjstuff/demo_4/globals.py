"""global vars"""
from my_constants import MAX_SINE_VOLUME
quit = False
state = 0
voice_command = ''
arrived_at_target = True
objects_buffer = [] # going to be obsolete
objects_data = {} #dictionaries of IDs from tracking and all the parameters [describe later]
announce_state = 0 # 0 is no danger or important object. 1 is important. 2 is danger.
is_guiding = False
current_target_to_guide = None

# Demo variables
frequency = 440.0  # Default frequency in Hz (A4)
volume = MAX_SINE_VOLUME       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)
sound = None
danger_detected = False
important_detected = False

# GUI
button_is_pressed = False
last_click_time = 0
is_double_clicked = False
is_held = False  # Tracks if the button is being held

# Performance timing
total_cycle_time = 0
total_inference_time = 0