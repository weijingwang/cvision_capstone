import globals
from my_constants import *
from utils import *

def input_listener():  # Function to listen for specific key inputs
    """Thread to listen for key inputs and print specific outputs."""
    # state = 0
    # voice_command = ''
    # objects_buffer = []

    while True:
        try:
            user_input = input("").strip().lower()
            if user_input == 'quit':
                globals.quit = True
            if not globals.quit:
                if globals.state == 0:  # main state 0
                    if user_input == '0':
                        globals.state = 0
                        # Reset all sounded_already flags to False so objects will be announced again
                        reset_all_sounded_flags()
                        print_notification("you're in the main state 0!")
                    elif user_input == '1':
                        globals.state = 1
                        print_notification("voice mode (state 1): enter ID to guide to (number for objects or 'a123' for ArUco marker 123)")
                    # elif user_input == '2':
                    #     globals.state = 2
                    #     print("safety mode")
                    else:
                        print_notification("Invalid key. Press '0' or '1'.")

                elif globals.state == 1: # waiting for input state 1
                    if user_input == '0':
                        print_notification("cancelling... returning to main state 0!")
                        globals.state = 0
                        # Reset all sounded_already flags when returning to main state
                        reset_all_sounded_flags()
                        
                    elif user_input == '':
                        print_notification('listing all the objects... returning to main state 0!')
                        # Print all available objects including ArUco markers
                        print_available_objects(globals.objects_data)
                        globals.state = 0 #move someplace else later
                    
                    # Check if input is for an ArUco marker (format: a123 for ArUco ID 123)
                    elif user_input.startswith('a') and user_input[1:].isdigit():
                        aruco_id = int(user_input[1:])
                        aruco_key = f"aruco_{aruco_id}"
                        
                        if is_key_in_dict(aruco_key, globals.objects_data):
                            globals.voice_command = f"ArUco marker {aruco_id}"
                            globals.is_guiding = True
                            globals.current_target_to_guide = aruco_key
                            print_notification(f'guiding you to {globals.voice_command}... (state 2)')
                            globals.state = 2
                        else:
                            print_notification(f"ArUco marker {aruco_id} not found - try again (state 1)")
                            globals.state = 1
                            
                    # Regular numeric object ID
                    elif user_input.isdigit() and is_key_in_dict(int(user_input), globals.objects_data):
                        globals.voice_command = user_input
                        globals.is_guiding = True
                        globals.current_target_to_guide = int(user_input)
                        print_notification(f'guiding you to {globals.voice_command}... (state 2)')
                        globals.state = 2
                    else:
                        print_notification("unknown object - try again (state 1)")
                        globals.state = 1 #move someplace else later
                        
                    
                # More state 2 is found in main.py because needs to be polled
                elif globals.state == 2: # waiting for input state 1
                    if user_input == '0':
                        globals.state = 0
                        globals.current_target_to_guide = None
                        globals.is_guiding = False
                        # Reset all sounded_already flags when cancelling guidance
                        reset_all_sounded_flags()
                        print_notification("cancelling... returning to main state 0!")
                        print_menu()

        except Exception as e:
            print(f"Error: {e}")


def reset_all_sounded_flags():
    """Reset all sounded_already flags to False in objects_data dictionary"""
    for track_id in globals.objects_data:
        if 'sounded_already' in globals.objects_data[track_id]:
            globals.objects_data[track_id]['sounded_already'] = False
            
def print_available_objects(objects_data):
    """Print available objects for tracking in a clear format"""
    print("\n===== AVAILABLE OBJECTS =====")
    print("YOLO Objects (enter ID number):")
    has_yolo = False
    for track_id, obj_data in objects_data.items():
        if isinstance(track_id, int) or (isinstance(track_id, str) and not track_id.startswith("aruco_")):
            has_yolo = True
            print(f"  ID: {track_id}, Class: {obj_data['class']}, Depth: {obj_data['depth']:.2f}m")
    
    if not has_yolo:
        print("  No YOLO objects detected")
        
    print("\nArUco Markers (enter 'a' followed by marker ID):")
    has_aruco = False
    for track_id, obj_data in objects_data.items():
        if isinstance(track_id, str) and track_id.startswith("aruco_"):
            has_aruco = True
            marker_id = track_id.split("_")[1]
            print(f"  Enter: a{marker_id}, ArUco ID: {marker_id}, Depth: {obj_data['depth']:.2f}m")
    
    if not has_aruco:
        print("  No ArUco markers detected")
    
    print("===============================")