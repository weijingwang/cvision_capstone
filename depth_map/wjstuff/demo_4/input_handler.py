"""Handle inputs like an interrupt
MY STATES input guide
[0] Main state
    - Continuously detect objects
        - If dangerous object detected lead user away with warning sound/drone
    - Announce important objects (stairs, walls)

    [1] Voice activated state
        - Else: Announce all the detected objects with appropriate volume and location
        - If say specific object: Play radar-like ping sound to lead user to object

"""

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
            if globals.state == 0:  # main state 0
                if user_input == '0':
                    globals.state = 0
                    print_notification("you're in the main state 0!")
                elif user_input == '1':
                    globals.state = 1
                    print_notification("voice mode: enter class name to guide to")
                # elif user_input == '2':
                #     globals.state = 2
                #     print("safety mode")
                else:
                    print_notification("Invalid key. Press '0' or '1'.")

            elif globals.state == 1: # waiting for input state 1
                if user_input == '':
                    print_notification('listing all the objects... returning to main state 0!')
                    # print(globals.objects_buffer)
                    globals.state = 0 #move someplace else later
                    
                elif is_word_in_set(user_input, MODEL_NAMES):
                    globals.voice_command = user_input
                    print_notification(f'guiding you to {globals.voice_command}... (state 2)')
                    globals.state = 2
                else:
                    print_notification("unknown object - try again (state 1)")
                    globals.state = 1 #move someplace else later
                    
                
            elif globals.state == 2: # wating for the guide to finish 2
                # if done guiding == TRUE:
                    # print('finished guiding you to target! returnung to mainstate 0!')
                    # globals.state = 0
                pass


        except Exception as e:
            print(f"Error: {e}")

