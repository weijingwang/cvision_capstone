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

global state
global voice_command


def input_listener():  # Function to listen for specific key inputs
    """Thread to listen for key inputs and print specific outputs."""
    state = 0
    voice_command = ''
    print("Press '0 for main' or '1 for voice': ")
    while True:
        try:
            user_input = input("").strip().lower()
            if state == 0:
                if user_input == '0':
                    state = 0
                    print("main")
                elif user_input == '1':
                    state = 1
                    print("voice")
                else:
                    print("Invalid key. Press '0' or '1'.")

            elif state == 1:
                if user_input == '':
                    print('listing all the objects...')
                    
                else:
                    voice_command = user_input
                    print(f'guiding you to {voice_command} and then it finished')
                state = 0 #move someplace else later

                
        
        except Exception as e:
            print(f"Error: {e}")





# def input_listener():  # Function to listen for specific key inputs
#     """Thread to listen for key inputs and print specific outputs."""
#     print("Press '0 for main' or '1 for voice': ")
#     while True:
#         try:
#             user_input = input("").strip().lower()
#             if state == 0:
#                 if user_input == '0':
#                     state = 0
#                     print("main")
#                 elif user_input == '1':
#                     state = 1
#                     print("voice")
#                 else:
#                     print("Invalid key. Press '0' or '1'.")

#             elif state == 1:
#                 if user_input == '':
#                     print('listing all the objects...')
                    
#                 else:
#                     voice_command = user_input
#                     print(f'guiding you to {voice_command} and then it finished')
#                 state = 0 #move someplace else later

                
        
#         except Exception as e:
#             print(f"Error: {e}")

