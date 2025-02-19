import cv2
from my_constants import *
import pyfiglet
import pygame
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import os

import numpy as np
from scipy import signal


import numpy as np
from scipy import signal

def resample_audio(audio_data, original_rate=8000, target_rate=44100):
    """
    Resamples audio data from one sample rate to another.
    
    Args:
        audio_data (np.ndarray): Input audio data
        original_rate (int): Original sampling rate (default: 8000)
        target_rate (int): Target sampling rate (default: 44100)
    
    Returns:
        np.ndarray: Resampled audio data
    """
    # Calculate the number of samples needed for the new sample rate
    # new_samples = original_samples * (new_rate / old_rate)
    num_samples = int(len(audio_data) * target_rate / original_rate)
    
    # Resample the audio data
    resampled_audio = signal.resample(audio_data, num_samples)
    
    return resampled_audio

# Example usage:
# original_audio = np.array([...])  # Your 8000 Hz audio data
# resampled_audio = resample_audio(original_audio)  # Converts to 44100 Hz


def convert_audio_format_to_pygame(audio_data, original_rate, target_rate):
    """
    Converts audio to 16-bit stereo at the specified sample rate.
    
    Args:
        audio_data (np.ndarray): Input audio data
        original_rate (int): Original sampling rate
        target_rate (int): Desired sampling rate
        
    Returns:
        np.ndarray: Resampled 16-bit stereo audio data
    """
    # Ensure input is a numpy array
    audio_data = np.array(audio_data)
    
    # Resample if rates don't match
    if original_rate != target_rate:
        # Calculate number of samples needed
        num_samples = int(len(audio_data) * target_rate / original_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    # Convert to 16-bit range (-32768 to 32767)
    if audio_data.dtype != np.int16:
        # Normalize to [-1, 1] first if not already
        if audio_data.max() > 1 or audio_data.min() < -1:
            audio_data = audio_data / np.max(np.abs(audio_data))
        # Convert to 16-bit
        audio_data = (audio_data * 32767).astype(np.int16)
    
    # Convert to stereo if mono
    if len(audio_data.shape) == 1:
        audio_data = np.column_stack((audio_data, audio_data))
    
    return audio_data


def load_sound_as_numpy(file_path):
    data, samplerate = sf.read(file_path, dtype='int16')  # Read as 16-bit PCM
    if len(data.shape) > 1:  # Convert stereo to mono if needed
        data = np.mean(data, axis=1, dtype=np.int16)
    return data, samplerate

def create_audio_dictionary(folder_path):
    """
    Creates a dictionary of audio data from all .ogg files in the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing .ogg files
        
    Returns:
        dict: Dictionary with filenames (without .ogg) as keys and tuples of (data, samplerate) as values
    """
    audio_dict = {}
    folder = Path(folder_path)
    
    # Process all .ogg files in the folder
    for file_path in folder.glob('*.ogg'):
        # Get filename without extension
        key = file_path.stem
        
        try:
            # Load the audio data
            data, samplerate = load_sound_as_numpy(file_path)
            audio_dict[key] = (data, samplerate)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return audio_dict


def quit_app():
    cv2.destroyAllWindows()  # Close any OpenCV windows
    pygame.quit()  # Uninitialize Pygame
    sys.exit()  # Exit the program


def print_block_letter_art(text):
    ascii_art = pyfiglet.figlet_format(text)
    print(ascii_art)

def is_word_in_set(input_word, word_set):
    return input_word in word_set

def is_key_in_dict(user_input, dictionary):
    return user_input in dictionary

def print_menu():
    menu = """
    ┌───────────────────────────────────┐
    │           MENU OPTIONS            │
    ├───────────────────────────────────┤
    │ Enter 0 → Main State              │
    │ Enter 1 → Voice Activation        │
    │ Enter quit → quit program         │
    └───────────────────────────────────┘
    """
    print(menu)

def print_logo():
    logo = """@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@
@@@@@@.    @@@@@@@@   @@@@@@@@@@    @@@:   @@@@@@@
@@@@@      @@@@@@      @@@@@@@@              @@@@@
@@@@       @@@@@       @@@@@@@@@@@@          @@@@@
@@@@                  @@@@@@@@@@@@@@@@@@@@  @@@@@@
@@@                    @@@@@@@@@@@@@@@@@@@@@@@@@  
@@           @@@@@@@@@     @@@@@@@@@@@@@@@@@@@  @@
@          @@@@ @@@@@@@@      @@@@@@@@@@@@@@  @@@@
@          @@   @@@@@  @@     @@@@@@@@@@@:   @@@@@
@                @@@         @@@@@@@@@@@@@@@@@@@@@
@                           @@@@@@@@@@@@@@@@@@@@@@
@                             @@@@@@@@@@@@@@@@@@@@
@                                %@@@@@@@@@@@@@@@@"""
    print(logo)




def print_notification(message):
    border = "─" * (len(message) + 4)
    notification = f"""
    ┌{border}┐
    │  {message}  │
    └{border}┘
    """
    print(notification)



def add_performance_text(raw_frame, performance_text):
    # # Get the text size and calculate the background rectangle
    # text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0] for line in performance_text]
    # max_width = max(w for w, h in text_sizes)
    # total_height = sum(h for w, h in text_sizes) + len(performance_text) * 10  # Add some padding

    # # Draw the background rectangle
    # rect_x = 0
    # rect_y = 30
    # rect_width = max_width + 20  # Add padding to the width
    # rect_height = total_height + 20  # Add padding to the height
    # cv2.rectangle(raw_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)  # Black rectangle

    # Put text on the image
    for i, line in enumerate(performance_text):
        cv2.putText(raw_frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return raw_frame

def display_dict_info(frame, data_dict, position=(10, 30), font_scale=0.7, font_color=(0, 255, 0), font_thickness=2, excluding=[]):
    """
    Displays the contents of a dictionary on an image/frame with numbers formatted to 1 decimal places.
    :param frame: The image/frame on which to display the dictionary
    :param data_dict: The dictionary whose contents will be displayed
    :param position: The starting position for displaying the text (default is top-left corner)
    :param font_scale: The scale of the text (default is 0.7)
    :param font_color: The color of the text (default is green)
    :param font_thickness: The thickness of the text (default is 2)
    :param excluding: List of keys to exclude from display
    """
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.1f}"
        elif isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items() if k not in excluding}
        elif isinstance(value, (list, tuple)):
            return [format_value(v) for v in value]
        return value

    # Filter out excluded keys and format the dictionary
    filtered_dict = {key: format_value(value) 
                    for key, value in data_dict.items() 
                    if key not in excluding}
    
    # Format the dictionary content into a string to display
    dict_info = "\n".join([f"{key}: {value}" for key, value in filtered_dict.items()])
    
    # Split the string into lines for better formatting if it becomes too long
    lines = dict_info.split('\n')
    
    # Draw each line of the dictionary on the frame
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (position[0], position[1] + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)