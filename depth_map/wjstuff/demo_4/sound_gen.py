"""Generate sounds"""
import pygame
import numpy as np
from scipy.signal import square
from my_constants import *
import globals
from hrtf import *

def generate_sine_wave(frequency, duration, volume, x_angle, y_angle, sample_rate=44100):
    # HRTF stuff test
    hrtf_file, sound_is_flipped = get_HRTF_params(y_angle, x_angle, HRTF_DIR)
    print(hrtf_file, sound_is_flipped, x_angle, y_angle)
        
    """Generate a sine wave of a given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t) * volume
    
    # Read the HRTF data from the WAV file
    hrtf_input, hrtf_fs = sf.read(hrtf_file)  # Use soundfile to read the HRTF WAV file
    
    # Process the signal with HRTF
    processed_sound = apply_hrtf(wave, sample_rate, hrtf_input, hrtf_fs, sound_is_flipped, distance=1)

    return processed_sound  # Output as a NumPy array

#     return (stereo_wave * 32767).astype(np.int16)
# def generate_sine_wave(frequency, duration, volume, x_angle, y_angle, sample_rate=44100):
#     # # HRTF stuff test
#     # """
#     # [WARNING!!!!] CHECKS HRFT EVERY TIME EVEN WHEN NOT TRACKING! WHEN NOT TRACKING, x_angle, y_angle = 0!!!!
#     # """
#     hrtf_file, sound_is_flipped = get_HRTF_params(y_angle, x_angle, HRTF_DIR)
#     print(hrtf_file, sound_is_flipped, x_angle, y_angle)
        
#     """Generate a sine wave of a given frequency and duration."""
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     processed_sound = apply_hrtf(wav_file, hrtf_file, is_flipped, distance)
#     # wave = np.sin(2 * np.pi * frequency * t) * volume
#     # # Apply panning
#     # left = wave * (1 - x_angle)
#     # right = wave * x_angle
#     # stereo_wave = np.column_stack((left, right))
 
#     return processed_sound#(stereo_wave * 32767).astype(np.int16)

def play_sine_tone(frequency_event, target_sound_data):
    """Plays a sine tone every second with frequency controlled by target_sound_data."""
    duration = 0.1  # Duration of the tone (seconds)
    
    while True:
        frequency_event.wait()  # Wait for the signal to play the tone
        
        frequency = target_sound_data[0]  # Get the current frequency from shared variable
        volume = target_sound_data[1]
        x_angle = target_sound_data[2]
        y_angle = target_sound_data[3]
        sine_wave = generate_sine_wave(frequency, duration, volume, x_angle, y_angle)
        sine_wave = np.ascontiguousarray(sine_wave)
        sound = pygame.sndarray.make_sound((sine_wave * 32767).astype(np.int16))
        sound.play()  # Play the sine wave sound
        
        frequency_event.clear()  # Reset the event to allow future updates
        # No extra sleep here, just wait for the next event trigger



# def generate_sound_wave(frequency, sample_rate, volume, panning, duration=0.1):
#     """Generate a stereo sine wave with the given parameters."""
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     wave = np.sin(2 * np.pi * frequency * t) * volume
#     # Apply panning
#     # left = wave * (1 - panning)
#     # right = wave * panning

#     # Combine into stereo
#     # stereo_wave = np.column_stack((left, right))
#     stereo_wave = np.column_stack((wave, wave))

#     return (stereo_wave * 32767).astype(np.int16)

# # Inverse square law function for volume
# def calculate_volume(depth):
#     return min(1.0, 1.0 / (depth ** 2))  # Inverse square law for distances > 1 meter


# def play_tracking_tone(is_guiding, frequency_event, frequency_variable):
#     """Plays a sine tone every second with frequency controlled by frequency_variable."""
#     if is_guiding:
#         duration = 0.1  # Duration of the tone (seconds)
        
#         while True:
#             frequency_event.wait()  # Wait for the signal to play the tone
            
#             frequency = frequency_variable[0]  # Get the current frequency from shared variable
#             sine_wave = generate_sound_wave(frequency, duration)
#             sound = pygame.sndarray.make_sound(sine_wave)
#             sound.play()  # Play the sine wave sound
            
#             frequency_event.clear()  # Reset the event to allow future updates
#             # No extra sleep here, just wait for the next event trigger



# def update_sound(depth_person, x_angle, frequency):
#     volume = calculate_volume(depth_person)
#     panning = max(0.0, min(x_angle, 1.0))  # Limit panning range
#     wave = generate_sound_wave(frequency, SAMPLE_RATE, volume, panning, DURATION)
#     return wave

# def play_sound(sound, wave):
#     if sound is None:
#         sound = pygame.sndarray.make_sound(wave)
#     else:
#         sound.stop()
#         sound = pygame.sndarray.make_sound(wave)
#     sound.play(loops=0)