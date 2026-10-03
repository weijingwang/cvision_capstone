"""Generate sounds"""
import pygame
import numpy as np
from scipy.signal import square
from my_constants import *
import globals

def generate_sound_wave(frequency, sample_rate, volume, panning, duration=0.1, squarewave=False):
    """Generate a stereo sine wave with the given parameters."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if squarewave:
        wave = square(2 * np.pi * frequency * t) * volume
    else:
        wave = np.sin(2 * np.pi * frequency * t) * volume
    # Apply panning
    left = wave * (1 - panning)
    right = wave * panning

    # Combine into stereo
    stereo_wave = np.column_stack((left, right))
    return (stereo_wave * 32767).astype(np.int16)

# Inverse square law function for volume
def calculate_volume(depth):
    if depth <= ARRIVAL_METERS: #1meter
        globals.voice_command = ''
        globals.arrived_at_target = True
        print('arrived at target- returning to main state')
        globals.state = 0 #move someplace else later
        return 1.0  # Max volume for distances <= 1 meter
    else:
        return min(1.0, 1.0 / (depth ** 2))  # Inverse square law for distances > 1 meter

def update_sound(depth_person, red_circle_position, frequency, apple_detected):
    volume = calculate_volume(depth_person)
    panning = max(0.0, min(red_circle_position, 1.0))  # Limit panning range
    wave = generate_sound_wave(frequency, SAMPLE_RATE, volume, panning, DURATION, squarewave=apple_detected)
    return wave

def play_sound(sound, wave):
    if sound is None:
        sound = pygame.sndarray.make_sound(wave)
    else:
        sound.stop()
        sound = pygame.sndarray.make_sound(wave)
    sound.play(loops=0)