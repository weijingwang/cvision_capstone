"""Generate sounds"""

import numpy as np
from scipy.signal import square

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