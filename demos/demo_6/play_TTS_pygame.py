import os
import pygame
import numpy as np
import soundfile as sf

# Initialize pygame mixer


# Function to load sound as a NumPy array
def load_sound_as_numpy(file_path):
    data, samplerate = sf.read(file_path, dtype='int16')  # Read as 16-bit PCM
    if len(data.shape) > 1:  # Convert stereo to mono if needed
        data = np.mean(data, axis=1, dtype=np.int16)
    return data, samplerate



# Load OGG file as NumPy array
file_path = "classnames_audio/bowl.ogg"
audio_data, sample_rate = load_sound_as_numpy(file_path)


pygame.mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
pygame.init()
# Convert to pygame sound object
sound = pygame.sndarray.make_sound(audio_data)

# Play the sound
sound.play()
pygame.time.delay(2000)  # Ensure sound has time to play

# Keep pygame running to avoid script exiting immediately
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
