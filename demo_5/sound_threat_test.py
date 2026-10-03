import pygame
import numpy as np
import threading
import time

# Initialize pygame
pygame.mixer.init(frequency=44100, size=-16, channels=2)

def generate_sine_wave(frequency, duration, sample_rate=44100):
    """Generate a sine wave of a given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767  # Sine wave formula

    audio_data = np.column_stack((audio_data, audio_data))
    audio_data = audio_data.astype(np.int16)
    return audio_data

def play_sine_tone(frequency_event, frequency_variable):
    """Plays a sine tone every second with frequency controlled by frequency_variable."""
    duration = 0.1  # Duration of the tone (seconds)
    
    while True:
        frequency_event.wait()  # Wait for the signal to play the tone
        
        frequency = frequency_variable[0]  # Get the current frequency from shared variable
        sine_wave = generate_sine_wave(frequency, duration)
        sound = pygame.sndarray.make_sound(sine_wave)
        sound.play()  # Play the sine wave sound
        
        frequency_event.clear()  # Reset the event to allow future updates
        # No extra sleep here, just wait for the next event trigger

def main_loop():
    """Simulates a busy main loop where the frequency changes based on some variable."""
    frequency_variable = [440]  # Start with 440 Hz (A4)
    frequency_event = threading.Event()  # Event to trigger tone playback

    # Start the sine tone thread
    tone_thread = threading.Thread(target=play_sine_tone, args=(frequency_event, frequency_variable))
    tone_thread.daemon = True  # Daemon thread will exit when the main program exits
    tone_thread.start()

    try:
        while True:
            # Simulate some busy loop, and change the frequency based on a condition
            frequency_variable[0] = (frequency_variable[0] + 10) % 1000  # Increase frequency by 10 Hz every loop
            print(f"Current Frequency: {frequency_variable[0]} Hz")

            # Trigger the sine tone to play with the updated frequency
            frequency_event.set()

            time.sleep(1)  # Sleep for 1 second before updating the frequency again
    except KeyboardInterrupt:
        print("Exiting program")

if __name__ == "__main__":
    main_loop()
