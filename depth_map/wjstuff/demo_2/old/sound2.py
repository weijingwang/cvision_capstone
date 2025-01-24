import pygame
import numpy as np
import threading

# Initialize global variables for frequency, volume, and panning
frequency = 440.0  # Default frequency in Hz (A4)
volume = 0.5       # Default volume (0.0 to 1.0)
panning = 0.5      # Default panning (0.0 = left, 1.0 = right)

# Sample rate
sample_rate = 44100

# Initialize Pygame mixer
pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)

def generate_sine_wave(frequency, volume, panning, duration=0.1):
    """Generate a stereo sine wave with the given parameters."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t) * volume

    # Apply panning
    left = wave * (1 - panning)
    right = wave * panning

    # Combine into stereo
    stereo_wave = np.column_stack((left, right))
    return (stereo_wave * 32767).astype(np.int16)
def play_sine_wave():
    """Continuously play the sine wave while allowing real-time adjustments."""
    global frequency, volume, panning

    # Preinitialize the sound object
    duration = 1  # Short buffer duration for real-time updates
    sound = None

    while True:
        print("asdf")
        # Generate a short segment of the sine wave
        wave = generate_sine_wave(frequency, volume, panning, duration)
        if sound is None:
            sound = pygame.sndarray.make_sound(wave)
        else:
            sound.stop()
            sound = pygame.sndarray.make_sound(wave)
        
        sound.play(loops=0)
        pygame.time.wait(int(duration * 1000))

def input_listener(): # like an interrupt for key detection
    """Thread to listen for user input and update parameters."""
    global frequency, volume, panning
    while True:
        try:
            user_input = input("Enter 'f <freq>' for frequency, 'v <vol>' for volume, 'p <pan>' for panning: ")
            cmd, value = user_input.split()
            value = float(value)
            if cmd == 'f':
                frequency = max(20.0, min(value, 20000.0))  # Limit frequency range
                print(f"Frequency set to {frequency} Hz")
            elif cmd == 'v':
                volume = max(0.0, min(value, 1.0))  # Limit volume range
                print(f"Volume set to {volume}")
            elif cmd == 'p':
                panning = max(0.0, min(value, 1.0))  # Limit panning range
                print(f"Panning set to {panning} (0.0 = left, 1.0 = right)")
            else:
                print("Invalid command. Use 'f', 'v', or 'p'.")
        except Exception as e:
            print(f"Error: {e}. Use format 'f <freq>', 'v <vol>', or 'p <pan>'.")

# Start the input listener thread
thread = threading.Thread(target=input_listener, daemon=True)
thread.start()

print("Sine wave generator started. Adjust frequency, volume, and panning in real time.")
print("Commands: 'f <freq>' (frequency), 'v <vol>' (volume), 'p <pan>' (panning)")

# Play the sine wave continuously
try:
    play_sine_wave()
except KeyboardInterrupt:
    print("Exiting...")
    pygame.quit()
