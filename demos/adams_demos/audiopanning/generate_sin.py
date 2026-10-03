import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment

# Parameters
duration_seconds = 60  # 1 minute
frequency = 440  # A4 note
sample_rate = 44100  # CD quality

# Generate the sine wave
t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # amplitude between -0.5 and 0.5

# Convert to 16-bit PCM format
wave_pcm = np.int16(wave * 32767)

# Save as a temporary WAV file
write("temp.wav", sample_rate, wave_pcm)

# Convert WAV to MP3 using pydub (requires ffmpeg installed)
sound = AudioSegment.from_wav("temp.wav")
sound.export("sine_440Hz.mp3", format="mp3")

print("Generated sine_440Hz.mp3")

