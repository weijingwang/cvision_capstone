import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
import os

# Initialize pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set a faster speech rate

# Function to handle speech with volume adjustments and spatial channels
def text_to_speech_proximity_spatial(objects, distances, positions, importance):
    combined_audio = AudioSegment.silent(duration=0)  # Start with an empty AudioSegment

    # Iterate over the objects and queue them for speech output
    for obj, dist, pos, imp in zip(objects, distances, positions, importance):
        # Adjust volume more noticeably based on distance
        # Closer distances will be louder, and farther ones quieter
        base_volume = max(-30, -1 * dist)  # Convert distance to a negative dB adjustment

        # Adjust volume further by factoring in importance
        adjusted_volume = base_volume + (imp / 10)  # Importance scaled to adjust volume

        # Generate TTS speech and save it to a temporary "output.mp3" file
        output_filename = "output.mp3"
        engine.save_to_file(obj, output_filename)
        engine.runAndWait()

        speech_audio = AudioSegment.from_file(output_filename)

        # Spatial panning based on position: left, right, or center
        if pos < 0.3:
            # Pan to the left channel
            panned_audio = speech_audio.pan(-1)  # Left
        elif pos > 0.7:
            # Pan to the right channel
            panned_audio = speech_audio.pan(1)   # Right
        else:
            # Center channel
            panned_audio = speech_audio.pan(0)   # Center

        # Apply distance-based and importance-based volume adjustment
        louder_audio = panned_audio + adjusted_volume  # Adjust volume in dB
        smoother_audio = louder_audio.fade_in(50).fade_out(50)  # Smooth transition

        # Append the processed audio to the combined_audio segment
        combined_audio += smoother_audio

    # Export the final combined audio to "test.mp3"
    combined_audio.export("test.mp3", format="mp3")
    print("Generated MP3 file: test.mp3")

    # Play the generated audio file
    play(combined_audio)

# Example detected objects, distances, positions, and importance
detected_objects = ["car", "person", "car"]
distances = [4.08, 4.95, 9.9]  # Proximity in meters
positions = [0.51, 0.065, 0.7]  # Spatial position: left (0), center (0.5), right (1)
importance = [5, 10, 3]  # Importance levels

# Call the function to output spatially-aware, proximity and importance-based TTS
text_to_speech_proximity_spatial(detected_objects, distances, positions, importance)