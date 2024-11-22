import json
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os

# Sample JSON data (replace with your actual data if needed)
detected_objects = [
    {
        "object": "car",
        "distance": 4.083666801452637,
        "angle": 0.5107421875
    },
    {
        "object": "person",
        "distance": 4.950667381286621,
        "angle": 0.0654296875
    },
    {
        "object": "car",
        "distance": 9.901939392089844,
        "angle": 0.7001953125
    }
]

# Function to generate spatially aware audio
def text_to_speech_proximity_spatial(objects, distances, positions, importance):
    combined_audio = AudioSegment.silent(duration=0)  # Start with an empty AudioSegment

    with tempfile.TemporaryDirectory() as temp_dir:
        for obj, dist, pos, imp in zip(objects, distances, positions, importance):
            base_volume = max(-30, -1 * dist)  # Distance-based volume adjustment
            adjusted_volume = base_volume + (imp / 10)  # Importance-based adjustment

            # Generate speech using gTTS
            tts = gTTS(text=obj, lang="en")
            tts_path = os.path.join(temp_dir, f"{obj}.mp3")
            tts.save(tts_path)

            # Load generated speech
            speech_audio = AudioSegment.from_file(tts_path)

            # Spatial panning
            if pos < 0.3:
                panned_audio = speech_audio.pan(-1)  # Left
            elif pos > 0.7:
                panned_audio = speech_audio.pan(1)   # Right
            else:
                panned_audio = speech_audio.pan(0)   # Center

            # Apply volume adjustments
            louder_audio = panned_audio + adjusted_volume
            smoother_audio = louder_audio.fade_in(50).fade_out(50)  # Smooth transitions

            combined_audio += smoother_audio

    # Export combined audio
    combined_audio.export("output.mp3", format="mp3")
    print("Generated MP3 file: output.mp3")
    play(combined_audio)

# Main function to process JSON data and generate audio
def process_json_and_generate_audio(data):
    # Extract fields from the JSON data
    objects = [item['object'] for item in data]
    distances = [item['distance'] for item in data]
    positions = [item['angle'] for item in data]  # 'angle' is equivalent to position
    importance = [10] * len(data)  # Default importance for now (can be dynamic)

    # Call the audio generation function
    text_to_speech_proximity_spatial(objects, distances, positions, importance)

# Run the process
if __name__ == "__main__":
    # Use the sample JSON data above
    process_json_and_generate_audio(detected_objects)
