import json
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os

# Audio generation function
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

# Pipeline to process input JSON and call the audio function
def process_input_and_generate_audio(json_file):
    # Read the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract fields from the JSON
    objects = [item['object'] for item in data]
    distances = [item['distance'] for item in data]
    positions = [item['angle'] for item in data]  # 'angle' is equivalent to position
    importance = [10] * len(data)  # Default importance for now (can be dynamic)

    # Call the audio generation function
    text_to_speech_proximity_spatial(objects, distances, positions, importance)

# Example usage
if __name__ == "__main__":
    # Assuming the JSON file is named 'detected_objects.json'
    process_input_and_generate_audio("detected_objects.json")
