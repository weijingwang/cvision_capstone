from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import os

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

# Example usage
detected_objects = ["car", "person", "tree"]
distances = [4.08, 2.5, 7.9]  # Distance in meters
positions = [0.1, 0.5, 0.8]  # Spatial positions: left (0), center (0.5), right (1)
importance = [5, 10, 3]  # Importance levels

text_to_speech_proximity_spatial(detected_objects, distances, positions, importance)
