from gtts import gTTS
from pydub import AudioSegment
from playsound import playsound
import tempfile
import os

def text_to_speech_proximity_spatial(objects, distances, angles):
    """
    Generates a combined audio output with proximity-based volume adjustment and spatial panning
    scaled by the given angles (normalized between 0 and 1).

    Parameters:
    - objects: List of strings (e.g., "car", "tree").
    - distances: List of floats (distance of each object in meters).
    - angles: List of floats (normalized between 0 and 1 for spatial panning).
    """
    combined_audio = AudioSegment.silent(duration=0)  # Start with an empty AudioSegment

    with tempfile.TemporaryDirectory() as temp_dir:
        for obj, dist, angle in zip(objects, distances, angles):
            base_volume = max(-30, -1 * dist)  # Distance-based volume adjustment
            adjusted_volume = base_volume + (dist / 10)  # Importance-based adjustment

            # Generate speech using gTTS
            tts = gTTS(text=obj, lang="en")
            tts_path = os.path.join(temp_dir, f"{obj}.wav")
            tts.save(tts_path)

            # Load generated speech
            speech_audio = AudioSegment.from_file(tts_path)

            # Apply spatial panning and volume adjustments
            panned_audio = speech_audio.pan(angle)
            louder_audio = panned_audio + adjusted_volume
            smoother_audio = louder_audio.fade_in(50).fade_out(50)  # Smooth transitions

            combined_audio += smoother_audio

        # Save the combined audio to a temporary file
        output_path = os.path.join(temp_dir, "output.wav")
        combined_audio.export(output_path, format="wav")
        print(f"Generated WAV file: {output_path}")

        # Play the audio using playsound
        playsound(output_path)

