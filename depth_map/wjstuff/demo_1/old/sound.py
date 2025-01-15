import soundfile as sf
import numpy as np
import argparse
import sounddevice as sd

def phase_shift_audio(input_file, output_file, shift_degrees):
    # Read the audio file
    audio_data, sample_rate = sf.read(input_file)

    # Calculate the phase shift in radians
    shift_radians = np.deg2rad(shift_degrees)

    # Create a time array for the audio data
    t = np.arange(audio_data.shape[0]) / sample_rate

    # Apply phase shift based on the sine and cosine components
    if len(audio_data.shape) == 1:  # Mono audio
        shifted_audio = np.real(audio_data * np.exp(1j * shift_radians))
    else:  # Stereo or multi-channel audio
        shifted_audio = np.array([
            np.real(channel * np.exp(1j * shift_radians)) for channel in audio_data.T
        ]).T

    # Mix the original and phase-shifted audio
    combined_audio = audio_data + shifted_audio

    # Normalize the mixed audio to prevent clipping
    max_val = np.max(np.abs(combined_audio))
    if max_val > 0:
        combined_audio = combined_audio / max_val

    # Write the mixed audio to the output file
    sf.write(output_file, combined_audio, sample_rate)
    print(f"Combined audio with {shift_degrees}-degree phase shift saved to {output_file}")

    # Play the combined audio
    print("Playing combined audio...")
    sd.play(combined_audio, samplerate=sample_rate)
    sd.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shift the phase of an audio file by a given degree and mix it with the original.")
    parser.add_argument("input_file", type=str, help="Path to the input audio file.")
    parser.add_argument("output_file", type=str, help="Path to save the combined audio file.")
    parser.add_argument("shift_degrees", type=float, help="Phase shift in degrees.")
    
    args = parser.parse_args()
    phase_shift_audio(args.input_file, args.output_file, args.shift_degrees)
