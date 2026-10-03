# File:     demo_windows.ipynb
# Author:   Will Schiff
# Date:     1/30/2025
# Description: WINDOWS demo for running object tracking/text to speech with HRTF spatial audio 
# and XTTS, given an input JSON
# Ensure you have a folder containing the HRTF wav files, a JSON to run, 
# radar/sample voice files, ffmpeg installed, as well as tts imported in your environment
# my python version=3.9.21, torch=2.5.0+cu118

# note: this runs very slowly right now in the cmd line with xtts, likely to switch out models

# 2/7/2025 changed a lot ... -wj
from my_constants import *

from pathlib import Path
import numpy as np
import sys, glob, os, re, librosa, subprocess, json, tempfile, time
from pathlib import Path
import soundfile as sf
from scipy import signal
#from IPython.display import Audio 


# BASE_DIR = Path(__file__).resolve().parent

# _INPUT_FILE = (BASE_DIR / "testobjects.JSON").resolve()
# _HRTF_DIR = (BASE_DIR / "HRTF/MIT/diffuse").resolve()
# _XTTS2_PATH = "C:\\users\\swill\\anaconda3\\envs\\tts_env\\Scripts\\tts.exe"
# _TTS_MODEL_PATH = 'tts_models/multilingual/multi-dataset/xtts_v2'
# _VOICE_CLONE_FILE = (BASE_DIR / "sounds/sample_voice.wav").resolve()
# _RADAR_SOUND_FILE = (BASE_DIR / "sounds/radar-35955.wav").resolve()

# if not _INPUT_FILE.exists():
#     raise FileNotFoundError(f"Missing input file: {_INPUT_FILE}")
# if not Path(_XTTS2_PATH).exists():
#     raise FileNotFoundError(f"XTTS2 executable not found: {_XTTS2_PATH}")
# if not _RADAR_SOUND_FILE.exists():
#     raise FileNotFoundError(f"Missing radar sound file: {_RADAR_SOUND_FILE}")


def get_closest_hrtf_file(input_elevation, input_angle, base_dir):
    base_dir = Path(base_dir)  # Ensure base_dir is a pathlib Path

    if not base_dir.exists():
        raise FileNotFoundError(f"ERROR: HRTF base directory does not exist: {base_dir}")

    #print(f" Searching for closest HRTF file in: {base_dir}")

    elevation_pattern = r'elev(-?\d+)'
    closest_elevation_folder = None
    min_elevation_diff = float('inf')

    # Convert `base_dir` to a Windows-compatible string
    all_folders = list(base_dir.glob("elev*"))  # Uses pathlib for cross-platform compatibility
    #print(f"Found elevation folders: {all_folders}")  

    # Find the closest elevation folder
    for folder_path in all_folders:
        folder_name = folder_path.name  # Get the folder name
        elevation_match = re.search(elevation_pattern, folder_name)

        if elevation_match:
            elevation = int(elevation_match.group(1))
            elevation_diff = abs(input_elevation - elevation)

            #print(f"Elevation: {elevation}, Input: {input_elevation}, Difference: {elevation_diff}")

            if elevation_diff < min_elevation_diff:
                min_elevation_diff = elevation_diff
                closest_elevation_folder = folder_path

    if closest_elevation_folder is None:
        print("No matching elevation folder found.")
        return None

    #print(f"Closest elevation folder: {closest_elevation_folder}")

    # Now find the closest angle file within the chosen elevation folder
    closest_file = None
    min_angle_diff = float('inf')

    angle_pattern = r'H(-?\d+)e(\d{3})a\.wav'
    all_files = list(closest_elevation_folder.glob("*.wav"))  # Uses pathlib for Windows

    #print(f"Files found in {closest_elevation_folder}: {all_files}")

    for file_path in all_files:
        file_name = file_path.name
        angle_match = re.search(angle_pattern, file_name)

        if angle_match:
            angle = int(angle_match.group(2))
            angle_diff = abs(input_angle - angle)

            #print(f"Checking file '{file_name}' with angle {angle}, Difference: {angle_diff}")

            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                closest_file = file_path

    if closest_file is None:
        print("No matching angle file found.")
    else:
        pass
        # print(f"Closest HRTF file: {closest_file}")

    return str(closest_file)  # Ensure function returns a string path (for compatibility with soundfile)


"""
Get HRTF file and flipped for one object 
Original function from will: def process_json(input_file, base_dir)
2/7/2025 -wj 
"""
def get_HRTF_params(input_elevation, input_angle, base_dir): 
    input_elevation = input_elevation * 90 

    # Angle input mapping to match get_closest_hrtf_file function
    if (input_angle <= 0.5): 
        is_flipped = True
        adjusted_angle = 90 - (input_angle * 180)
    if (input_angle > 0.5):
        is_flipped = False
        adjusted_angle = ((input_angle - 0.5) * 180)
    
    closest_file = get_closest_hrtf_file(input_elevation, adjusted_angle, base_dir)

    return closest_file, is_flipped



def fs_resample(s1, f1, s2, f2): 
    if f1 != f2:
        if f2 < f1:
            s2 = signal.resample(s2.transpose(), int(len(s2) * f1 / f2)).T
        else:
            s1 = signal.resample(s1.transpose(), int(len(s1) * f2 / f1)).T
    fmax = max([f1, f2])
    f1 = fmax
    f2 = fmax
    return s1, f1, s2, f2


def apply_hrtf(signal_input, signal_fs, hrtf_input, hrtf_fs, is_flipped, distance):
    if hrtf_input is None:
        raise ValueError("ERROR: `hrtf_input` is None. Check how it's assigned!")

    if len(signal_input.shape) > 1 and signal_input.shape[1] > 1:  # Stereo audio
        sig = np.mean(signal_input, axis=1)  # Convert to mono by averaging channels
    else:  # Mono audio
        sig = signal_input

    [sig, signal_fs, hrtf_input, hrtf_fs] = fs_resample(sig, signal_fs, hrtf_input, hrtf_fs)

    if is_flipped == False: 
        s_L = signal.convolve(sig, hrtf_input[:, 0], method='auto')
        s_R = signal.convolve(sig, hrtf_input[:, 1], method='auto')
    else: 
        s_L = signal.convolve(sig, hrtf_input[:, 1], method='auto')
        s_R = signal.convolve(sig, hrtf_input[:, 0], method='auto')
        
    if distance > 0:
        scaling_factor = 1 / (distance**2)  # Inverse square law scaling
    else:
        scaling_factor = 1.0  # Prevent division by zero
    
    s_L *= scaling_factor
    s_R *= scaling_factor
    
    Bin_Mix = np.vstack([s_L, s_R]).transpose()
    
    return Bin_Mix


"""
Not working for me so comment out -wj 2/7/2025
"""
# def generate_and_play_audio_with_xtts2(tts_executable, model_name, temp_dir, tasks):
#     speaker_wav_file = _VOICE_CLONE_FILE
#     for task in tasks:
#         text_to_speak = task["text"]
#         closest_file = task["closest_file"]
#         is_flipped = task["is_flipped"]
#         distance = task["distance"]
#         is_target = task["is_target"]

#         if is_target:
#             radar_wav_path = os.path.join(temp_dir, "radar_temp.wav")

#             sf_data, sf_rate = sf.read(_RADAR_SOUND_FILE)
#             sf.write(radar_wav_path, sf_data, sf_rate)

#             num_loops_beep = 1
#             for i in range(num_loops_beep):
#                 modified_radar_wav = apply_hrtf(radar_wav_path, closest_file, is_flipped, distance)
#                 subprocess.run(["ffplay", "-nodisp", "-autoexit", str(modified_radar_wav)], check=True)
#                 time.sleep(0.25)

#             os.unlink(radar_wav_path)

#         else:
#             output_wav_path = os.path.join(temp_dir, "output.wav")

#             command = [
#                 tts_executable, 
#                 "--model_name", model_name,
#                 "--text", text_to_speak,
#                 "--speaker_wav", speaker_wav_file,
#                 "--language_idx", "en",
#                 "--out_path", output_wav_path,
#                 "--use_cuda", "true"
#             ]

#             try:
#                 subprocess.run(command, check=True)
                
#                 if not Path(output_wav_path).exists():
#                     raise FileNotFoundError(f"ERROR: TTS did not generate output file: {output_wav_path}")

#                 modified_wav = apply_hrtf(output_wav_path, closest_file, is_flipped, distance)

#                 # Play the modified audio using ffplay (Windows compatible)
#                 subprocess.run(["ffplay", "-nodisp", "-autoexit", str(modified_wav)], check=True)

#             finally:
#                 if Path(output_wav_path).exists():
#                     os.unlink(output_wav_path)


"""
TODO
"""


# Uncomment if you want to paste wav files in a temp dir
#test_dir = '/home/wrschiff/PycharmProjects/capstone/test_dir' 
def play_HRFT_sound(input_file, base_dir, xtts2_path, model_path):
    # Process JSON to extract necessary information
    objects, distances, closest_files, is_flipped, targets = process_json(input_file, base_dir)
    # Prepare tasks
    tasks = [
        {"text": obj, "closest_file": file, "is_flipped": flipped, "distance": dist, "is_target": is_target}
        for obj, file, dist, flipped, is_target in zip(objects, closest_files, distances, is_flipped, targets)
    ]
    # Uncomment if you want wav files in a temp dir
    #generate_and_play_audio_with_piper(piper_path, model_path, test_dir, tasks)

# play_HRFT_sound(_INPUT_FILE, _HRTF_DIR, _XTTS2_PATH, _TTS_MODEL_PATH)


