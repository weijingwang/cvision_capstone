# File:     demo.ipynb
# Author:   Will Schiff
# Date:     11/22/2024
# Description: demo for running piper TTS with HRTF spatial audio, given an input JSON
# Ensure you have a folder containing the HRTF wav files, a folder with the piper model, and a JSON to run

import numpy as np
import sys, glob, os, re, librosa, subprocess, json, tempfile, time
from pathlib import Path
import soundfile as sf
from scipy import signal
from IPython.display import Audio 

# Change to your paths 
_INPUT_FILE = 'your/path/testobjects.JSON'
_HRTF_DIR = 'your/path/HRTF/MIT/diffuse' 
_PIPER_PATH = 'your/path/piper/piper'
_MODEL_PATH = 'your/path/piper/en-us-ryan-high.onnx' 


def get_closest_hrtf_file(input_elevation, input_angle, base_dir):
    # Pattern to extract elevation from folder names (e.g., "elev30", "elev-20")
    elevation_pattern = r'elev(-?\d+)'
    
    closest_elevation_folder = None
    min_elevation_diff = float('inf')
    
    # List all folders in base directory for debugging
    #all_items = os.listdir(base_dir)
    #print("All items in base directory:", all_items)   
    all_folders = glob.glob(os.path.join(base_dir, 'elev*/'))
    all_folders.sort()
    #print("All folders found:", all_folders)  # Debugging line
    
    # Find the closest elevation folder
    for folder_path in all_folders:
        folder_name = os.path.basename(folder_path.rstrip('/'))
        #print("Checking folder:", folder_name)  # Debugging line
        
        # Match the elevation pattern
        elevation_match = re.search(elevation_pattern, folder_name)
        
        if elevation_match:
            elevation = int(elevation_match.group(1))
            elevation_diff = abs(input_elevation - elevation)
            
            #print(f"File: {folder_name}, Angle: {input_elevation}, Difference: {elevation_diff}")# Debugging line
            
            if elevation_diff < min_elevation_diff:
                min_elevation_diff = elevation_diff
                closest_elevation_folder = folder_path

    # Check if a matching elevation folder was found
    if closest_elevation_folder is None:
        print("No matching elevation folder found.")
        return None

    #print("Closest elevation folder:", closest_elevation_folder)  # Debugging line

    # Now find the closest angle file within the chosen elevation folder
    closest_file = None
    min_angle_diff = float('inf')
    
    for file_path in glob.glob(os.path.join(closest_elevation_folder, '*.wav')):
        file_name = os.path.basename(file_path)
        angle_match = re.search(r'H(-?\d+)e(\d{3})a\.wav', file_name)
        
        if angle_match:
            angle = int(angle_match.group(2))  # Extract the horizontal angle
            angle_diff = abs(input_angle - angle)
            
            #print(f"Checking file '{file_name}' with angle {angle}")  # Debugging line
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                closest_file = file_path

    return closest_file


def process_json(input_file, base_dir): 
    with open(input_file, 'r') as file:
        data = json.load(file)
    objects = []
    distances = []
    closest_files = []
    is_flipped = []

    for item in data: 
        input_elevation = item.get("y_angle")
        # Assuming elevation angle is between -1 and 1 (although lowest HRTF file is at -0.2)
        input_elevation = input_elevation * 90 
        # Assuming input azimuth angle is between 0 and 1
        input_angle = item.get("x_angle") 
        
        # Angle input mapping to match get_closest_hrtf_file function
        if (input_angle <= 0.5): 
            flip = False
            adjusted_angle = 90 - (input_angle * 180)
        if (input_angle > 0.5):
            flip = True
            adjusted_angle = ((input_angle - 0.5) * 180)
            
        closest_files.append(get_closest_hrtf_file(input_elevation, adjusted_angle, base_dir))
        objects.append(item.get("object"))
        distances.append(item.get("distance"))
        is_flipped.append(flip)
    
    return objects, distances, closest_files, is_flipped


# Maybe change to scipy.signal resample because it's faster, this is higher quality
def fs_resample(s1,f1,s2,f2): 
    if f1 != f2:
        if f2 < f1:
            s2 = librosa.core.resample(s2.transpose(),orig_sr=f2,target_sr=f1).T
            s2 = s2.transpose
        else:
            s1 = librosa.core.resample(s1.transpose(),orig_sr=f1,target_sr=f2).T
            s1 = s1.transpose()
    fmax = max([f1, f2])
    f1 = fmax
    f2 = fmax
    #print('Resampled at: ', fmax, 'Hz')
    return s1, f1, s2, f2


def apply_hrtf(wav_file, hrtf_file, is_flipped, distance):
    [HRIR, fs_H] = sf.read(hrtf_file)
    [sig, fs_s] = sf.read(wav_file)
    
    if len(sig.shape) > 1 and sig.shape[1] > 1:  # Stereo audio
        sig_ = np.mean(sig, axis=1)  # Convert to mono by averaging channels
    else:  # Mono audio
        sig_ = sig
    
    [sig, fs_s, HRIR, fs_H] = fs_resample(sig, fs_s, HRIR, fs_H)
    # HRTF angle goes from 0 (head on) to 90 (right)
    # so if angle is on left side, utilize symmetry and flip channels
    if is_flipped == False: 
        s_L = signal.convolve(sig,HRIR[:,0], method='auto')
        s_R = signal.convolve(sig,HRIR[:,1], method='auto')
    else: 
        s_L = signal.convolve(sig,HRIR[:,1], method='auto')
        s_R = signal.convolve(sig,HRIR[:,0], method='auto')
        
    if distance > 0:
        scaling_factor = 1 / (distance**2)  # Inverse square law scaling
    else:
        scaling_factor = 1.0  # Prevent division by zero
    
    s_L *= scaling_factor
    s_R *= scaling_factor
    
    Bin_Mix = np.vstack([s_L,s_R]).transpose()
    
    sf.write(wav_file,Bin_Mix, fs_s)
    
    return wav_file


def generate_and_play_audio_with_piper(piper_path, model_path, temp_dir, tasks):
    command = [piper_path, "--model", model_path, "--output_dir", str(temp_dir)]  
    
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    ) as proc:
        for task in tasks:
            # Extract details for the current object
            text_to_speak = task["text"]
            closest_file = task["closest_file"]
            is_flipped = task["is_flipped"]
            distance = task["distance"]

            # Generate, process, and play audio for this object
            print(text_to_speak, file=proc.stdin, flush=True)
            wav_file = proc.stdout.readline().strip()
            #print(f"Generated WAV file at: {wav_file}")

            # Apply HRTF
            modified_wav = apply_hrtf(wav_file, closest_file, is_flipped, distance)
            
            # Play the modified audio
            try: 
                result = subprocess.run(["aplay", "-D", "default", modified_wav],check=True)
            # Clean up temporary files
            finally: 
                os.unlink(wav_file)
            
            #time.sleep(1) #for delay between each sound file


# Uncomment if you want to paste wav files in a dir
#test_dir = '/home/wrschiff/PycharmProjects/capstone/test_dir' 
def main(input_file, base_dir, piper_path, model_path):
    # Process JSON to extract necessary information
    objects, distances, closest_files, is_flipped = process_json(input_file, base_dir)
    
    # Prepare tasks
    tasks = [
        {"text": obj, "closest_file": file, "is_flipped": flipped, "distance": dist}
        for obj, file, dist, flipped in zip(objects, closest_files, distances, is_flipped)
    ]
    # Uncomment if you want wav files in a dir
    #generate_and_play_audio_with_piper(piper_path, model_path, test_dir, tasks)
    
    # Temporary directory for Piper outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate and play audio using Piper
        generate_and_play_audio_with_piper(piper_path, model_path, temp_dir, tasks)



main(_INPUT_FILE, _HRTF_DIR, _PIPER_PATH, _MODEL_PATH)
