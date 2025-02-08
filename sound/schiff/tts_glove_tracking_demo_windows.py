import numpy as np
import json
import re, os, subprocess, tempfile, time
from pathlib import Path
import librosa
import io
import soundfile as sf
from scipy import signal

# Import realtime TTS components from RealtimeTTS.
from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice, SystemEngine

#############################################
# 1. LOAD GLOVE EMBEDDINGS & CLASSIFY OBJECTS
#############################################

# Path to the GloVe embeddings file (update as needed)
GLOVE_FILE_PATH = (Path(__file__).resolve().parent / "glove.6B/glove.6B.100d.txt").resolve()

# Load the GloVe embeddings into a dictionary.
embeddings_dict = {}
with open(GLOVE_FILE_PATH, 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embeddings_dict[word] = vector

# Predefined categories (feel free to add or modify these)
categories = {
    "Vehicles": ["car", "bike", "bus", "train", "plane", "ship"],
    "Animals":  ["dog", "cat", "fish", "tiger", "bird"]
}

def compute_similarity(word1, word2):
    """Compute cosine similarity if both words are in the embeddings."""
    if word1 in embeddings_dict and word2 in embeddings_dict:
        vec1 = embeddings_dict[word1]
        vec2 = embeddings_dict[word2]
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return 0.0

def classify_word(word, similarity_threshold=0.5):
    """
    Classify the word into one of the predefined categories.
    If the word exactly matches or is semantically similar (above the threshold)
    to one of the words in a category, that category is returned.
    """
    word = word.lower()
    for cat, words in categories.items():
        if word in words:
            return cat
        for w in words:
            if compute_similarity(word, w) >= similarity_threshold:
                return cat
    return "Uncategorized"

#############################################
# 2. AUDIO DEMO, HRTF, AND PLAYBACK FUNCTIONS
#############################################

# Paths and constants (update as needed)
_BASE_DIR = Path(__file__).resolve().parent
_INPUT_DIR = (_BASE_DIR / "json_frames").resolve()   # New JSON frames go here.
_HRTF_DIR = (_BASE_DIR / "HRTF/MIT/diffuse").resolve()
_RADAR_SOUND_FILE = (_BASE_DIR / "sounds/radar-35955.wav").resolve()

# For Piper realtime TTS, set these paths (update as needed)
PIPER_EXECUTABLE = (_BASE_DIR / "piper/piper.exe").resolve()  # Ensure this is the Windows executable.
PIPER_MODEL = (_BASE_DIR / "piper/en_US-ryan-medium.onnx").resolve()
# Assume that the config file is in the same folder with the same base name and a .json suffix.
PIPER_CONFIG = (_BASE_DIR / "piper/en_US-ryan-medium.onnx.json").resolve()

def get_closest_hrtf_file(input_elevation, input_angle, base_dir):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"HRTF base directory does not exist: {base_dir}")

    elevation_pattern = r'elev(-?\d+)'
    closest_elevation_folder = None
    min_elevation_diff = float('inf')
    for folder_path in base_dir.glob("elev*"):
        folder_name = folder_path.name
        elevation_match = re.search(elevation_pattern, folder_name)
        if elevation_match:
            elevation = int(elevation_match.group(1))
            elevation_diff = abs(input_elevation - elevation)
            if elevation_diff < min_elevation_diff:
                min_elevation_diff = elevation_diff
                closest_elevation_folder = folder_path

    if closest_elevation_folder is None:
        print("No matching elevation folder found.")
        return None

    closest_file = None
    min_angle_diff = float('inf')
    angle_pattern = r'H(-?\d+)e(\d{3})a\.wav'
    for file_path in closest_elevation_folder.glob("*.wav"):
        file_name = file_path.name
        angle_match = re.search(angle_pattern, file_name)
        if angle_match:
            angle = int(angle_match.group(2))
            angle_diff = abs(input_angle - angle)
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                closest_file = file_path

    if closest_file is None:
        print("No matching angle file found.")
    else:
        print(f"Closest HRTF file: {closest_file}")

    return str(closest_file)

def fs_resample(s1, f1, s2, f2):
    if f1 != f2:
        if f2 < f1:
            s2 = librosa.core.resample(s2.T, orig_sr=f2, target_sr=f1).T
        else:
            s1 = librosa.core.resample(s1.T, orig_sr=f1, target_sr=f2).T
        fmax = max(f1, f2)
        f1 = fmax
        f2 = fmax
    return s1, f1, s2, f2

def apply_hrtf(wav_file, hrtf_file, is_flipped, distance):
    if hrtf_file is None:
        raise ValueError("HRTF file is None!")
    if not Path(hrtf_file).exists():
        raise FileNotFoundError(f"HRTF file does not exist at: {hrtf_file}")
    print(f"Applying HRTF using file: {hrtf_file}")
    HRIR, fs_H = sf.read(hrtf_file)
    sig_, fs_s = sf.read(wav_file)
    if len(sig_.shape) > 1 and sig_.shape[1] > 1:
        sig = np.mean(sig_, axis=1)
    else:
        sig = sig_
    sig, fs_s, HRIR, fs_H = fs_resample(sig, fs_s, HRIR, fs_H)
    if not is_flipped:
        s_L = signal.convolve(sig, HRIR[:, 0], method='auto')
        s_R = signal.convolve(sig, HRIR[:, 1], method='auto')
    else:
        s_L = signal.convolve(sig, HRIR[:, 1], method='auto')
        s_R = signal.convolve(sig, HRIR[:, 0], method='auto')
    scaling_factor = 1 / (distance**2) if distance > 0 else 1.0
    s_L *= scaling_factor
    s_R *= scaling_factor
    Bin_Mix = np.vstack([s_L, s_R]).T
    sf.write(wav_file, Bin_Mix, fs_s)
    return wav_file

#############################################
# 3. PROCESS A JSON FRAME & CREATE TASKS
#############################################

def process_json_frame(json_data, base_dir):
    """
    Given a JSON object (one frame of inputs), parse it and create tasks.
    Expected JSON keys: "object", "distance", "x_angle", "y_angle", "target", "moving_closer"
    """
    objects = []
    distances = []
    closest_files = []
    is_flipped = []
    target_flags = []   # explicit radar target
    urgent_flags = []   # urgent event (e.g. vehicle moving closer)

    for item in json_data:
        input_elevation = item.get("y_angle", 0.5) * 90
        input_angle_raw = item.get("x_angle", 0.5)
        if input_angle_raw <= 0.5:
            flip = False
            adjusted_angle = 90 - (input_angle_raw * 180)
        else:
            flip = True
            adjusted_angle = ((input_angle_raw - 0.5) * 180)

        closest_hrtf = get_closest_hrtf_file(input_elevation, adjusted_angle, base_dir)
        closest_files.append(closest_hrtf)

        obj_label = item.get("object", "unknown")
        objects.append(obj_label)
        distances.append(item.get("distance", 1.0))
        is_flipped.append(flip)

        json_target = item.get("target", False)
        moving_closer = item.get("moving_closer", False)
        is_vehicle = (classify_word(obj_label) == "Vehicles")

        if json_target:
            target_flags.append(True)
            urgent_flags.append(False)
        elif moving_closer and is_vehicle:
            target_flags.append(False)
            urgent_flags.append(True)
        else:
            target_flags.append(False)
            urgent_flags.append(False)

    tasks = []
    for obj, dist, hrtf_file, flip, is_target, is_urgent in zip(
            objects, distances, closest_files, is_flipped, target_flags, urgent_flags):
        text = obj
        tasks.append({
            "text": text,
            "closest_file": hrtf_file,
            "is_flipped": flip,
            "distance": dist,
            "is_target": is_target,
            "is_urgent": is_urgent
        })
    return tasks

#############################################
# 4. REALTIME TTS HELPER FUNCTION
#############################################

def synthesize_text(tts_stream, text):
    """
    Feeds the provided text to the realtime TTS stream and writes the output
    to a temporary WAV file. Returns the path to the generated WAV file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_filename = tmp.name
    # Feed the text (as an iterable) to the TTS stream.
    tts_stream.feed([text])
    # Synthesize and write the audio to the temporary file.
    tts_stream.play(output_wavfile=temp_filename, muted=True)
    return temp_filename

#############################################
# 5. CONTINUOUS PROCESSING OF FRAMES WITH REALTIME TTS
#############################################

def run_continuous(frames_source, base_dir, tts_stream):
    """
    frames_source should be an iterable or generator that yields JSON data (one frame at a time).
    For each frame, tasks are created and:
      - Radar tasks are handled by playing a radar sound.
      - TTS tasks use the realtime TTS stream to synthesize audio.
    """
    while True:
        for json_frame in frames_source:
            print("Processing new frame...")
            tasks = process_json_frame(json_frame, base_dir)
            for task in tasks:
                if task["is_target"]:
                    # Handle radar target tasks.
                    radar_wav_path = os.path.join(str(Path(tempfile.gettempdir())), "radar_temp.wav")
                    sf_data, sf_rate = sf.read(_RADAR_SOUND_FILE)
                    sf.write(radar_wav_path, sf_data, sf_rate)
                    num_loops_beep = 1
                    for i in range(num_loops_beep):
                        modified_radar_wav = apply_hrtf(radar_wav_path, task["closest_file"], task["is_flipped"], task["distance"])
                        subprocess.run(["ffplay", "-nodisp", "-autoexit", str(modified_radar_wav)], check=True)
                        time.sleep(0.25)
                    os.unlink(radar_wav_path)
                else:
                    # For TTS tasks, optionally prefix urgent text.
                    if task["is_urgent"]:
                        #output_text = "test"
                        output_text = f"Urgent: {task['text']} detected, {task['distance']} meters away."
                    else:
                        #output_text = "test"
                        output_text = task["text"]
                    # Synthesize TTS audio using the realtime TTS stream.
                    wav_file = synthesize_text(tts_stream, output_text)
                    # Apply HRTF to the synthesized audio.
                    if task["is_urgent"]:
                        modified_wav = wav_file
                    else:
                        modified_wav = apply_hrtf(wav_file, task["closest_file"], task["is_flipped"], task["distance"])
                    subprocess.run(["ffplay", "-nodisp", "-autoexit", str(modified_wav)], check=True)
                    os.unlink(wav_file)
            # Wait a bit before processing the next frame.
            time.sleep(1)

#############################################
# 6. EXAMPLE FRAME GENERATOR
#############################################

def frame_generator(input_dir):
    """
    A simple generator that polls an input directory for new JSON files (each representing a frame),
    yields their content, and (for testing) does not delete the file.
    """
    input_dir = Path(input_dir)
    while True:
        json_files = list(input_dir.glob("*.json"))
        if json_files:
            for jf in json_files:
                with open(jf, 'r') as f:
                    data = json.load(f)
                # For testing, do not delete the file:
                jf.unlink()
                yield data
        else:
            time.sleep(0.5)

#############################################
# 7. MAIN
#############################################

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    FRAMES_DIR = (BASE_DIR / "json_frames").resolve()
    FRAMES_DIR.mkdir(exist_ok=True)

    system_engine = SystemEngine()
    # Initialize realtime TTS with Piper.
    voice = PiperVoice(
        model_file=PIPER_MODEL,
        config_file=PIPER_CONFIG
    )
    engine = PiperEngine(
        piper_path=PIPER_EXECUTABLE,
        voice=voice
    )
    tts_stream = TextToAudioStream(system_engine)

    # Run the continuous processing loop with realtime TTS.
    run_continuous(frame_generator(FRAMES_DIR), _HRTF_DIR, tts_stream)
