import os
import subprocess
from gtts import gTTS

# Define the words
MODEL_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Generate TTS files and convert them to smaller OGG
for word in MODEL_NAMES:
    mp3_path = os.path.join(output_dir, f"{word}.mp3")
    ogg_path = os.path.join(output_dir, f"{word}.ogg")

    # Generate the MP3 file
    tts = gTTS(word, lang="en")
    tts.save(mp3_path)

    # Convert to low-bitrate OGG using ffmpeg
    subprocess.run([
        "ffmpeg", "-i", mp3_path, 
        "-codec:a", "libvorbis", 
        "-b:a", "16k",  # Lower bitrate (16 kbps)
        "-ar", "8000",  # Lower sample rate (8 kHz)
        "-q:a", "2",    # Lower quality setting
        ogg_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Remove the MP3 file to save space
    os.remove(mp3_path)

    print(f"Saved: {ogg_path}")

print("Optimized TTS OGG audio generation complete.")
