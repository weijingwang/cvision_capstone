import os
import subprocess
from gtts import gTTS

# Define the words
# MODEL_NAMES = [
#    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
#    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
#    'seventeen', 'eighteen', 'nineteen', 'twenty', 'twenty-one', 'twenty-two',
#    'twenty-three', 'twenty-four', 'twenty-five', 'twenty-six', 'twenty-seven',
#    'twenty-eight', 'twenty-nine', 'thirty', 'thirty-one', 'thirty-two',
#    'thirty-three', 'thirty-four', 'thirty-five', 'thirty-six', 'thirty-seven',
#    'thirty-eight', 'thirty-nine', 'forty', 'forty-one', 'forty-two', 'forty-three',
#    'forty-four', 'forty-five', 'forty-six', 'forty-seven', 'forty-eight', 'forty-nine'
# ]

MODEL_NAMES = [
	'exit', 'aipad'
]

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Generate TTS files and convert them to smaller OGG
for i, word in enumerate(MODEL_NAMES):
    mp3_path = os.path.join(output_dir, f"temp_{i}.mp3")
    ogg_path = os.path.join(output_dir, f"aruco_{i}.ogg")  # <-- renamed here

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
