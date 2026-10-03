# CViSion

<!-- TODO: one-paragraph summary. CViSion is an audio guide for the visually impaired that uses object detection, depth maps and spatial (HRTF) audio. -->

<table>
<tr>
<td width="35%"><img src="assets/images/logo6.png" alt="CViSion logo"></td>
<td width="65%"><img src="assets/images/demo4.png" alt="CViSion demo"></td>
</tr>
</table>

**[Watch the trailer on YouTube](https://www.youtube.com/watch?v=NxqoAR_yYxM)**

## Installation

Run these from the project root:

```bash
# 1. Create the environment
conda create -n cvision python=3.11 -y
conda activate cvision

# 2. Install PyTorch first (Windows + NVIDIA: see note below)
pip install torch torchvision

# 3. Install the other dependencies
pip install -r requirements.txt

# 4. Download the models
git clone https://github.com/DepthAnything/Depth-Anything-V2 DA2
./checkpoints/download_ckpts.sh
```

**Windows + NVIDIA GPU:** in step 2, use the CUDA command from [pytorch.org](https://pytorch.org/get-started/locally/) instead. On Mac and Linux, the plain command already uses your GPU.

The YOLO model (`checkpoints/yolov8x-seg.pt`) downloads automatically on the first run.

## Running

```bash
python main.py
```
Always run it from the project root, because file paths are relative to it.

### Controls

<!-- TODO: check these are still accurate -->
- `0`: main state
- `1`: voice/command mode
  - type an object ID to be guided to it
  - type `a<ID>` (e.g. `a7`) to be guided to an ArUco marker
  - press Enter to list detected objects

## Features

<!-- TODO: fill in -->
- Normal mode: 
- Guide (tracking) mode: 
- Danger mode: 
- ArUco markers: printable markers are in `assets/aruco_markers` (DICT_4X4_50, IDs 0-49)

<details>
<summary><b>Settings</b></summary>

<!-- TODO: fill in. Key ones in my_constants.py: -->
- `WEBCAM_PATH`: camera index or video file
- `MIRROR_WEBCAM`: `True` for a laptop webcam facing you, `False` for a camera facing outward
- `DANGER_METER`, `ALWAYS_IGNORE`, `IGNORE_OBJECTS`

</details>

<details>
<summary><b>Project structure</b></summary>

<!-- TODO: fill in -->
```
main.py          # entry point
my_constants.py  # settings
globals.py       # shared state
cvision/         # 
assets/          # 
checkpoints/     # model weights (downloaded, not in git)
DA2/             # Depth-Anything-V2 (cloned, not in git)
demos/           # old demos, kept for reference
winter_report/   # 
```

</details>

<details>
<summary><b>Troubleshooting and known warnings</b></summary>

- **Check PyTorch sees your GPU:** `python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"`. The program uses CUDA, then MPS (Mac), then CPU.
- **Mac: PyTorch error about an unsupported MPS operation.** Run `export PYTORCH_ENABLE_MPS_FALLBACK=1` before `python main.py`.
- **`objc: Class SDL... is implemented in both ...`** (macOS): pygame and OpenCV each ship their own copy of SDL2. Safe to ignore.
- **`xFormers not available`**: an optional NVIDIA speed-up used by Depth-Anything. It isn't needed.
- **`ByteTrack was deprecated`**: supervision is pinned below 0.31 in `requirements.txt`, so it still works.

</details>

## Team

<!-- TODO -->

## Credits

- khw11044 for the tutorial on Depth Anything with a webcam: https://github.com/khw11044/Depth-Anything-V2-streaming
- marmik_ch19 for the command line fix for the PyTorch MPS error on Mac: https://www.reddit.com/r/pytorch/comments/1c3kwwg/how_do_i_fix_the_mps_notimplemented_error_for_m1/
- Warning sound by foosiemac: https://freesound.org/people/foosiemac/sounds/110395/
