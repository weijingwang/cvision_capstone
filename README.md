# CViSion

<!-- TODO: one-paragraph summary. CViSion is an audio guide for the visually impaired that uses object detection, depth maps and spatial (HRTF) audio. -->

<img src="assets/images/logo6.png" alt="CViSion logo" width="300">

![Demo](assets/images/demo4.png)

## Features

<!-- TODO: fill in -->
- Normal mode: 
- Guide (tracking) mode: 
- Danger mode: 
- ArUco markers: 

## Project structure

<!-- TODO: fill in -->
```
main.py          # entry point
my_constants.py  # settings (camera, danger distance, ignored classes, ...)
globals.py       # shared state
cvision/         # 
assets/          # 
checkpoints/     # depth model weights (downloaded, not in git)
DA2/             # Depth-Anything-V2 (cloned, not in git)
demos/           # old demos, kept for reference
```

## Installation

Tested with Python 3.11 and conda.

### 1. Create the environment

```bash
conda create -n cvision python=3.11 -y
conda activate cvision
```

### 2. Install PyTorch (pick your machine)

Install PyTorch **before** `requirements.txt`. Otherwise pip may pull in a build that doesn't match your GPU.

**Mac (Apple Silicon):** the normal build already supports the Mac GPU (MPS).
```bash
pip install torch torchvision
```

**Linux with an NVIDIA GPU:** the normal build from PyPI already includes CUDA.
```bash
pip install torch torchvision
```

**Windows with an NVIDIA GPU:** the PyPI build is CPU-only, so use PyTorch's CUDA index.
Get the exact command for your CUDA version from https://pytorch.org/get-started/locally/. It looks like:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

**No GPU:** `pip install torch torchvision` works, but it will be slow.

Check it worked:
```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), 'MPS:', torch.backends.mps.is_available())"
```
The program picks CUDA, then MPS, then CPU automatically.

### 3. Install the other dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Depth-Anything-V2

`DA2/` is not in git. From the project root, clone it straight into a folder named `DA2`:
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 DA2
```

### 5. Download the depth checkpoint

The program loads `checkpoints/depth_anything_v2_metric_hypersim_vitb.pth` (metric, Base size).
```bash
./checkpoints/download_ckpts.sh
```

The YOLO model (`yolov8x-seg.pt`) downloads automatically on the first run.

## Running

Run from the project root, because asset paths are relative to it:
```bash
python main.py
```

On a Mac, if PyTorch errors on an unsupported MPS operation:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Controls

<!-- TODO: check these are still accurate -->
- `0`: main state
- `1`: voice/command mode
  - type an object ID to be guided to it
  - type `a<ID>` (e.g. `a7`) to be guided to an ArUco marker
  - press Enter to list detected objects

### Settings

<!-- TODO: fill in. Key ones in my_constants.py: -->
- `WEBCAM_PATH`: camera index or video file
- `MIRROR_WEBCAM`: `True` for a laptop webcam facing you, `False` for a camera facing outward
- `DANGER_METER`, `ALWAYS_IGNORE`, `IGNORE_OBJECTS`

### ArUco markers

<!-- TODO: printable markers are in assets/aruco_markers (DICT_4X4_50, IDs 0-49) -->

## Known warnings (safe to ignore)

- `objc: Class SDL... is implemented in both ...` on macOS: pygame and OpenCV each ship their own copy of SDL2.
- `xFormers not available`: an optional NVIDIA speed-up used by Depth-Anything. It isn't needed.
- `ByteTrack was deprecated`: supervision is pinned below 0.31 in `requirements.txt`, so it still works.

## Team

<!-- TODO -->

## Credits

Thanks to khw11044 for the tutorial on Depth Anything with a webcam:
https://github.com/khw11044/Depth-Anything-V2-streaming

And marmik_ch19 for the temporary command line fix for the PyTorch error on Mac:
https://www.reddit.com/r/pytorch/comments/1c3kwwg/how_do_i_fix_the_mps_notimplemented_error_for_m1/

Warning sound by foosiemac:
https://freesound.org/people/foosiemac/sounds/110395/
