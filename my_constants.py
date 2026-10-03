"""
CViSion settings.

The sections at the top are the ones you'd normally change.
Distances are in meters.

WARNING: other files use `from my_constants import *`, so don't rename or delete a
setting without searching the code for it first. Changing values is safe.
"""

# =============================================================================
# CAMERA
# =============================================================================
WEBCAM_PATH = 0                 # 0 = default webcam, 1 = next camera, or a video file path, e.g. "person_walk_test_low.mov"
WEBCAM_RESOLUTION = (640, 480)  # (width, height). (1280, 720) is sharper but slower
MIRROR_WEBCAM = True            # True = laptop webcam facing you, False = camera facing outward (flips left/right)


# =============================================================================
# WHICH OBJECTS GET ANNOUNCED
# Names must match YOLO's class names exactly. The full list is MODEL_NAMES at the bottom.
# Each detected object is checked in this order:
#   1. in ALWAYS_IGNORE?   -> never announced, even if dangerous
#   2. dangerous?          -> always announced (see DANGER section below)
#   3. in IGNORE_OBJECTS?  -> not announced
#   4. otherwise           -> announced
# =============================================================================
ALWAYS_IGNORE = ['poo',
     'airplane', 'boat', 'traffic light',
     'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
     'elephant', 'bear', 'zebra', 'giraffe',  'tie',  'frisbee',
     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
     'wine glass', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  'potted plant',
      'mouse', 'remote', 'keyboard',  'microwave', 'oven',
     'toaster', 'sink', 'refrigerator',  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
     'book', 'laptop', 'cell phone',
     'toilet', 'laptop', 'backpack', 'tv', 'bottle'
]
# To test ArUco markers only (ignore every YOLO object), uncomment:
# ALWAYS_IGNORE = MODEL_NAMES

IGNORE_OBJECTS = ['person', 'chair']  # only announced when dangerous (temporary, to reduce noise while testing)


# =============================================================================
# DANGER AND GUIDING
# =============================================================================
DANGER_METER = 1.1    # anything closer than this triggers the danger warning
DANGEROUS_OBJECTS = ['apple', 'car', 'bus', 'train', 'truck', 'bear', 'THINGS SPEEDING AT YOU', 'CLIFF', 'bicycle']
                      # ^ dangerous at any distance. UPPERCASE entries are ideas YOLO can't detect yet.
                      #   Note: entries also in ALWAYS_IGNORE (e.g. 'apple', 'bear') are never announced.
ARRIVAL_METERS = 1.3  # guide mode finishes when you're this close to the target


# =============================================================================
# AUDIO
# Object volume = 1 / (1 + e^(SIG_STEEP * (distance - SIG_MID))), so it gets quieter with distance.
# =============================================================================
SIG_MID = 0     # distance (m) where an object is at half volume
SIG_STEEP = 3   # how quickly the volume drops off with distance
SAMPLE_RATE = 44100                    # Hz
HRTF_DIR = "./assets/HRTF/MIT/diffuse"  # spatial audio data


# =============================================================================
# PERFORMANCE (higher skip = faster, but slower to react)
# =============================================================================
DEPTH_MAP_FRAME_SKIP = 5      # run the depth model every N frames
ARUCO_FRAME_SKIP = 1          # look for ArUco markers every N frames
ARUCO_PERSISTENCE_FRAMES = 5  # keep a marker this many frames after it disappears


# =============================================================================
# PHONE MOCK-UP WINDOW (pygame). Rarely needs changing.
# =============================================================================
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
PYGAME_FPS = 60
SQUARE_SIZE = 200
SQUARE_X = (SCREEN_WIDTH - SQUARE_SIZE) // 2
SQUARE_Y = (SCREEN_HEIGHT - SQUARE_SIZE) // 2
DOUBLE_CLICK_THRESHOLD = 0.25  # seconds
GREEN = (0, 255, 0)
DARK_GREEN = (0, 155, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# =============================================================================
# REFERENCE / NOT USED YET
# =============================================================================
# YOLO (COCO) class names, for copying into the lists above.
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
IMPORTANT_OBJECTS = ['person', 'traffic light', 'stop sign', 'WALL', 'STAIRS', 'STEP', 'OBJECTS IN WAY', 'PATH']  # planned, not used
MAX_SINE_VOLUME = 0.3  # not used anymore, but globals.py imports it
