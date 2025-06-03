"""My constants"""

# sigmoid volume parameters
SIG_MID = 1.6 # where sigmoid midpoint in meters (play mid volume)
SIG_STEEP = 5 # how steep the sigmoid is

# Add to my_constants.py if not already there
DEPTH_MAP_FRAME_SKIP = 5  # Process depth map every N frames
ARUCO_FRAME_SKIP = 1  # Process ArUco detection every N frames (can be different from depth map skip)
ARUCO_PERSISTENCE_FRAMES = 5  # Number of frames to keep ArUco markers in memory after they disappear



DANGER_METER = 1.4 # when its considered dangerous
ARRIVAL_METERS = 1.6 # in tracking mode, when u arrive.
WEBCAM_RESOLUTION = (640, 480)#(1280, 720)
SAMPLE_RATE = 44100
DURATION = 1  # Short buffer duration for real-time updates
MARGIN_WIDTH = 50
MAX_SINE_VOLUME = 0.3
WEBCAM_PATH = 1#"person_walk_test_low.mov"#0 #'apple_phone_low.mp4'
# 'person_walk_test_HD.mp4'
#COCO dataset. CAPS are what we want but not in dataset
IMPORTANT_OBJECTS = ['person', 'traffic light', 'stop sign', 'WALL', 'STAIRS', 'STEP', 'OBJECTS IN WAY', 'PATH']
DANGEROUS_OBJECTS = ['apple', 'car', 'bus', 'train', 'truck', 'bear', 'THINGS SPEEDING AT YOU', 'CLIFF', 'bicycle']
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

# Define objects to always ignore regardless of danger level
ALWAYS_IGNORE = ['poo',
     'airplane', 'boat', 'traffic light',
     'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
     'elephant', 'bear', 'zebra', 'giraffe',  'tie',  'frisbee',
     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
     'wine glass', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  'potted plant',
      'mouse', 'remote', 'keyboard',  'microwave', 'oven',
     'toaster', 'sink', 'refrigerator',  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
     'book', 'laptop', 'cell phone' # tempo?
]

# ignore unless too close
IGNORE_OBJECTS = [
'person', 'chair',
                  'tv' # temp
                  ]


#pygame demo stuff
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
GREEN = (0, 255, 0)
DARK_GREEN = (0, 155, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PYGAME_FPS = 60
SQUARE_SIZE = 200
SQUARE_X = (SCREEN_WIDTH - SQUARE_SIZE) // 2
SQUARE_Y = (SCREEN_HEIGHT - SQUARE_SIZE) // 2
DOUBLE_CLICK_THRESHOLD = 0.25  # 250 ms for a double click

# Will HRTF stuff
HRTF_DIR = "./HRTF/MIT/diffuse"




# ASCII characters from dark to light
ASCII_CHARS = "@%#*+=-:. "


