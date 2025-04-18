"""My constants"""



ARRIVAL_METERS = 1
WEBCAM_RESOLUTION = (640, 480)#(1280, 720)
SAMPLE_RATE = 44100
DURATION = 1  # Short buffer duration for real-time updates
MARGIN_WIDTH = 50
MAX_SINE_VOLUME = 0.3
WEBCAM_PATH = 0#"person_walk_test_low.mov"#0 #'apple_phone_low.mp4'
# 'person_walk_test_HD.mp4'
#COCO dataset. CAPS are what we want but not in dataset
IMPORTANT_OBJECTS = ['person', 'traffic light', 'stop sign', 'WALL', 'STAIRS', 'STEP', 'OBJECTS IN WAY', 'PATH']
DANGEROUS_OBJECTS = ['apple', 'car', 'bus', 'train', 'truck', 'bear', 'THINGS SPEEDING AT YOU', 'CLIFF', 'cell phone']
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
IGNORE_OBJECTS = ['airplane', 'scissors', 'tie', 'refrigerator', 'person', 'kite', 'cell phone', 'tv']

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
DOUBLE_CLICK_THRESHOLD = 0.3  # 300 ms for a double click

# Will HRTF stuff
HRTF_DIR = "./HRTF/MIT/diffuse"




# ASCII characters from dark to light
ASCII_CHARS = "@%#*+=-:. "