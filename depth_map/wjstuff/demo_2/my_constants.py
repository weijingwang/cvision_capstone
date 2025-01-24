"""My constants"""

SAMPLE_RATE = 44100
DURATION = 1  # Short buffer duration for real-time updates
MARGIN_WIDTH = 50
FONT_SCALE = 1.5  # Adjust this for desired text size

#COCO dataset. CAPS are what we want but not in dataset
IMPORTANT_OBJECTS = ['traffic light', 'stop sign', 'WALL', 'STAIRS', 'STEP', 'OBJECTS IN WAY', 'PATH']
DANGEROUS_OBJECTS = ['bus', 'train', 'truck', 'bear', 'THINGS SPEEDING AT YOU', 'CLIFF']

#pygame demo stuff
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
GREEN = (0, 255, 0)
DARK_GREEN = (0, 155, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PYGAME_FPS = 60

MAX_SINE_VOLUME = 0.3