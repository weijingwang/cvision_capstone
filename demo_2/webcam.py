import matplotlib
import cv2
from my_constants import *

def webcam_init():
    cmap = matplotlib.colormaps.get_cmap('gray')
    cap = cv2.VideoCapture(WEBCAM_PATH) # webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_RESOLUTION[1])
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, cmap, frame_width, frame_height, frame_rate

def close_webcam(cap):
    cap.release()
    cv2.destroyAllWindows()