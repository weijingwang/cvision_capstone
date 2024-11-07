import yaml
import json
from ultralytics import YOLO
from torchvision.datasets import CocoDetection
from get_yolo_json import output_yolo_json
from get_yolo_json import find_centers
from get_yolo_json import extract_depth
from output_final import output_final
from PIL import Image
import depth_pro
from get_depth_map import get_depth_map

#get_depth_map('phelps_1210_steps')
file_name = 'bus'

output_yolo_json(file_name)
find_centers(file_name)
extract_depth(file_name)
output_final(file_name)
