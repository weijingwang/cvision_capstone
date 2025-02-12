import cv2
from my_constants import *

def is_word_in_set(input_word, word_set):
    return input_word in word_set


def add_performance_text(raw_frame, performance_text):
    # # Get the text size and calculate the background rectangle
    # text_sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0] for line in performance_text]
    # max_width = max(w for w, h in text_sizes)
    # total_height = sum(h for w, h in text_sizes) + len(performance_text) * 10  # Add some padding

    # # Draw the background rectangle
    # rect_x = 0
    # rect_y = 30
    # rect_width = max_width + 20  # Add padding to the width
    # rect_height = total_height + 20  # Add padding to the height
    # cv2.rectangle(raw_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)  # Black rectangle

    # Put text on the image
    for i, line in enumerate(performance_text):
        cv2.putText(raw_frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return raw_frame
