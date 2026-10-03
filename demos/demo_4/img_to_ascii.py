import cv2
from my_constants import ASCII_CHARS








def image_to_ascii(image_path, width=10):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize while maintaining aspect ratio
    aspect_ratio = img.shape[1] / img.shape[0]
    height = int(width / aspect_ratio * 0.55)  # Adjust height scaling
    img = cv2.resize(img, (width, height))
    
    # Normalize pixel values to match ASCII characters
    ascii_image = ""
    num_chars = len(ASCII_CHARS)
    for row in img:
        for pixel in row:
            ascii_image += ASCII_CHARS[min(num_chars - 1, pixel * num_chars // 256)]  # Fix index error
        ascii_image += "\n"
    
    return ascii_image

# Example usage
image_path = "logo6crop.png"  # Change to your image path
ascii_art = image_to_ascii(image_path, width=50)
print(ascii_art)
