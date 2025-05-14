import cv2
import numpy as np

def generate_aruco_markers():
    # Create directory to save markers
    import os
    if not os.path.exists('aruco_markers'):
        os.makedirs('aruco_markers')
    
    # Get the ArUco dictionary
    try:
        # Try newer API first
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        # Fall back to older API
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    
    # Generate and save several markers
    marker_size = 200  # pixels
    for id in range(50):  # Generate 5 markers with IDs 0-4
        # Create the marker image
        try:
            # Newer API
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, marker_size)
        except AttributeError:
            # Older API
            marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, id, marker_size, marker_img, 1)
        
        # Add a white border for better detection
        border_size = 20
        with_border = np.ones((marker_size + 2*border_size, marker_size + 2*border_size), dtype=np.uint8) * 255
        with_border[border_size:border_size+marker_size, border_size:border_size+marker_size] = marker_img
        
        # Save the marker
        filename = f'aruco_markers/marker_id_{id}.png'
        cv2.imwrite(filename, with_border)
        print(f"Saved marker ID {id} to {filename}")

if __name__ == "__main__":
    generate_aruco_markers()
