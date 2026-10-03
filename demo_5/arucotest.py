import cv2
import numpy as np
import sys

def detect_aruco_markers():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Print OpenCV version for debugging
    print(f"OpenCV version: {cv2.__version__}")
    
    # Set up the ArUco dictionary and parameters based on OpenCV version
    opencv_major_ver = int(cv2.__version__.split('.')[0])
    
    if opencv_major_ver >= 4:
        try:
            # Try newer API first (OpenCV 4.7+)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            def detect_func(img):
                return detector.detectMarkers(img)
                
        except AttributeError:
            # Fall back to older API (OpenCV 4.0-4.6)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            
            def detect_func(img):
                return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    else:
        # Very old OpenCV 3.x
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters_create()
        
        def detect_func(img):
            return cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers using the version-appropriate function
        corners, ids, rejected = detect_func(gray)
        
        # If markers are detected
        if ids is not None and len(ids) > 0:
            # Draw the detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw bounding boxes around each marker
            for i in range(len(ids)):
                # Get the corners of the marker
                corner = corners[i][0]
                
                # Convert to integer points
                corner = corner.astype(np.int32)
                
                # Get the bounding box
                x_min = int(min(corner[:, 0]))
                y_min = int(min(corner[:, 1]))
                x_max = int(max(corner[:, 0]))
                y_max = int(max(corner[:, 1]))
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add marker ID text
                cv2.putText(frame, f"ID: {ids[i][0]}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_markers()
