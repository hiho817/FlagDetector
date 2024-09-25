# Import the FlagDetector class from FlagDetector_class.py
from FlagDetector_class import FlagDetector
import cv2 as cv

# Main execution
if __name__ == "__main__":
    # Paths to the reference and frame images
    reference_path = r'red_flag_detector\reference.png'

    # Define custom HSV ranges (example for red and blue)
    custom_hsv_ranges = [
        {"lower": [0, 120, 70], "upper": [10, 255, 255]},    # Red range 1
        {"lower": [170, 120, 70], "upper": [180, 255, 255]}, # Red range 2
        {"lower": [100, 150, 0], "upper": [140, 255, 255]}   # Blue range
    ]

    safe_colume_width_ratio = 0.5

    # Initialize the detector
    fd = FlagDetector(reference_path, custom_hsv_ranges, safe_colume_width_ratio = 0.5)
    
    # Capture video from the webcam
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break
        # Process the frame
        processed_frame = fd.process_frame(frame)
        
        print(fd.is_flag)
        print(fd.error)
        print(fd.aiming)
        print(fd.detected_num)
        print(fd.is_within_safe_colume)
        print(fd.pixels_per_meter)

        cv.imshow('Detected Shapes', cv.resize(frame, (fd.display_width, fd.display_height)))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the capture and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()