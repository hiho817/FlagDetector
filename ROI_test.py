import cv2 as cv
from FlagDetector_class import FlagDetector
# Load your image
image = cv.imread(r'D:\doc\Team_unnamed\worksapce\red_flag_detector\test_image\realworld_test_4.jpg')

# Suppose you have the corners of the contour
# For example: [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
corners = [(100, 200), (400, 180), (420, 500), (80, 520), (50, 300)]

# Initialize FlagDetector with your reference image path
fd = FlagDetector(reference_path=r'D:\doc\Team_unnamed\worksapce\red_flag_detector\reference.png')

# Extract ROI using masking
roi_masked = fd.extract_roi_masking(image, corners)

# Display the masked ROI
cv.imshow("Masked ROI", roi_masked)
cv.waitKey(0)
cv.destroyAllWindows()

# Extract ROI using bounding box (optional)
roi_cropped = fd.extract_roi_bounding_box(image, corners)

# Display the cropped ROI
cv.imshow("Cropped ROI", roi_cropped)
cv.waitKey(0)
cv.destroyAllWindows()
