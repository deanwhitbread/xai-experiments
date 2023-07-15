'''
    The TumorDetector class contains methods to detect tumors in 
    MRI images.
'''
__author__ = 'Dean Whitbread'
__version__ = '15-07-2023'

import cv2
import numpy as np

#class TumorDetector:
#    def __init__(self):
#        pass

def detect_and_highlight_tumor(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(grayscale_image, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=35,
        maxRadius=60,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            centre_pixel_colour = check_pixel_color(image, x, y)
            threshold = 160
            if centre_pixel_colour >= (threshold, threshold, threshold):
                # Draw the circle on the original image
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)

    return image

def check_pixel_color(image, x, y):
    # Get the color values at the specified pixel (x, y)
    b, g, r = image[y, x]

    # Print the RGB values
    print(f"Pixel at ({x}, {y}) has RGB color: ({r}, {g}, {b})")
    return (r, g, b)
