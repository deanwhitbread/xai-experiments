'''
    This file contains methods to detect brain tumors in MRI scans.
'''
__author__ = 'Dean Whitbread'
__version__ = '15-07-2023'

import cv2
import numpy as np

RGB_THRESHOLD = (160,160,160)

def __get_detected_circles(blurred_image):
    '''Return a list of detected circles.

    If no circles are detected, None is returned.

    Parameter:
    blurred_image: The target image with a gaussian blur applied. 
    '''
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

    return circles

def detect_and_highlight_tumor(image):
    '''Return the image with any brain tumors highlighed.
    
    If no tumors are detected, the original image is returned.
    
    Parameters:
    image: The original image to detect brain tumors from.
    '''
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(grayscale_image, (9, 9), 2)

    circles = __get_detected_circles(blurred_image)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if __check_pixel_colour(image,x,y) >= RGB_THRESHOLD:
                # Draw the circle on the original image
                cv2.circle(image, (x, y), r, (255, 0, 0), 4)

    return image

def __check_pixel_colour(image, x, y):
    '''Return the pixel colour in the format: (R, G, B). 

    Parameters:
    image: The image being checked.
    x: The x-coordinate.
    y: The y-coordinate.
    '''
    b, g, r = image[y, x]

    # Print the RGB values
    #print(f"Pixel at ({x}, {y}) has RGB color: ({r}, {g}, {b})")

    return (r, g, b)
