'''
    This file contains methods to detect brain tumors in MRI scans.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

import cv2
import numpy as np

RGB_THRESHOLD = (160,160,160)

def __detect_tumor_areas(blurred_image):
    '''Return a list of circle coordinates where tumors have 
    been detected.

    If no tumors are detected, None is returned.

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
    
    circles = np.round(circles[0, :]).astype("int")
    return circles

def __find_areas_above_colour_threshold(regions, image):
    '''Return a list of areas where the pixel colour of the centre point 
    is above a set colour threashold. 

    If no areas are above the colour threashold, an empty list is 
    returned.

    Parameters:
    regions: A list of circle coordinates that represent tumor regions.
    image: The orginal image. 
    '''
    areas = []

    for (x, y, r) in regions:
        pixel_weight = __check_pixel_colour(image,x,y)
        if pixel_weight >= RGB_THRESHOLD:
            region = (x, y, r)
            areas.append(region)

    return areas

def __find_optimal_areas(areas, image):
    '''Return a list of optimal areas.

    The optimal area is selected based on the whitest pixel centre 
    point colour. 

    Parameters:
    areas: A list of circle coordinates that represent tumor regions.
    image: The original image. 
    '''
    optimal_areas = []

    if len(areas):
        optimal = None
        for (x, y, r) in areas:
            if optimal is None:
                optimal = (x, y, r)
            else:
                (x2, y2, r2) = optimal
                optimal_pixel_weight = __check_pixel_colour(image,x2,y2)
                other_pixel_weight = __check_pixel_colour(image,x,y)

                if other_pixel_weight > optimal_pixel_weight:
                    optimal = (x, y, r)

        optimal_areas.append(optimal)

    return optimal_areas

def __optimise_detection(circles, image):
    '''Return a list of circle coordinates that have been optimised.

    Parameters:
    circles: A list of circle coordinates.
    image: The original image.
    '''
    areas = __find_areas_above_colour_threshold(circles, image)
    optimised_areas = __find_optimal_areas(areas, image)

    return optimised_areas

def detect_and_highlight_tumor(image):
    '''Return the image with any brain tumors highlighed.
    
    If no tumors are detected, the original image is returned.
    
    Parameters:
    image: The original image to detect brain tumors from.
    '''
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(grayscale_image, (9, 9), 2)

    areas = __detect_tumor_areas(blurred_image)
    if areas is not None:
        regions = __optimise_detection(areas, image)
        __draw_regions(regions, image)
    
    return image

def __draw_regions(regions, image):
    if regions is not None:
        for (x, y, r) in regions:
            cv2.circle(image, (x,y), r, (255,0,0), 4)

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
