'''
    The TumorDetector class identifies brain tumors in a MRI scan. 
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

import cv2
import numpy as np
from analyser.detector.drawer.image_drawer import ImageDrawer

RGB_THRESHOLD = (130,130,130)

class TumorDetector:
    def __init__(self, image):
        '''Construct the TumorDetection object.
        
        Parameters:
        image: The MRI image to detect tumors from. 
        '''
        self.image = image
        self.optimal_coord = self.find_optimal_tumor_coord()

    def find_optimal_tumor_coord(self):
        '''Return a list containing the coordinates and radii of the 
        optimal tumor detected.

        If no tumors are detected, None is returned. 
        '''
        tumor_areas = self.__detect_tumor_areas()
        optimal_coord = None
        if tumor_areas is not None:
            optimal_coord = self.__optimise_detection(tumor_areas)
        
        if optimal_coord == []:
            # incorrectly identified tumors removed from subset.
            return None
        else:
            return optimal_coord

    def highlight_tumor_on_image(self):
        '''Return the original image with the tumors highlighted.
        
        If no tumors are detected, the original image is returned. 
        '''
        if self.optimal_coord is not None:
            image_drawer = ImageDrawer(self.optimal_coord, self.image)
            image_drawer.draw()

        return self.image
    
    def image_has_tumor(self):
        '''Return if the image contains a tumor.'''
        return self.find_optimal_tumor_coord() != None

    def get_tumor_area_coords(self):
        '''Return a list of coordinate that represent each coord of the
        detected region.

        Coords in the list are in the order: 
            [left, right, top, bottom]
        '''
        (x, y, r) = self.find_optimal_tumor_coord()[0]
        top = y+r
        bottom = y-r
        left = x-r
        right = x+r

        coords = [left, right, top, bottom]

        for i in range(0, len(coords)):
            coord = coords[i]
            if coord < 0:
                coords[i] = 0
            elif coord > 240:
                coords[i] = 240

        return coords


    def __get_blurred_image(self):
        '''Return a blurred version of the image.'''
        grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(grayscale_image, (9, 9), 2)
        return blurred_image

    def __detect_tumor_areas(self):
        '''Return a list of circle coordinates and radius of where 
        tumors have been detected in the image.

        If no tumors are detected, None is returned.
        '''
        blurred_image = self.__get_blurred_image()

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
    
    def __optimise_detection(self, areas):
        '''Return a list of optimised circle coordinates and radii.

        Reduction methods are used to reduce the set of detected tumors.

        Parameters:
        areas: A list of circle coordinates and radii.
        '''
        threshold_areas = self.__reduce_using_colour_threshold(areas)
        optimised_areas = self.__reduce_using_pixel_weight(
                    areas=threshold_areas
                )
        return optimised_areas

    '''Reduction methods'''
    def __reduce_using_colour_threshold(self, areas):
        '''Reduce the areas detected as tumors by using the colour
        threshold.

        If no areas are above the colour threashold, an empty list is 
        returned.

        Parameters:
        areas: A list of circle coordinates and radii.
        '''
        optimised_areas = []

        for (x, y, r) in areas:
            pixel_weight = self.__check_pixel_colour(x,y)
            if pixel_weight >= RGB_THRESHOLD:
                region = (x, y, r)
                optimised_areas.append(region)

        return optimised_areas

    def __reduce_using_pixel_weight(self, areas):
        '''Reduce the areas detected as tumors by selecting the areas
        with the whitest pixel color. 

        Parameters:
        areas: A list of circle coordinates and radii.
        '''
        if len(areas) == 0:
            return []

        optimal_areas = [] 
        optimal = areas[0]
        for (x, y, r) in areas:
            (x2, y2, r2) = optimal
            optimal_pixel_weight = self.__check_pixel_colour(x2,y2)
            other_pixel_weight = self.__check_pixel_colour(x,y)

            if other_pixel_weight > optimal_pixel_weight:
                optimal = (x, y, r)

        optimal_areas.append(optimal)

        return optimal_areas

    def __check_pixel_colour(self, x, y):
        '''Return the pixel colour in the format: (R, G, B). 

        Parameters:
        x: The x-coordinate in the image.
        y: The y-coordinate in the image.
        '''
        b, g, r = self.image[y, x]
        return (r, g, b)
