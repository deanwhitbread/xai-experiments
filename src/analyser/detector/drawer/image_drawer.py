'''
    The ImageDrawer class highlights regions on an image.

    The class uses  circle coordinates to draw circles onto the 
    original image.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

import cv2

class ImageDrawer:
    def __init__(self, regions, image):
        '''Construct an ImageDrawer object.

        Parameters:
        regions: A list of circle coordinates.
        image: The image to draw on.
        '''
        self.regions = regions
        self.image = image

    def draw(self, line_colour=(255,0,0)):
        '''Draws regions of interest onto the image.
        
        Parameters:
        line_colour: A RGB triple that changes the line colour.
        '''
        if self.regions is not None:
            for (x, y, r) in self.regions:
                cv2.circle(self.image, (x,y), r, line_colour, 4)
