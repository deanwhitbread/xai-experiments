'''
    PixelRange class represents a range of pixel coordinate in 
    one-dimension. 
'''
__author__ = 'Dean Whitbread'
__version__ = '21-07-2023'

class PixelRange:
    def __init__(self, start, end):
        '''Construct the PixelRange object.

        Parameters:
        start: An integer value representing the start value in the range.
        end: An integer value representing the end value in the range.
        '''
        self.start = start
        self.end = end

    def get_start(self):
        '''Return the start value of the one-dimensional range.'''
        return self.start

    def get_end(self):
        '''Return the end value of the one-dimensional range.'''
        return self.end

    def __str__(self):
        '''Represent the object as a string.'''
        return f'Start Pixel: {self.start}, End Pixel: {self.end}'
