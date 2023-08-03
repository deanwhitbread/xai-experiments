'''
    PixelAnalyser class analyses the RBG values of pixels. 

    The class contains only static methods. 
'''
__author__='Dean Whitbread'
__version__='21-07-2023'

class PixelAnalyser:
    
    @staticmethod
    def is_same_saturation(pixel_colour):
        '''Return if the pixel saturation is the same for the RGB 
        colours.

        Parameters:
        pixel_colour: The RGB value of the pixel.

        Raises: 
        ValueError: When the argument is a RGBA array. 
        '''
        if PixelAnalyser.__is_RGBA_colour(pixel_colour):
            raise ValueError('Parameter must be a RGB tuple not RGBA.')

        (r, g, b) = pixel_colour
        return r==g==b
    
    @staticmethod
    def is_negative(pixel_colour, xai_method):
        '''Return if the pixel represents a negative pixel.

        Parameters:
        pixel_colour: The RGB value of the pixel.

        Raises:
        ValueError: When the argument is a RGBA array.
        '''
        if PixelAnalyser.__is_RGBA_colour(pixel_colour):
            raise ValueError('Parameter must be a RGB tuple not RGBA.')
        
        (r, g, b) = pixel_colour
        
        # TODO: Check that both SHAP and Lime ignore colours with same 
        # saturation. 

        if PixelAnalyser.__is_lime(xai_method):
            return (r>b and r>g)
        else:
            return ((b>r and b>g) or (g>r and g>b and g>140 and b>r))

    @staticmethod
    def __is_RGBA_colour(pixel_colour):
        '''Return if the pixel colour format is RGBA.

        Parameters:
        pixel_colour: The pixel format to test.
        '''
        return len(pixel_colour)==4
    
    @staticmethod
    def __is_lime(xai_tool):
        '''Return if the explainable AI (XAI) method used to explain the
           image was LIME.

        Parameters:
        xai_tool: The XAI tool used to explain the prediction. 
        '''
        return xai_tool=='lime'

