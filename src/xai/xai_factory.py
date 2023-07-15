'''
    XaiFactory is an abstract class that is used as a factory 
    for constructing explainable AI (XAI) tools.

    XAI tools are used to explain predictions.
'''
__author__ = 'Dean Whitbread'
__version__ = '05-07-2023'

from abc import ABC, abstractmethod
import misc.wrapper as wrapper
import cv2
from analyser import tumor_detector as td

class XaiFactory:

    def __init__(self, impath, model):
        '''Construct the XaiFactory abstract class.

        Parameters:
        impath: The directory path to the target image. 
        model: The classifcation model used to classify the target image.
        '''
        self.impath = impath
        self.model = model
        self.target_im = self.__get_image_from_path(self.impath)
        self.highlight_im = td.detect_and_highlight_tumor(self.__get_image_from_path(self.impath))

    def __get_image_from_path(self, path, size=240):
        '''Return an image from the directory path. 
           
        The returned image will be ready for use with the model. 

        Parameters:
        path: The directory path where the target image is stored. 
        size: The size of the image for the model. Default is 240 pixels. 
        '''
        image = wrapper.crop(
                    cv2.imread(path)
                )
        image = cv2.resize(
                    image, 
                    dsize=(size, size), 
                    interpolation=cv2.INTER_CUBIC
                )   
        return image
    
    def get_image_path(self):
        '''Return the directory path of the target image.'''
        return self.impath

    def get_model(self):
        '''Return the model used to classify the target image.'''
        return self.model
    
    def get_target_image(self):
        '''Return the target image being classified.'''
        return self.target_im

    def get_highlight_image(self):
        return self.highlight_im

    @abstractmethod
    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool used by the class.'''
        pass
