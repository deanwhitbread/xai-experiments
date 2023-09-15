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
from analyser.detector.tumor_detector import TumorDetector

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
        self.td = TumorDetector(self.__get_image_from_path(self.impath))
        self.highlight_im = self.td.highlight_tumor_on_image()

    def __get_image_from_path(self, path):
        '''Return an image from the directory path. 
           
        The returned image will be ready for use with the model. 

        Parameters:
        path: The directory path where the target image is stored. 
        '''
        image = cv2.imread(path)
        x, y, depth = image.shape

        image = wrapper.crop(
                    image
                )
        image = cv2.resize(
                    image, 
                    dsize=(x, y), 
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
