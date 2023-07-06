#!/usr/bin/env python
'''
    XaiFactory abstract class is a factory for constructing
    explainable AI (XAI) tools to interpret predictions. 

Author:
    Dean Whitbread
Version: 
    05-07-2023
'''
from abc import ABC, abstractmethod
import misc.wrapper as wrapper
import cv2

class XaiFactory:

    def __init__(self, impath, model):
        '''Construct the XaiFactory abstract class.

        Arguments:
            impath: The directory path to the target image. 
            model: The classifcation model used to
                   classify the target image.
        '''
        self.impath = impath
        self.model = model
        self.target_im = self.__get_image_from_path(self.impath)

    def __get_image_from_path(self, path, size=240):
        '''Return an image from the directory path. 
           The returned image will be ready to be 
           used with the model. 

        Arguments:
            path: The directory path where the image
                  is stored. 
            size: The size of the image for the model.
                  Default is 240 pixels. 
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
        '''Return the directory path of the target
           image.
        '''
        return self.impath

    def get_model(self):
        '''Return the model used to classify the 
           target image.
        '''
        return self.model
    
    def get_target_image(self):
        '''Return the target image being classified.'''
        return self.target_im

    @abstractmethod
    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool
           to interpret the prediction of the 
           target image. 
        '''
        pass
