'''
    The LimeXaiTool concrete class represents
    as explainable AI (XAI) tool used to 
    interpret predictions using LIME. 
'''
__author__ = 'Dean Whitbread'
__version__ = '05-07-2023'

from xai.tools.xai_tool import XaiTool
from lime.lime_image import LimeImageExplainer
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from analyser.image_analyser import ImageAnalyser

class LimeXaiTool(XaiTool):

    def __init__(self, target_im, model, highlight_im=None):
        '''Constructor for LimeXaiTool class.

        Parameters:
        target_im: The target image being classified.
        model: The classifcation model used to classify the target image.
        '''
        self.lime = LimeImageExplainer(random_state=3)
        self.target_image = target_im
        
        expl_object = self.get_explaination(model)
        self.set_explained_image(image=None, expl_object=expl_object)

        self.highlight_image = highlight_im
    
    def get_explaination(self, model) -> object:
        '''Return the explaination object of the xai tool.

        Parameters:
        target_im: The target image being classified.
        model: The classifcation model used to classify the target image.
        '''
        return self.lime.explain_instance(
                    self.get_target_image(), 
                    model,
                )

    def show(self):
        '''Display the XAI tool's explaination.'''
        if self.highlight_image is not None:
            fig, ax = plt.subplots(1,3)
        else:
            fig, ax = plt.subplots(1,2)

        ax[0].imshow(self.get_target_image())
        ax[1].imshow(self.get_explained_image())
    
        if self.highlight_image is not None:
            ax[2].imshow(self.highlight_image)
        
        analyser = ImageAnalyser(self)
        print(analyser.results())

        plt.show()

    def set_explained_image(self, image, expl_object=None):
        '''Set the image explained by the XAI tool.

        Parameters:
        image: The image explained by the XAI tool.
        expl_object: The explaination object generated by the XAI tool.
                     Default is None.
        '''
        label =  expl_object.top_labels[0]
        image, mask = expl_object.get_image_and_mask(
                    label, positive_only=False
                )

        self.explained_image = mark_boundaries(image, mask)
        
    def get_target_image(self):
        '''Return the target image being explained by the XAI tool.'''
        return self.target_image

    def get_explained_image(self):
        '''Return the image explained by the XAI tool.'''
        return self.explained_image

