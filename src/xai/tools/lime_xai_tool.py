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

class LimeXaiTool(XaiTool):

    def __init__(self, target_im, model):
        '''Constructor for LimeXaiTool class.

        Parameters:
        target_im: The target image being classified.
        model: The classifcation model used to classify the target image.
        '''
        self.lime = LimeImageExplainer(random_state=3)
        self.target_im = target_im
        self.expl = self.get_explaination(target_im, model)
    
    def get_explaination(self, target_im, model) -> object:
        '''Return the explaination object of the xai tool.

        Parameters:
        target_im: The target image being classified.
        model: The classifcation model used to classify the target image.
        '''
        return self.lime.explain_instance(target_im, model)

    def show(self):
        '''Display the XAI tool's explaination.'''
        label =  self.expl.top_labels[0] 
        image, mask = self.expl.get_image_and_mask(label)

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(self.target_im)
        ax[1].imshow(mark_boundaries(image, mask, color=(255,0,0)))

        plt.show()
