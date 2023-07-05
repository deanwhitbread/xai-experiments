'''
    The LimeXaiTool concrete class represents
    as explainable AI (XAI) tool used to 
    interpret predictions using LIME. 

Author:
    Dean Whitbread
Version: 
    05-07-2023
'''
from xai.tools.xai_tool import XaiTool
from lime.lime_image import LimeImageExplainer
#import cv2
#import wrapper
import matplotlib.pyplot as plt
import numpy as np

class LimeXaiTool(XaiTool):

    def __init__(self, target_im, model):
        '''Constructor for Lime class.

        Arguments:
            target_im: The target image being classified.
            model: The classifcation model used to
                   classify the target image.
        '''
        self.lime = LimeImageExplainer(random_state=3)
        self.target_im = target_im
        self.expl = self.get_explaination(target_im, model)
    
    def get_explaination(self, target_im, model) -> object:
        '''Return the explaination object of the xai tool.

        Arguments:
            target_im: The target image being classified.
            model: The classifcation model used to
                   classify the target image.
        '''
        return self.lime.explain_instance(target_im, model)

    def show(self):
        '''Display the XAI tool's explaination
        
        Arguments:
            expl: The explaination object of the 
                  xai tool. 
            target_im: The target image being classified.
        '''
        #Select the same class as the top prediction.
        ind =  self.expl.top_labels[0]
        
        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.expl.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(self.expl.segments)
       
        # display target image and heatmap
        fig, ax = plt.subplots(1,2)

        ax[0].imshow(self.target_im)
        ax[1].imshow(heatmap, cmap = 'RdGy', vmin  = -heatmap.max(), vmax = heatmap.max())

        plt.show()
