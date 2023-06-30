'''
    lime_xai.py contains the class to 
    interpret predictions using LIME. 

    Author:
        Dean Whitbread
    Version: 
        30-06-2023
'''
from lime.lime_image import LimeImageExplainer
import cv2
import wrapper
import matplotlib.pyplot as plt
import numpy as np

class Lime:

    def __init__(self, impath, model):
        '''Constructor for Lime class.

        Arguments:
            impath: The path to the target image.
            model: The model to classify the target image. 
        '''
        self.lime = LimeImageExplainer(random_state=3)
        self.target_image = wrapper.crop(cv2.imread(impath))
        self.target_image = cv2.resize(self.target_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        self.exp = self.get_explaination(model)

    def get_explaination(self, model):
        '''Return the LIME explaination of the target image.
        
        Arguments:
            model: The model to classify the target image.
        '''
        return self.lime.explain_instance(self.target_image, model)

    def show(self):
        '''Display the target image and heatmap of the LIME interpretation.'''
        #Select the same class as the top prediction.
        ind =  self.exp.top_labels[0]
        
        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(self.exp.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(self.exp.segments)
       
        # display target image and heatmap
        fig, ax = plt.subplots(1,2)

        ax[0].imshow(self.target_image)
        ax[1].imshow(heatmap, cmap = 'RdGy', vmin  = -heatmap.max(), vmax = heatmap.max())

        plt.show()
