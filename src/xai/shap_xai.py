'''
    shap_xai.py contains the class to 
    interpret predictions using SHAP XAI. 

    Author:
        Dean Whitbread
    Version: 
        04-07-2023
'''
# ignore warning messages
from warnings import filterwarnings
filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import shap
import wrapper
import cv2

class Shap:
    def __init__(self, impath, model, images):
        '''Construct the Shap object.
            
        Arguments:
            impath: The path to the target image.
            model: The classification model used to classify the image. 
            images: A numpy matrix of all the images in the dataset. 
        '''
        self.target_image = wrapper.crop(cv2.imread(impath))
        self.target_image = cv2.resize(self.target_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        self.images = images
        self.images.append(wrapper.prepare_image(impath))
        self.exp = self.get_explaination(model)

    def get_explaination(self, model):
        '''Return an explaination of the prediction using SHAP. 

        Arguments:
            model: The classification model used to classify the image. 
            images: A numpy matrix of all the images in the dataset. 
        '''
        masker = shap.maskers.Image('blur(240,240)', self.target_image.shape)
        explainer = shap.Explainer(model, masker)

        return explainer(self.images[-1], max_evals=5000, batch_size=100, outputs=shap.Explanation.argsort.flip[:4])

    def show(self):
        '''Display the target image and the XAI interpretation. '''
        shap.image_plot(self.exp)
