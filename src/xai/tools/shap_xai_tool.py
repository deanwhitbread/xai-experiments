'''
    The ShapXaiTool concrete class represents
    as explainable AI (XAI) tool used to
    interpret predictions using SHAP.

Author:
    Dean Whitbread
Version:
    05-07-2023
'''
# ignore warning messages when import shap
from warnings import filterwarnings
filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from xai.tools.xai_tool import XaiTool
import shap

class ShapXaiTool(XaiTool):
    def __init__(self, target_im, model, images):
        '''Construct the Shap object.
            
        Arguments:
            target_im: The target image being classified.
            model: The classifcation model used to
                   classify the target image.
            images: A numpy matrix of all the images in the dataset. 
        '''
        self.images = images
        self.exp = self.get_explaination(target_im, model)

    def get_explaination(self, target_im, model) -> object:
        '''Return the explaination object of the xai tool. 

        Arguments:
            target_im: The target image being classified. 
            model: The classifcation model used to 
                   classify the target image. 
        '''
        masker = shap.maskers.Image('blur(240,240)', target_im.shape)
        explainer = shap.Explainer(model, masker)

        return explainer(self.images[-1], max_evals=5000, batch_size=100, outputs=shap.Explanation.argsort.flip[:4])

    def show(self):
        '''Display the XAI tool's explaination.'''
        shap.image_plot(self.exp)
