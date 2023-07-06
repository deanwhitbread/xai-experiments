'''
    ShapXaiFactory class is a concrete class that 
    constructs the SHAP explainable AI tool. 
'''
__author__ = 'Dean Whitbread'
__version__ = '05-07-2023'

from xai.xai_factory import XaiFactory
from xai.tools.shap_xai_tool import ShapXaiTool
import misc.wrapper as wrapper

class ShapXaiFactory(XaiFactory):

    def __init__(self, impath, model, images):
        '''Construct the ShapXaiFactory abstract class.

        Parameters:
        impath: The directory path to the target image.
        model: The classifcation model used to classify the target image.
        images: A list of images converted to nparray format.
        '''
        super().__init__(impath, model)
        self.images = images
        self.images.append(wrapper.prepare_image(impath))

    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool used by the class.'''
        return ShapXaiTool(
                    self.get_target_image(),
                    self.get_model(), 
                    self.images
                )
