'''
    ShapXaiFactory class is a concrete factory class
    that constructs the SHAP explainable AI tool. 

Author:
    Dean Whitbread
Version: 
    05-07-2023
'''
from xai.xai_factory import XaiFactory
from xai.tools.shap_xai_tool import ShapXaiTool
import wrapper

class ShapXaiFactory(XaiFactory):

    def __init__(self, impath, model, images):
        '''Construct the XaiFactory abstract class.

        Arguments:
            impath: The directory path to the target image.
            model: The classifcation model used to
                   classify the target image.
            images: A list of images converted to nparray format.
        '''
        super().__init__(impath, model)
        self.images = images
        self.images.append(wrapper.prepare_image(impath))

    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool
           to interpret the prediction of the
           target image.
        '''
        return ShapXaiTool(
                    self.get_target_image(),
                    self.get_model(), 
                    self.images
                )
