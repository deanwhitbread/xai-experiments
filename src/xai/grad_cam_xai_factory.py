'''
    GradCamXaiFactory class is a concrete factory class that 
    constructs the Grad-CAM explainable AI tool. 
'''
__author__ = 'Dean Whitbread'
__version__ = '05-07-2023'

from xai.xai_factory import XaiFactory
from xai.tools.grad_cam_xai_tool import GradCamXaiTool

class GradCamXaiFactory(XaiFactory):

    def __init__(self, impath, model):
        '''Construct the GradCamXaiFactory abstract class.

        Parameters:
        impath: The directory path to the target image.
        model: The classifcation model used to classify the target image.
        '''
        super().__init__(impath, model)

    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool used by the class.'''
        return GradCamXaiTool(
                    self.get_image_path(), 
                    self.get_target_image(), 
                    self.get_model(),
                    self.get_highlight_image()
                )
