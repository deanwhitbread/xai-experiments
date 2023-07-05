'''
    LimeXaiFactory class is a concrete factory class
    that constructs the SHAP explainable AI tool.

Author:
    Dean Whitbread
Version: 
    05-07-2023
'''

from xai.xai_factory import XaiFactory
from xai.tools.lime_xai_tool import LimeXaiTool

class LimeXaiFactory(XaiFactory):

    def __init__(self, impath, model):
        '''Construct the XaiFactory abstract class.

        Arguments:
            impath: The directory path to the target image.
            model: The classifcation model used to
                   classify the target image.
        '''
        super().__init__(impath, model)

    def get_xai_tool(self):
        '''Return the explainable AI (XAI) tool
           to interpret the prediction of the
           target image.
        '''
        return LimeXaiTool(
                    self.get_target_image(), 
                    self.get_model()
                )
