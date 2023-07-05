'''
    The XaiTool interface represents an explainable
    AI (XAI) tool that interprets predictions. 

Author:
    Dean Whitbread
Version:
    05-07-2023
'''
from abc import ABC, abstractmethod

class XaiTool(ABC):
    
    @abstractmethod
    def get_explaination(self, target_im, model) -> object:
        '''Return the explaination object of the xai tool. 

        Arguments:
            target_im: The target image being classified. 
            model: The classifcation model used to 
                   classify the target image. 
        '''
        pass

    @abstractmethod
    def show(self):
        '''Display the XAI tool's explaination.''' 
        pass
