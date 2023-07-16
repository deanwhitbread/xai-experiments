'''
    The ImageAnalyser class analyses images explained by explainable
    AI (XAI) tools and produces a score for precision and recall.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

import analyse.tumor_detector as td

class ImageAnalyser:
    def __init__(self, original_image, explained_image):
        '''Construct an ImageAnalyser object.

        Parameters:
        original_image: The original image that was explained by the xai.
        explained_image: The image produced by the xai tool. 
        '''
        self.image = original_image
        self.xai_image = explained_image

    def precision_score(self):
        '''Return the precision score of the explained image.'''
        pass

    def recall_score(self):
        '''Return the recall score of the explained image.'''
        pass
    
