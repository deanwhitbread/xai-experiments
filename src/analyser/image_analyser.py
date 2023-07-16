'''
    The ImageAnalyser class analyses images explained by explainable
    AI (XAI) tools and produces a score for precision and recall.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

from analyse.detector.tumor_detector import TumorDetector

xai_method = {
            'lime': 1,
            'shap': 2,
            'grad-cam':3,
        }

class ImageAnalyser:
    def __init__(self, original_image, explained_image, xai_method):
        '''Construct an ImageAnalyser object.

        Parameters:
        original_image: The original image that was explained by the xai.
        explained_image: The image produced by the xai tool. 
        xai_method: The method used to explain the image.
        '''
        self.image = original_image
        self.xai_image = explained_image
        self.xai_method = xai_method.strip().lower()
        self.td = TumorDetector(self.image)

    def precision_score(self):
        '''Return the precision score of the explained image.'''
        score_map = self.__analyse_image()
        tp = score_map['tp']
        fp = score_map['fp']

        return tp / (tp + fp)

    def recall_score(self):
        '''Return the recall score of the explained image.'''
        score_map = self.__analyse_image()
        tp = score_map['tp']
        fn = score_map['fn']

        return tp / (tp + fn)

    def results(self):
        '''Display the precision score and recall score.'''
        header = '*' * 10
        output = f'{header}\nResults\n{header}'
        output += f'Precision Score: {self.precision_score}\n'
        output += f'Recall Score: {self.recall_score}\n'
        output += header

        return output

    def __analyse_image(self):
        score_map = {}
        score_map['tp'] = self.__find_true_positive()
        score_map['tn'] = self.__find_true_negative()
        score_map['fp'] = self.__find_false_positive()
        score_map['fn'] = self.__find_false_negative()

        return score_map

    def __find_true_positive(self):
        pass

    def __find_true_negative(self):
        pass

    def __find_false_positive(self):
        pass

    def __find_false_negative(self):
        pass
