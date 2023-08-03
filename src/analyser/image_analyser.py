'''
    The ImageAnalyser class analyses images explained by explainable
    AI (XAI) tools and produces a score for precision and recall.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

from analyser.detector.tumor_detector import TumorDetector
from analyser.pixel_analyser import PixelAnalyser as pixel

class ImageAnalyser:
    def __init__(self, xai_tool):
        '''Construct an ImageAnalyser object.

        Parameters:
        xai_tool: The XaiTool object used to explain the image. 
        '''
        self.image = xai_tool.get_target_image()
        self.xai_image = xai_tool.get_explained_image()
        self.xai_method = self.__get_xai_method_name(xai_tool)
        self.td = TumorDetector(self.image)
        self.score_map = self.__analyse_image()

    def precision_score(self):
        '''Return the precision score of the explained image.'''
        tp = self.score_map['tp']
        fp = self.score_map['fp']
        
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0

    def recall_score(self):
        '''Return the recall score of the explained image.'''
        tp = self.score_map['tp']
        fn = self.score_map['fn']
        
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0
    
    def accuracy_score(self):
        '''Return the accuracy score of the explained_image.'''
        tp = self.score_map['tp']
        tn = self.score_map['tn']
        fp = self.score_map['fp']
        fn = self.score_map['fn']

        return (tp+tn) / (tp+tn+fp+fn)

    def f1_score(self):
        '''Return the F1 score of the explained image.'''
        p_score = self.precision_score()
        r_score = self.recall_score()

        return (2*p_score*r_score) / (p_score+r_score)

    def results(self):
        '''Display the precision score and recall score.'''
        header = '*' * 10
        output = f'\n{header}\nResults\n{header}\n'
        output += f'Accuracy Score: {self.accuracy_score()}\n'
        output += f'Precision Score: {self.precision_score()}\n'
        output += f'Recall Score: {self.recall_score()}\n'
        output += f'F1 Score: {self.f1_score()}\n'
        output += header

        return output

    def __analyse_image(self):
        '''Analyse the image and return a map of scores for true 
        positives (tp), true negatives (tn), false positives (fp),
        and false negatives (fn).

        The key for each scores is the acronym inside the bracket.
        '''
        score_map = {}
        pn_map = self.__create_positive_negative_map()
        
        if self.td.image_has_tumor():
            pixel_ranges = self.td.get_tumor_area_ranges()
            x_range = pixel_ranges[0]
            y_range = pixel_ranges[1]

            tumor_pn_map = self.__create_positive_negative_map(
                    x_range.get_start(),
                    y_range.get_start(),
                    x_range.get_end(),
                    y_range.get_end()
                )
        else:
            tumor_pn_map = None

        score_map['tp'] = self.__find_true_positive(pn_map, tumor_pn_map)
        score_map['tn'] = self.__find_true_negative(pn_map, tumor_pn_map)
        score_map['fp'] = self.__find_false_positive(pn_map, tumor_pn_map)
        score_map['fn'] = self.__find_false_negative(pn_map, tumor_pn_map)
       
        return score_map

    def __find_true_positive(self, pn_map, tumor_pn_map):
        '''Return an integer value representing the true positive.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in 
                the entire image. 
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image. 
        '''
        if tumor_pn_map is not None:
            return tumor_pn_map['total'] - tumor_pn_map['n']
        else:
            return pn_map['p']

    def __find_true_negative(self, pn_map, tumor_pn_map):
        '''Return an integer value representing the true negative.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
        '''
        if tumor_pn_map is not None:
            return tumor_pn_map['total'] - tumor_pn_map['p']
        else:
            return pn_map['p']

    def __find_false_positive(self, pn_map, tumor_pn_map):
        '''Return an integer value representing the false positive.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
                      If no tumor is present, a value from the map of
                      the entire image is returned.
        '''
        if tumor_pn_map is not None:
            return abs(pn_map['p']-(tumor_pn_map['total']-tumor_pn_map['n']))
        else:
            return pn_map['n']

    def __find_false_negative(self, pn_map, tumor_pn_map):
        '''Return a integer value representing the false negative.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
                      If no tumor is present, a value from the map of
                      the entire image is returned.
        '''
        if tumor_pn_map is not None:
            return abs(pn_map['n']-(tumor_pn_map['total']-tumor_pn_map['p']))
        else:
            return pn_map['n']

    def __create_positive_negative_map(self, x_start=0, y_start=0, 
            x_end=None, y_end=None):
        '''Scan the image between start and end ranges, and return a
        map containing the number of positive and negative pixels.

        The keys to the map are the following strings:
            Positive Pixel Key: p
            Negative Pixel Key: n

        x_end and y_end is None by default. When these arguments 
        are None, the shape of the image is used. 
        
        Parameters:
        x: The maximum x-coordinate of the image. Default is None.
        y: The maximum y-coordinate of the image. Default is None.
        '''
        if x_end == None:
            x_end = self.xai_image.shape[0]
        if y_end == None:
            y_end = self.xai_image.shape[1]

        pn_map = {'p': 0, 'n': 0}   # positive-negative map

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                pixel_colour = self.xai_image[y][x]

                if len(pixel_colour) == 4:
                    #convert from RGBA to RGB
                    pixel_colour = pixel_colour[:3]
                
                if pixel.is_same_saturation(pixel_colour):
                    # neither a positive or negative colour
                    continue
                if pixel.is_negative(pixel_colour, self.xai_method):
                    counter = pn_map['n']
                    counter += 1
                    pn_map['n'] = counter
                else:
                    # pixel is positive colour
                    counter = pn_map['p']
                    counter += 1
                    pn_map['p'] = counter

        # store total pixels counted
        pn_map['total'] = (x_end-x_start)*(y_end-y_start)

        return pn_map

    def __get_xai_method_name(self, xai_tool):
        '''Return the name of the method used to explain the
           image.

        Parameters:
        xai_tool: The XaiTool object used to explain the image.
        '''
        name = str(xai_tool)
        return name[name.rfind('.')+1:name.find('X')].lower()
