'''
    The ImageAnalyser class analyses images explained by explainable
    AI (XAI) tools and produces a score for precision and recall.
'''
__author__ = 'Dean Whitbread'
__version__ = '16-07-2023'

from analyser.detector.tumor_detector import TumorDetector

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
        print(self.xai_method)       

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
        output = f'\n{header}\nResults\n{header}\n'
        output += f'Precision Score: {self.precision_score()}\n'
        output += f'Recall Score: {self.recall_score()}\n'
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
            tumor_pn_map = self.__get_tumor_positive_negative_map()
        else:
            tumor_pn_map = None

        score_map['tp'] = self.__find_true_positive(pn_map, tumor_pn_map)
        score_map['tn'] = self.__find_true_negative(pn_map, tumor_pn_map)
        score_map['fp'] = self.__find_false_positive(pn_map, tumor_pn_map)
        score_map['fn'] = self.__find_false_negative(pn_map, tumor_pn_map)

        return score_map

    def __find_true_positive(self, pn_map, tumor_pn_map):
        '''Return a numerical value representing the true positive.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in 
                the entire image. 
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image. 
                      If no tumor is present, a value from the map of 
                      the entire image is returned.
        '''
        if tumor_pn_map is not None:
            return tumor_pn_map['positive']
        else:
            return pn_map['positive']

    def __find_true_negative(self, pn_map, tumor_pn_map):
        '''Return a numerical value representing the true negative.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
                      If no tumor is present, a value from the map of
                      the entire image is returned.
        '''

        if tumor_pn_map is not None:
            return tumor_pn_map['negative']
        else:
            return pn_map['negative']

    def __find_false_positive(self, pn_map, tumor_pn_map):
        '''Return a numerical value representing the false positive.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
                      If no tumor is present, a value from the map of
                      the entire image is returned.
        '''

        if tumor_pn_map is not None:
            return pn_map['positive'] - tumor_pn_map['positive']
        else:
            return pn_map['positive']

    def __find_false_negative(self, pn_map, tumor_pn_map):
        '''Return a numerical value representing the false negative.

        Parameters:
        pn_map: A map of the values of positive and negative pixels in
                the entire image.
        tumor_pn_map: A map of the values of positive and negative pixels
                      where the tumor is located on the image.
                      If no tumor is present, a value from the map of
                      the entire image is returned.
        '''
        if tumor_pn_map is not None:
            return pn_map['negative'] - tumor_pn_map['negative']
        else:
            return pn_map['negative']

    def __create_positive_negative_map(self, x_start=0, y_start=0, 
            x_end=240, y_end=240):
        '''Scan the entire image and find all positive and 
        negative pixels.
        
        Parameters:
        x: The maximum x-coordinate of the image. Default is 240.
        y: The maximum y-coordinate of the image. Default is 240.
        '''
        positive_negative_map = {}
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                pixel_colour = self.xai_image[i][j]
                if self.__is_positive_pixel(pixel_colour):
                    try:
                        counter = positive_negative_map['positive']
                        counter += 1
                    except KeyError:
                        counter = 0

                    positive_negative_map['positive'] = counter
                else:
                    try:
                        counter = positive_negative_map['negative']
                        counter += 1
                    except KeyError:
                        counter = 0

                    positive_negative_map['negative'] = counter
        
        return positive_negative_map

    def __get_tumor_positive_negative_map(self):
        '''Return a map with 'positive' and 'negative' keys that hold
           the value of positive and negative pixels in the area where
           the tumor is located. 
        '''
        area_coords = self.td.get_tumor_area_coords()
        tumor_pn_map = self.__create_positive_negative_map(
                    x_start=area_coords[0],
                    y_start=area_coords[3],
                    x_end=area_coords[1],
                    y_end=area_coords[2]
                )
        return tumor_pn_map

    def __is_positive_pixel(self, rgb_value):
        '''Return True if the RGB values is a positive pixel, False 
        otherwise.

        Parameters:
        rgb_value: RGB value to check.
        '''
        (r, g, b) = rgb_value
        if self.__xai_method_is_lime():
            return g > r > b
        else:
            return r > g > b

    def __xai_method_is_lime(self):
        '''Return if the explainable AI method used to explain the
           image was LIME. 
        '''
        return self.xai_method == 'lime'

    def __get_xai_method_name(self, xai_tool):
        '''Return the name of the method used to explain the
           image.

        Parameters:
        xai_tool: The XaiTool object used to explain the image.
        '''
        name = str(xai_tool)
        return name[name.rfind('.')+1:name.find('X')].lower()
