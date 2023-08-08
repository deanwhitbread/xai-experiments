'''
    ImageSelector class is responsible for storing the choosen 
    images to use in the experiment. 

    Images are choosen using the 'select_jpg_images.sh' script located
    within the scripts folder. 
'''
__author__="Dean Whitbread"
__version__="07-08-2023"

import os
from misc import wrapper
import random as rand

IMAGES_PATH = '../../dataset/images_used' 
rand.seed(13)

class ImageSelector:
    def __init__(self, dataset_path=IMAGES_PATH):
        '''Construct an ImageSelector object.
        
        Parameters:
        dataset_path: The path to the dataset where the images being
                      used in the experiment are stored. 
        '''
        self.dataset_path = dataset_path

    def get_image_paths(self):
        '''Return a list containing the paths to the images.'''
        paths = []
        for image in os.listdir(self.dataset_path):
            paths.append(f'{self.dataset_path}/{image}')
        
        if len(paths):
           rand.shuffle(paths)

        return paths

    def get_dataset_images(self):
        '''Return a list of images formatted according to the model's 
        input data format.'''
        images = self.get_image_paths()
        for i in range(0, len(images)):
            path = images[i]
            image = wrapper.prepare_image(path)
            images[i] = image
        
        return images
