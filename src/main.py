'''
    The main script that executes the experiments.

    This script loads the pretrained model, generates
    a list of dataset images to use in the experiement,
    and handles the choices choosen by the user. 
'''
__author__ = 'Dean Whitbread'
__version__ = '11-07-2023'

print('Welcome!\nLoading imports...')

import pandas as pd
from tensorflow.keras.models import load_model
from misc.helpers import (get_shortcut_key_str, list_to_str, 
        get_dataset_images, get_image_paths, is_input_str_this_choice)
from misc.wrapper import run as predict
from xai.grad_cam_xai_factory import GradCamXaiFactory
from xai.lime_xai_factory import LimeXaiFactory
from xai.shap_xai_factory import ShapXaiFactory

# Constants
CHOICES = [
           get_shortcut_key_str('LIME', 'l'),
           get_shortcut_key_str('SHAP', 's'), 
           get_shortcut_key_str('Grad-Cam', 'g'),
           get_shortcut_key_str('Next Image', 'n'),
           get_shortcut_key_str('Quit', 'q')
        ]
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Loading model...')
model = load_model(MODEL_PATH)

print('Choosing first image...')
paths = get_image_paths(DATASET_PATH)
paths_index = 0

print('Generating dataset images list...')
images = get_dataset_images(DATASET_PATH)

while True:
    image_path = paths[paths_index]

    print(f'\nModel Prediction: {predict(image_path, model)}')
    print('How do you want to interpret the prediction?')
    
    input_str = input(f'Choices: {list_to_str(CHOICES)}: ')

    if is_input_str_this_choice(input_str, CHOICES[-1]):
        print('Goodbye')
        break;
    elif is_input_str_this_choice(input_str, CHOICES[-2]):
        paths_index += 1
        continue;
    elif is_input_str_this_choice(input_str, CHOICES[0]):
        xai = LimeXaiFactory(image_path, model)
    elif is_input_str_this_choice(input_str, CHOICES[1]):
        xai = ShapXaiFactory(image_path, model, images)
    elif is_input_str_this_choice(input_str, CHOICES[2]):
        xai = GradCamXaiFactory(image_path, model)
    else:
        print('Invalid choice. Try again.')
        continue;

    xai.get_xai_tool().show()
