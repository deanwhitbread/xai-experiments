'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        04-07-2023
'''
print('Welcome!\nLoading imports...')

import pandas as pd
from tensorflow.keras.models import load_model
from misc.helpers import get_shortcut_key_str, get_shortcut_key, get_choices, get_dataset_images, get_image_paths
from misc.wrapper import run as predict
from xai.grad_cam_xai_factory import GradCamXaiFactory
from xai.lime_xai_factory import LimeXaiFactory
from xai.shap_xai_factory import ShapXaiFactory

# Constants
XAI_CHOICES = [
           get_shortcut_key_str('LIME', 'l'),
           get_shortcut_key_str('SHAP', 's'), 
           get_shortcut_key_str('Grad-CAM', 'g'),
           get_shortcut_key_str('Next Image', 'n'),
        ]
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Loading model...')
model = load_model(MODEL_PATH)

print('Choosing first image...')
paths = get_image_paths(DATASET_PATH)
index = 0

print('Generating dataset images list...')
images = get_dataset_images(DATASET_PATH)

while True:
    image_path = paths[index]

    print(f'\nModel Prediction: {predict(image_path, model)}')
    print('How do you want to interpret the prediction?')
    
    opt = input(f'Choices: {get_choices(XAI_CHOICES)}:').lower().strip()

    if opt == 'quit' or opt == 'q':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower() or opt == get_shortcut_key(XAI_CHOICES[0]):
        xai = LimeXaiFactory(image_path, model)
    elif opt == XAI_CHOICES[1].lower() or opt == get_shortcut_key(XAI_CHOICES[1]):
        xai = ShapXaiFactory(image_path, model, images)
    elif opt == XAI_CHOICES[2].lower() or opt == get_shortcut_key(XAI_CHOICES[2]):
        xai = GradCamXaiFactory(image_path, model)
    elif opt == XAI_CHOICES[3].lower() or opt == get_shortcut_key(XAI_CHOICES[3]):
        index += 1
        continue;
    else:
        print('Invalid choice. Try again.')
        continue;

    xai.get_xai_tool().show()
