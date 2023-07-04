'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        29-06-2023
'''
import pandas as pd
from tensorflow.keras.models import load_model
from helpers import get_next_image_path, get_shortcut_key_str, get_shortcut_key, get_choices, get_dataset_images
from xai.grad_cam_xai import GradCam
from xai.lime_xai import Lime
from xai.shap_xai import Shap
from wrapper import run as predict
from os import listdir

# Constants
XAI_CHOICES = [
           get_shortcut_key_str('LIME', 'l'),
           get_shortcut_key_str('SHAP', 's'), 
           get_shortcut_key_str('Grad-CAM', 'g'), 
        ]
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Welcome!\nLoading model...')
model = load_model(MODEL_PATH)

print('Choosing first image...')
image_path = get_next_image_path(DATASET_PATH)

print('Generating a list of dataset images...')
images = get_dataset_images(DATASET_PATH)

while True:
    print(f'\nModel Prediction: {predict(image_path, model)}')
    print('How do you want to interpret the prediction?')
    
    opt = input(f'Choices: {get_choices(XAI_CHOICES)}:').lower().strip()

    if opt == 'quit' or opt == 'q':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower() or opt == get_shortcut_key(XAI_CHOICES[0]):
        xai = Lime(image_path, model)
    elif opt == XAI_CHOICES[1].lower() or opt == get_shortcut_key(XAI_CHOICES[1]):
        xai= Shap(image_path, model, images)
    elif opt == XAI_CHOICES[2].lower() or opt == get_shortcut_key(XAI_CHOICES[2]):
        xai = GradCam(image_path, model)
    else:
        print('Invalid choice. Try again.')
        continue;

    xai.show()
