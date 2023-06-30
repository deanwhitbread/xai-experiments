'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        29-06-2023
'''
import pandas as pd
from tensorflow.keras.models import load_model
from helpers import get_next_image_path, get_shortcut_key_str, get_shortcut_key, get_choices
from xai.grad_cam_xai import GradCam
from xai.lime_xai import Lime

# Constants
XAI_CHOICES = [
           get_shortcut_key_str('LIME', 'l'),
           get_shortcut_key_str('SHAP', 's'), 
           get_shortcut_key_str('Grad-CAM', 'g'), 
        ]
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Welcome!\nLoading dataset and model...')
brats_2018 = pd.read_csv(f'{DATASET_PATH}/survival_data.csv', sep=",")
model = load_model(MODEL_PATH)

print('Choosing image...')
image_path = get_next_image_path(DATASET_PATH)

while True:
    print('\nHow do you want to interpret the prediction?')
    
    opt = input(f'Choices: {get_choices(XAI_CHOICES)}:').lower().strip()

    if opt == 'quit' or opt == 'q':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower() or opt == get_shortcut_key(XAI_CHOICES[0]):
        # LIME
        continue;
    elif opt == XAI_CHOICES[1].lower() or opt == get_shortcut_key(XAI_CHOICES[1]):
        # SHAP
        continue;
    elif opt == XAI_CHOICES[2].lower() or opt == get_shortcut_key(XAI_CHOICES[2]):
        xai = GradCam(image_path, model)
    else:
        print('Invalid choice. Try again.')
        continue;

    xai.show()
