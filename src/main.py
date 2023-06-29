'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        29-06-2023
'''
import pandas as pd
from tensorflow.keras.models import load_model
from helpers import get_next_image_path
from xai.grad_cam_xai import GradCam

XAI_CHOICES = ['LIME', 'SHAP', 'Grad-CAM']
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Welcome!\nLoading dataset and model...')
brats_2018 = pd.read_csv(f'{DATASET_PATH}/survival_data.csv', sep=",")
model = load_model(MODEL_PATH)

print('Choosing image...')
image_path = get_next_image_path(DATASET_PATH)

def get_choices():
    '''Return the available xai interpretor choices. '''
    avail_opts = ""
    for choice in XAI_CHOICES:
        if choice != XAI_CHOICES[-1]:
            avail_opts += choice.lower().strip() + ', '
        else:
            avail_opts += choice.lower().strip()

    return avail_opts

while True:
    print('\nHow do you want to interpret the prediction?')
    
    opt = input(f'Choices: {get_choices()}, quit: ').lower().strip()

    if opt == 'quit':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower():
        # LIME
        pass
    elif opt == XAI_CHOICES[1].lower():
        # SHAP
        pass
    elif opt == XAI_CHOICES[2].lower():
        xai = GradCam(image_path, model)
    else:
        print('Invalid choice. Try again.')
        continue;

    xai.show()
