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
from analyser.image_analyser import ImageAnalyser

# Constants
XAI_CHOICES = [
            get_shortcut_key_str('LIME', 'l'),
            get_shortcut_key_str('SHAP', 's'),
            get_shortcut_key_str('Grad-Cam', 'g'),
           ]

CHOICES = [
           get_shortcut_key_str('Explain', 'e'),
           get_shortcut_key_str('Results', 'r'),
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

def __get_all_results():
    paths_index = 0
    precision_score = {'lime':0,'shap':0, 'gradcam':0}
    recall_score = {'lime':0,'shap':0, 'gradcam':0}

    #while paths_index < len(paths):         # 8610 images
    while paths_index < 500:
        image_path = paths[paths_index]
        paths_index += 1
        print(f'Analysing image {paths_index}...', end='\r')

        xai = LimeXaiFactory(image_path, model)
        tool = xai.get_xai_tool()
        analyser = ImageAnalyser(tool)
        p_score = analyser.precision_score()
        r_score = analyser.recall_score()

        precision_score['lime'] = (precision_score['lime'] + p_score) / paths_index
        recall_score['lime'] = (recall_score['lime'] + r_score) / paths_index

        del xai, analyser

        xai = ShapXaiFactory(image_path, model, images)
        tool = xai.get_xai_tool()
        analyser = ImageAnalyser(tool)
        p_score = analyser.precision_score()
        r_score = analyser.recall_score()

        precision_score['shap'] = (precision_score['shap'] + p_score) / paths_index
        recall_score['shap'] = (recall_score['shap'] + r_score) / paths_index

        del xai, analyser

        xai = GradCamXaiFactory(image_path, model)
        tool = xai.get_xai_tool()
        analyser = ImageAnalyser(tool)
        p_score = analyser.precision_score()
        r_score = analyser.recall_score()

        precision_score['gradcam'] = (precision_score['gradcam'] + p_score) / paths_index
        recall_score['gradcam'] = (recall_score['gradcam'] + r_score) / paths_index

        del xai, analyser
    
    output = ("Lime:\n" + 
        (" " * 5) + f"Precision Score: {precision_score['lime']}\n" +
        (" " * 5) + f"Recall Score: {recall_score['lime']}\n" + 
        "Shap:\n" + 
        (" " * 5) + f"Precision Score: {precision_score['shap']}\n" +
        (" " * 5) + f"Recall Score: {recall_score['shap']}\n" +
        "GradCam:\n" + 
        (" " * 5) + f"Precision Score: {precision_score['gradcam']}\n" +
        (" " * 5) + f"Recall Score: {recall_score['gradcam']}\n"
    )

    return output

# testing tools
index = 0
test_cmd = ['r', 'q']

while True:
    image_path = paths[paths_index]

    print(f'\nModel Prediction: {predict(image_path, model)}')
    print('Show explained prediction, analyse all results, or load next image?')
    choice = input(f'Choices: {list_to_str(CHOICES)}: ')

    if is_input_str_this_choice(choice, CHOICES[-1]):
        print('Goodbye')
        break;
    elif is_input_str_this_choice(choice, CHOICES[-2]):
        paths_index += 1
        continue;
    elif is_input_str_this_choice(choice, CHOICES[0]):
        show_all_results = False
    elif is_input_str_this_choice(choice, CHOICES[1]):
        show_all_results = True
    else:
        print('Invalid choice. Try again.')
        continue;

    if show_all_results:
        scores = __get_all_results()
        print(scores)
        continue
    else:
        print('\nWhich XAI tool do you want to use?')
        tool_choice = input(f'Choices: {list_to_str(XAI_CHOICES)}: ')
        
        if is_input_str_this_choice(tool_choice, XAI_CHOICES[0]):
            xai = LimeXaiFactory(image_path, model)
        elif is_input_str_this_choice(tool_choice, XAI_CHOICES[1]):
            xai = ShapXaiFactory(image_path, model, images)
        elif is_input_str_this_choice(tool_choice, XAI_CHOICES[2]):
            xai = GradCamXaiFactory(image_path, model)
        else:
            print('Invalid choice. Heading back to start.')
            continue
        xai.get_xai_tool().show()
