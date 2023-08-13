'''
    The main script that executes the experiments.

    This script handles user choices choosen in the terminal window.
'''
__author__ = 'Dean Whitbread'
__version__ = '07-08-2023'

print('Welcome!\nLoading imports...')

from misc.helpers import (
        get_shortcut_key_str, list_to_str, is_this_choice,
        )
from experiments.experimental_data import ExperimentalData
from experiments.xai_experiments import XaiExperiment, XAI_CHOICES

# Constants
CHOICES = [
           get_shortcut_key_str('Explain', 'e'),
           get_shortcut_key_str('Next Image', 'n'),
           get_shortcut_key_str('Quit', 'q')
        ]

DATASET_PATH = '../dataset/images_used'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

if __name__=='__main__':
    data = ExperimentalData(DATASET_PATH, MODEL_PATH)
    xai_exp = XaiExperiment(data)

    while True: 
        image_path = xai_exp.get_current_image_path()

        print(f'\nModel Prediction: {xai_exp.get_model_prediction()}')
        print(('Show explained prediction, analyse all results, or load ' + 
                'next image?'))
        choice = input(f'Choices: {list_to_str(CHOICES)}: ')

        if is_this_choice(choice, CHOICES[-1]):
            print('Goodbye')
            break
        elif is_this_choice(choice, CHOICES[-2]):
            xai_exp.paths_index += 1
        elif is_this_choice(choice, CHOICES[0]):
            print('\nWhich XAI tool do you want to use?')
            tool_choice = input(f'Choices: {list_to_str(XAI_CHOICES)}: ')
            xai_exp.run(tool_choice)
        else:
            print('Invalid choice. Try again.')
