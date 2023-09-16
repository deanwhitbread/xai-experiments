'''
    The no_ui_main script that executes all the experiments without 
    supervision.
'''
__author__ = 'Dean Whitbread'
__version__ = '07-08-2023'

print('Welcome!\nLoading imports...')

from misc.helpers import get_shortcut_key_str, list_to_str, is_this_choice
from experiments.experimental_data import ExperimentalData
from experiments.xai_experiments import XaiExperiment

DATASET_PATH = '../dataset/images_used'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

if __name__=='__main__':
    data = ExperimentalData(DATASET_PATH, MODEL_PATH)
    xai_exp = XaiExperiment(data)
    try:
        xai_exp.run()
    except Exception as e:
        with open('runtime_errors.txt', 'a') as report:
            report.write('\n' + e)
    print('Goodbye.')
