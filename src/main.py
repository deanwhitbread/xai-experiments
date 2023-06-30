'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        26-06-2023
'''
import pandas as pd
from tensorflow.keras.models import load_model
import wrapper
from sklearn.model_selection import train_test_split
from helpers import get_next_image_path, get_shortcut_key_str, get_shortcut_key

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

print('Generating predictions...')
#image_path = get_next_image_path(DATASET_PATH)
#print(image_path)

pred = wrapper.run(
        #f'{DATASET_PATH}/HGG/Brats18_2013_10_1/jpg/output-slice070.jpg',
        path=get_next_image_path(DATASET_PATH),
        model=model,
    )

#print(pred)

'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

img_1 = wrapper.crop(cv2.imread(f'{DATASET_PATH}/HGG/Brats18_2013_10_1/jpg/output-slice070.jpg'))

plt.figure()
plt.imshow(img_1)
plt.show()
'''

# Interpret the prediction
while True:
    print('\nHow do you want to interpret the predictions?')
    
    ''' Prepare available options '''
    avail_opts = ""
    for choice in XAI_CHOICES:
        if choice != XAI_CHOICES[len(XAI_CHOICES)-1]:
            avail_opts += choice.lower().strip() + ', '
        else:
            avail_opts += choice.lower().strip()
    quit_str = get_shortcut_key_str('quit', 'q')
    opt = input(f'Choices: {avail_opts}, {quit_str}: ').lower().strip()

    ''' Show XAI interpretation or quit '''
    if opt == 'quit' or opt == 'q':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower() or opt == get_shortcut_key(XAI_CHOICES[0]):
        # LIME
        pass
    elif opt == XAI_CHOICES[1].lower() or opt == get_shortcut_key(XAI_CHOICES[1]):
        # SHAP
        pass
    elif opt == XAI_CHOICES[2].lower() or opt == get_shortcut_key(XAI_CHOICES[2]):
        # Grad-CAM
        pass
    else:
        print('Invalid choice. Try again.')
