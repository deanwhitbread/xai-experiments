'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        26-06-2023
'''
import os
import pandas as pd
import wrapper
from sklearn.model_selection import train_test_split

# Constants
XAI_CHOICES = ['LIME', 'SHAP', 'Grad-CAM']
DATASET_PATH = '../dataset/MICCAI_BraTS_2018_Data_Training'
MODEL_PATH = '../models/cnn-parameters-improvement-23-0.91.model'

print('Welcome!')

# Load the dataset
print('Loading dataset...')
#print(os.listdir(f'{DATASET_PATH}/HGG/Brats18_2013_2_1'))
brats_2018 = pd.read_csv(f'{DATASET_PATH}/survival_data.csv', sep=",")


#url: https://github.com/MohamedAliHabib/Brain-Tumor-Detection/blob/master/Brain%20Tumor%20Detection.ipynb

from tensorflow.keras.models import load_model

img = wrapper.run(
#pred = wrapper.get_prediction(
        f'{DATASET_PATH}/HGG/Brats18_2013_10_1/jpg/output-slice070.jpg',    # test a single image 
        MODEL_PATH 
        #load_model(MODEL_PATH), 240, 240
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

# Train the model
print('Training model...')

# Make a prediction
print('Creating predictions...')

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

    opt = input(f'Choices: {avail_opts}, quit: ').lower().strip()

    ''' Show XAI interpretation or quit '''
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
        # Grad-CAM
        pass
    else:
        print('Invalid choice. Try again.')
