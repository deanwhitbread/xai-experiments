# Evaluating Explainable AI Tools using Brain MRI Images
## Overview
This data science project aims to objectively assess how well the leading explainable AI (XAI) tools explain model predictions of brain MRI scans. [LIME](https://github.com/marcotcr/lime) (Local Interpretable Model-Agnostic Explanations), [SHAP](https://github.com/shap/shap) (Shapley Additive Explanations), and [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) (Gradient-weighted Class Activation Mapping) are the XAI tools that were tested in this experiment.

A pretrained classification model, [trained to recognise brain tumours in axial MRI scans](https://github.com/MohamedAliHabib/Brain-Tumor-Detection/tree/master), is used to generate the prediction of the input MRI image. The MRI images used in this experiment are from the [BraTS 2018 dataset](https://www.kaggle.com/datasets/sanglequang/brats2018).

## Evaluation Method
An objective evaluation approach is used to measure each tool's effectiveness at explaining the predicted result. Thus, the tool's accuracy, precision, recall, and F1 score is calculated by analysing the image produced by the XAI tool. 

## Setting up the Experiment
Begin by downloading the images used in this experiment from this [OneDrive folder](https://emckclac-my.sharepoint.com/:f:/g/personal/k21014289_kcl_ac_uk/Er5MXLVEmS5AjAV8HAxiGYMBg3_Hiw33zRFHoBYFSRmSOg). (If this link has expired, please follow the instructions in the ['Image Setup Instructions'](https://github.com/deanwhitbread/xai-experiments/README.md#image-setup) section below).

Next, extract the folder and move the extracted folder into the 'dataset' project folder. 

__Note__: The extracted folder name should be 'images_used'. 

Lastly, install the required dependencies.
```
$ pip install -r requirements.txt
```

## Executing the Experiment
In the terminal, navigate to the 'src' project folder. 
```
$ cd [path]/xai-experiments/src
```
There are two ways to execute the experiments:
### 1. With User Interaction
This method generates the XAI tool's evaluation result for the current image.
```
$ python main.py
```
When prompt, choose to explain the result.
```
$ explain        (alternatively, type 'e')
```
Select which XAI tool to use to explain the prediction.
```
$ lime          (alternatively, type 'l')
OR
$ shap          (alternatively, type 's')
OR
$ gradcam       (alternatively, type 'g')
```
After closing the image, you can either explain the current image again using a different tool (follow the previous two steps), or load the next image.
```
$ next          (alternatively, type 'n')
```
To exit the program, use the quit command.
```
$ quit          (alternatively, type 'q')
```
### 2. Without User Interaction
This method automatically evaluates all images used in this experiment for all XAI tools. 

There are 2,793 images in total evaluated in this experiment. On average, one image takes up to 3 minutes to be evaluated by all three XAI tools. Therefore, this choice is best used when running the experiment on a server. 
```
$ python no_ui_main.py
```  
## Results
### Grad-CAM
![Grad-CAM heatmap of the MRI image.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/gc-tumour.png "Grad-CAM Heatmap")

![Evaluation results for Grad-CAM.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/gc-tumour-results.png "Grad-Cam Results")
### LIME
![LIME heatmap of the MRI image.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/lime-tumour.png "LIME Heatmap")

![Evaluation results for LIME.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/lime-tumour-results.png "LIME Results")
### SHAP
![SHAP heatmap of the MRI image.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/shap-tumour.png "SHAP Heatmap")

![Evaluation results for SHAP.](https://github.com/deanwhitbread/xai-experiments/blob/main/results/images/shap-tumour-results.png "SHAP Results")
### Entire Dataset
The 'results' folder in this project contains a CSV file for each XAI tool. The CSV file contains the individual score for all images evaluted in the experiment.
## Conclusion

## Acknowledgements and Remarks
### Dataset
1. The images in the dataset were orignally .nii images but were converted to .jpg images, using the [med2image](https://github.com/FNNDSC/med2image) library.
2. The image selection process involves randomly selecting 10 images from each subfolder within the dataset.
3. The image selection process filters images to a specific range, selecting only axial MRI scans of the brain.
### Evaluation Algorithm
1. An algorithm to locate potential brain tumours in the image was built. This algorithm uses a set of reduction methods to locate a potential tumour region in the image. This region is further analysed to generate the evaluation scores when a potential tumour is present.

## Image Setup Instructions
### Linux/Unix Environment
1. Download the [BraTS 2018 Data Training dataset](https://www.kaggle.com/datasets/sanglequang/brats2018?select=MICCAI_BraTS_2018_Data_Training) and save the extracted folder within the 'dataset' folder of this project.

2. Create a new folder called 'images_used' within the 'dataset' project folder (not inside in the newly extracted folder).
```
$ mkdir images_used
```

3. Next, in the terminal and from within the main project folder, navigate to the 'scripts' project folder and execute the 'convert_to_jpg.sh' script.

   __Note__: This script can take up to 90 minutes to complete. 
```
$ cd scripts
$ bash convert_to_jpg.sh
```
4. Lastly, execute the 'select_jpg_images.sh' script.
```
$ bash select_jpg_images.sh
```

