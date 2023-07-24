'''
    XaiExperiment class is a driver class that runs the explainable AI 
    (XAI) experiments. 
'''
__author__='Dean Whitbread'
__version__='24-07-2023'

from tensorflow.keras.models import load_model
from misc.helpers import (
        get_image_paths, get_dataset_images, is_this_choice,
        get_shortcut_key_str,
        )
from misc.wrapper import run as predict
from xai.grad_cam_xai_factory import GradCamXaiFactory
from xai.lime_xai_factory import LimeXaiFactory
from xai.shap_xai_factory import ShapXaiFactory
from analyser.image_analyser import ImageAnalyser

XAI_CHOICES = [
            get_shortcut_key_str('LIME', 'l'),
            get_shortcut_key_str('SHAP', 's'),
            get_shortcut_key_str('Grad-Cam', 'g'),
           ]

class XaiExperiment:
    def __init__(self, exp_data):
        '''Construct a XaiExperiment object.
        
        Parameters:
        exp_data: The ExperimentalData object used for the experiment.
        '''
        self.model = self.__prepare_model(exp_data.get_model_path())
        self.paths, self.images = self.__prepare_dataset(
                    exp_data.get_dataset_path()
                )
        self.paths_index = 0
    
    def __prepare_model(self, model_path):
        '''Prepare the pretrained model for the experiment.
        
        Parameters:
        model_path: The directory path to where the saved model is 
                    located.
        '''
        print('Loading model...')
        return load_model(model_path)

    def __prepare_dataset(self, dataset_path):
        '''Prepare a list of image paths and a list of images from the
           dataset.

        Parameters:
        dataset_path: The directory path to the parent dataset folder.
        '''
        print('Choosing first image...')
        paths = get_image_paths(dataset_path)

        print('Generating dataset images list...')
        images = get_dataset_images(dataset_path)
        return (paths, images)

    def get_current_image_path(self):
        '''Return the directory path of the current image.'''
        return self.paths[self.paths_index]

    def get_model_prediction(self):
        '''Return the model predicition for the input image.'''
        image_path = self.get_current_image_path()
        return predict(image_path, self.model)

    def run(self, user_cmd=None):
        '''Execute the experiments. 

        If user_cmd is None, the experiements is executed for all XAI 
        methods using the entire dataset. 

        Parameters:
        user_cmd: The xai method to execute for the experiment. Default 
                  is None.
        '''
        if not user_cmd:
            p_score_map, r_score_map = self.__get_all_results()
            self.display_results(p_score_map, r_score_map)
        else:
            image_path = self.get_current_image_path()

            if is_this_choice(user_cmd, XAI_CHOICES[0]):
                xai = LimeXaiFactory(image_path, self.model)
            elif is_this_choice(user_cmd, XAI_CHOICES[1]):
                xai = ShapXaiFactory(image_path, self.model, self.images)
            elif is_this_choice(user_cmd, XAI_CHOICES[2]):
                xai = GradCamXaiFactory(image_path, self.model)
            else:
                print('Invalid choice. Heading back to start.')
                return

            xai.get_xai_tool().show()

    def __get_xai_tools(self, image_path):
        '''Return a list of XaiTool objects for the image.

        Parameters:
        image_path: The directory path to the image being explained by 
                    the XAI tool.
        '''
        xai = []
        xai.append(LimeXaiFactory(image_path, self.model))
        xai.append(ShapXaiFactory(image_path, self.model, self.images))
        xai.append(GradCamXaiFactory(image_path, self.model))
        return xai

    def __get_tool_scores(self, xai):
        '''Return the precision and recall score for the tools.

        Parameters:
        xai: The XaiTool object used to explain the input image.  
        '''
        tool = xai.get_xai_tool()
        analyser = ImageAnalyser(tool)
        p_score = analyser.precision_score()
        r_score = analyser.recall_score()
        tool_name = analyser.xai_method

        return (p_score, r_score, tool_name)

    def __get_all_results(self):
        '''Return the precision score and recall scores for all the XAI
        tools, across the entire dataset.
        '''
        index = 0
        p_score_map = {'lime':0,'shap':0,'gradcam':0} # precision score
        r_score_map = {'lime':0,'shap':0,'gradcam':0} # recall score

        while index < 1000:        # len(paths) == 8610
            image_path = self.paths[index]
            index += 1
            print(f'Analysing image {index}...', end='\r')

            xai_tools = self.__get_xai_tools(image_path)

            for item in xai_tools:
                p_score, r_score, tool_name = self.__get_tool_scores(item)

                p_score_map[tool_name] = (
                        (p_score_map[tool_name] + p_score) / index
                        )
                r_score_map[tool_name] = (
                        (r_score_map[tool_name] + r_score) / index
                        )

            del xai_tools

        return (p_score_map, r_score_map)

    def display_results(self, p_score_map, r_score_map):
        '''Return the overall score for the XAI tool.'''
        output = ""
        for name in p_score_map.keys():
            output += (f"{name.title()}:\n"+(" " * 5)+
                    f"Precision Score: {p_score_map[name]}\n"+(" " * 5)
                    +f"Recall Score: {r_score_map[name]}\n"
            )
        print(output)