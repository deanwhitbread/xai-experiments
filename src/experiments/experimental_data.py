'''
    ExperimentalData class represents data that is used in an experiment
    involving explainable AI (XAI) methods.
'''
__author__='Dean Whitbread'
__version__='24-07-2023'

class ExperimentalData:
    def __init__(self, dataset_path, model_path):
        '''Contruct an Experimental Data object.

        Parameters:
        dataset_path: The directory path to the dataset used in the 
                      experiment.
        model_path: The dirtectory path to the saved model used in the 
                    experiment.
        '''
        self.dataset_path = dataset_path
        self.model_path = model_path

    def get_dataset_path(self):
        '''Return the directory path to the dataset.'''
        return self.dataset_path

    def get_model_path(self):
        '''Return the directory path to the model.'''
        return self.model_path

    def __str__(self):
        '''Return the object as a string.'''
        return (f'Data Path: {self.get_dataset_path()}\n' + 
            f'Model Path: {self.get_model_path()}')
