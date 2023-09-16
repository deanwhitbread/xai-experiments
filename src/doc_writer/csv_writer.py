'''
    The CsvWriter class holds the File object of each explainable AI 
    (XAI) tool. 

    The class contains a getter method for each XAI tool used in the 
    experiment. 
'''
__author__='Dean Whitbread'
__version__='06-08-2023'

import os
from datetime import datetime
from doc_writer.file import File

CSV_TITLES = 'id,accuracy,precision,recall,f1,tumour_present'

class CsvWriter:
    def __init__(self):
        '''Construct a CsvWriter object.'''
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        self.lime_csv = File(f'lime-{timestamp}.csv', CSV_TITLES)
        self.gradcam_csv = File(f'gradcam-{timestamp}.csv', CSV_TITLES)
        self.shap_csv = File(f'shap-{timestamp}.csv', CSV_TITLES)

    def get_lime_csv_file(self):
        '''Return the csv file for XAI tool.'''
        return self.lime_csv

    def get_gradcam_csv_file(self):
        '''Return the csv file for XAI tool.'''
        return self.gradcam_csv

    def get_shap_csv_file(self):
        '''Return the csv file for XAI tool.'''
        return self.shap_csv
