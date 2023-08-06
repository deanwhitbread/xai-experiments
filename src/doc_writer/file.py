'''
    The File class represents a new file in the system. 

    The file is saved within the 'results' subdirectory and the extension
    of the filetype (.txt, .csv, etc) must be included in the 'filename'
    when initialising the File object.
'''
__author__='Dean Whitbread'
__version__='06-08-2023'

import os

class File:
    def __init__(self, filename, initial_message=None):
        '''Construct a new File object.
        
        filename: The name of the new file. 
        initial_message: Initialise the file with a message. Default is 
                         None. 
        '''
        new_file = self.__create_file(filename, initial_message)
        self.filename = new_file.name

    def __create_file(self, filename, initial_message):
        '''Create a new file saved inside the 'results' sub-directoy.
        
        Parameters:
        filename: The name of the file. 
        initial_message: Initialise the file with a message. Default is
                         None.
        '''
        os.chdir('results')
        csv_file = open(filename, 'w')

        if initial_message:
            csv_file.write(initial_message)
        
        csv_file.close()
        os.chdir('..')

        return csv_file

    def write(self, message):
        '''Write a new message to the file.

        Parameters:
        message: The message to add to the file.
        '''
        os.chdir('results')

        with open(self.filename, 'r') as file:
            if not len(file.readlines())==0:
                message = '\n' + message

        with open(self.filename, 'a') as file:
            file.write(message)
            file.close()
        os.chdir('..')
