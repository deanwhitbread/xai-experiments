'''
    helpers.py file contains various helper methods.

    Author:
        Dean Whitbread
    Version:
        29-06-2023
'''
import os
import random as rand

rand.seed(3)

def __get_next_directory_name(path):
    ''' Return the name of a randomly selected directory. 
    
    Arguments:
        path - The path of the current directory where child
               directory folders are located. 
    '''
    dir_list = next(os.walk(path))[1]
    index = rand.randint(0, len(dir_list)-1)
    return dir_list[index]

def __get_image_name(path):
    ''' Return the name of a randomly selected image.

    Arguments:
        path - The path of the current directory where the 
               image file are located. 
    '''
    image_list = os.listdir(path)
    return image_list[rand.randint(0, len(image_list)-1)]

def get_next_image_path(path):
    ''' Return the path where the randomly selected image
        is located.

    Arguments:
        path - The initial path to start the selection.
        
    '''
    while len(next(os.walk(path))[1]):
        next_dir = __get_next_directory_name(path)
        path = f'{path}/{next_dir}'

    return f'{path}/{__get_image_name(path)}'

