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

def get_shortcut_key_str(word, key):
    '''Return a string highlighting the shortcut key.

    Arguments:
        word The word being highlighted.
        key The shortcut key in the word. 
    '''
    output = ""
    for letter in word:
        if letter.lower() == key:
            output = '(' + letter +')'
        else:
            output += letter
    
    return output

def get_shortcut_key(word):
    '''Return the shortcut key for a word.
    
    Arguments:
        word The word to extract the shortcut key from.
    '''
    for letter in word:
        if letter == '(':
            index = word.find(letter)
            return word[index + 1].lower()

    return ''

def get_choices(choices_array):
    '''Return the string representation of the available
       XAI interpretor options.

    Arguments:
        choices_array The array that contains all the 
                      available choices.
    '''
    avail_opts = ""
    for choice in choices_array:
        avail_opts += choice.lower().strip() + ', '

    avail_opts += get_shortcut_key_str('quit', 'q')

    return avail_opts

