'''
    helpers.py file contains various methods that
    assist with the creation of the experiments.

    All methods are publc and the file uses a seeder
    to generate reproducable results.
'''
__author__ = 'Dean Whitbread'
__version__ = '04-07-2023'

import os
import random as rand
import misc.wrapper as wrapper

rand.seed(3)

def get_shortcut_key_str(word, key):
    '''Return a string highlighting the shortcut key.

    Parameters:
    word: The word to highlight.
    key: The shortcut key letter in the word. 
    '''
    output = ""
    for letter in word:
        if letter.lower() == key:
            output = '(' + letter +')'
        else:
            output += letter
    
    return output

def get_shortcut_key(word):
    '''Return the shortcut key from a word.
    
    Parameters:
    word: The word to extract the shortcut key from.
    '''
    for letter in word:
        if letter == '(':
            index = word.find(letter)
            return word[index + 1].lower()

    return word

def get_choices(choices):
    '''Return the available options from the list of choices.

    Parameters:
    choices: The array that contains all the available choices.
    '''
    avail_opts = ""
    for choice in choices:
        avail_opts += choice.lower().strip() + ', '

    avail_opts += get_shortcut_key_str('quit', 'q')

    return avail_opts


def __get_image_number(image_name):
    '''Return the image number. 

    The numbers in the image name are in the
    format [name]-###.jpg, where # represents a digit
    between 0-9.

    Parameters:
    image_name: The path of the image.
    '''
    if image_name[-7] != 0:
        return int(image_name[-7] + image_name[-6] + image_name[-5])
    else:
        return int(image_name[-6] + image_name[-5])

def __get_images(path, folder):
    '''Return a list of paths to all the images in the dataset. 

    Parameters:
    path: The path to the datasets' parent directory.
    folder: The child folder stored in the first level of the parent folder. 
    '''
    cd = f'{path}/{folder}'
    
    new_list = []
    for folder in os.listdir(cd):
        jpg_folder = os.listdir(f'{cd}/{folder}/jpg')
        for image in jpg_folder:
            image_number = __get_image_number(image)

            if image_number >= 70 and image_number <= 110:
                new_list.append(f'{cd}/{folder}/jpg/{image}')
            
    return new_list

def get_dataset_images(path, n=1000):
    '''Return a numpy matrix of n-images in the dataset.

    Parameters:
    path: The path to the datasets' parent directory.
    n: The number of images to extract from the dataset. Default is 1000.
    '''
    dataset_images = (__get_images(path, 'HGG') + __get_images(path, 'LGG'))
    rand.shuffle(dataset_images)
    
    images = []
    for i in range(0, n):
        image_array = wrapper.prepare_image(dataset_images[i])
        images.append(image_array)

    return images

def get_image_paths(path):
    '''Return a list of paths of all images in the dataset. 
      
    Parameters: 
    path: The path to the datatsets' parent directory.
    '''
    paths = []
    for i in range(0, len(os.listdir(path))):
        if os.listdir(path)[i] == 'convert_to_jpg.sh':
            break;
        else:
            paths += __get_images(path, os.listdir(path)[i])
    
    rand.shuffle(paths)

    return paths
