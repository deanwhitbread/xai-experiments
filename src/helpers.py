'''
    helpers.py file contains various helper methods.

    Author:
        Dean Whitbread
    Version:
        04-07-2023
'''
import os
import random as rand
import wrapper

rand.seed(3)

def __get_next_directory_name(path):
    ''' Return the name of a randomly selected directory. 
    
    Arguments:
        path: The path of the current directory where child
               directory folders are located. 
    '''
    dir_list = next(os.walk(path))[1]
    index = rand.randint(0, len(dir_list)-1)
    return dir_list[index]

def __get_image_name(path):
    ''' Return the name of a randomly selected image.

    Arguments:
        path: The path of the current directory where the 
               image file are located. 
    '''
    image_list = os.listdir(path)
    return image_list[rand.randint(0, len(image_list)-1)]

def get_next_image_path(path):
    ''' Return the path where the randomly selected image
        is located.

    Arguments:
        path: The initial path to start the selection.
        
    '''
    while len(next(os.walk(path))[1]):
        next_dir = __get_next_directory_name(path)
        path = f'{path}/{next_dir}'

    return f'{path}/{__get_image_name(path)}'

def get_shortcut_key_str(word, key):
    '''Return a string highlighting the shortcut key.

    Arguments:
        word: The word being highlighted.
        key: The shortcut key in the word. 
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
        word: The word to extract the shortcut key from.
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
        choices_array: The array that contains all the 
                      available choices.
    '''
    avail_opts = ""
    for choice in choices_array:
        avail_opts += choice.lower().strip() + ', '

    avail_opts += get_shortcut_key_str('quit', 'q')

    return avail_opts


def __get_image_number(image_name):
    '''Return the last digits of the image name. 

    Arguments:
        image_name: The name of the image to extract the digits from.
    '''
    if image_name[-7] != 0:
        return int(image_name[-7] + image_name[-6] + image_name[-5])
    else:
        return int(image_name[-6] + image_name[-5])

def __get_images(path, folder):
    '''Return a list of paths to all the images in the dataset. 

    Arguments:
        path: The path to the dataset parent directory.
        folder: The child folder stored in the first level
                of the parent folder. 
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
    '''Return a numpy matrix of n images in the dataset.

    Arguments:
        path: The path to the dataset parent directory.
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
      
    Arguments: 
        path: The path to the datatset parent directory.

    '''
    paths = []
    for i in range(0, len(os.listdir(path))):
        if os.listdir(path)[i] == 'convert_to_jpg.sh':
            break;
        else:
            paths += __get_images(path, os.listdir(path)[i])
    
    rand.shuffle(paths)

    return paths
