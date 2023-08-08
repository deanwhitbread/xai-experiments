'''
    helpers.py file contains various methods that
    assist with the creation of the experiments.
'''
__author__ = 'Dean Whitbread'
__version__ = '07-08-2023'

import os
import misc.wrapper as wrapper

def get_shortcut_key_str(word, key):
    '''Return a string highlighting the shortcut key with brackets.

    Parameters:
    word: The word to highlight.
    key: The shortcut key letter in the word. 
    '''
    output = ""
    
    for letter in word:
        letter = letter.lower()
        if letter == key.lower():
            output = '(' + letter +')'
        else:
            output += letter
    
    return output

def __get_shortcut_key(word):
    '''Return the shortcut key from a word.
    
    Parameters:
    word: The word to extract the shortcut key from.
    '''
    for letter in word:
        if letter == '(':
            letter_index = word.find(letter)
            return word[letter_index + 1].lower()

    return word

def list_to_str(items_list):
    '''Return the names of items in a list as a comma-serparated string.

    Parameters:
    items_list: A list containing items.
    '''
    output_str = ''
    for i in range(0, len(items_list)):
        item = items_list[i].lower().strip()

        if item == items_list[-1]:
            output_str += item
        else:
            output_str += item + ', '

    return output_str

def __strip_choice_str(choice_str):
    '''Strip any open or closed brackets from the choice string. 

    Parameters: 
    choice_str: The string of the choice to be stripped. 
    '''
    return choice_str.lower().replace('(', '').replace(')', '')

def is_this_choice(input_str, choice_str):
    '''Check if the input string matches the choice string. 

    True if they match, False otherwise. 

    Parameters:
        input_str: The input string.
        choice_str: The choice string. 
    '''
    input_str = input_str.lower().strip()
    return (input_str == __strip_choice_str(choice_str) 
            or input_str == __get_shortcut_key(choice_str))
