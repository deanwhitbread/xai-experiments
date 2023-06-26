'''
    The main file that executes the experiments. 
    
    Author:
        Dean Whitbread
    Version:
        26-06-2023
'''

XAI_CHOICES = ['LIME', 'SHAP', 'Grad-CAM']

print('Welcome!')

''' Load the dataset '''
print('Loading dataset...')

''' Train the model '''
print('Preparing model...')

''' Make a prediction '''
print('Creating predictions...')

''' Interpret the prediction '''
while True:
    print('\nHow do you want to interpret the predictions?')
    avail_opts = ""
    for choice in XAI_CHOICES:
        avail_opts += choice.lower().strip() + ', '

    opt = input(f'Choices: {avail_opts}quit: ').lower().strip()

    if opt == 'quit':
        print('Goodbye')
        break;
    elif opt == XAI_CHOICES[0].lower():
        # LIME
        pass
    elif opt == XAI_CHOICES[1].lower():
        # SHAP
        pass
    elif opt == XAI_CHOICES[2].lower():
        # Grad-CAM
        pass
    else:
        # Ask again. 
        print('Invalid choice. Try again.')
