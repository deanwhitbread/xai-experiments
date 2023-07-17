'''
    The GradCamXaiTool concrete class represents
    as explainable AI (XAI) tool used to 
    interpret predictions using Grad-CAM. 
'''
__author__ = 'Dean Whitbread'
__version__ = '05-07-2023'

from xai.tools.xai_tool import XaiTool
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import cv2
import misc.wrapper as wrapper
import matplotlib.pyplot as plt
from analyser.image_analyser import ImageAnalyser

class GradCamXaiTool(XaiTool):
    def __init__(self, impath, target_im, model, highlight_im):
        '''Constructor for GradCamXaiTool object. 

        Parameters:
        impath: The directory path to the target image. 
        model: The classifcation model used to classify the target image.
        '''
        self.target_im = target_im
        self.target_layer = self.get_target_layer(model)
        self.heatmap = self.get_heatmap(impath, model)
        self.expl = self.get_explaination(self.target_im, model)[-1]
        self.highlight_im = highlight_im

    def get_target_layer(self, model):
        '''Return the final convolutional layer in the model.

        Parameters:
        model: The classifcation model used to classify the target image. 
        '''
        for layer in reversed(model.layers):
            # check layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply Grad-CAM.")

    def get_heatmap(self, impath, model):
        '''Return the heatmap image for the target image. 
        
        Parameters:
        impath: The directory path to the target image. 
        model: The classifcation model used to classify the target image.
        '''
        im_nparray = wrapper.prepare_image(impath)

        gradModel = Model(
                inputs=[model.inputs],
                outputs=[
                        model.get_layer(self.target_layer).output, 
                        model.output
                    ]
            )
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(im_nparray, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        
        # compute the average of the gradient values
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        (w, h) = (im_nparray.shape[2], im_nparray.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range [0, 1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 1e-8 # eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap
    
    def get_explaination(self, target_im, model) -> object:
        '''Return the explaination object of the xai tool.

        Parameters:
        target_im: The target image being classified.
        model: The classifcation model used to classify the target image.
        '''
        colormap = cv2.COLORMAP_VIRIDIS
        alpha = 0.5

        heatmap = cv2.applyColorMap(self.heatmap, colormap)
        output = cv2.addWeighted(target_im, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

    def show(self):
        '''Display the XAI tool's explaination.'''
        fig, ax = plt.subplots(1, 3)
        
        ax[0].imshow(self.target_im)
        ax[1].imshow(self.expl)
        ax[2].imshow(self.highlight_im)
        
        analyser = ImageAnalyser(self)
        print(analyser.results())

        plt.show()
