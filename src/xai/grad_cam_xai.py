'''
    grad-cam-xai.py contains the class to 
    interpret predictions using Grad-CAM. 

    Author:
        Dean Whitbread
    Version: 
        29-06-2023
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import cv2
import wrapper
import matplotlib.pyplot as plt

class GradCam:
    def __init__(self, impath, model):
        '''Constructor for GradCAMXai object. 

        Arguments:
            impath The path to the target image. 
            model The classification model. 
        '''
        self.image = wrapper.crop(cv2.imread(impath))
        self.image = cv2.resize(self.image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
        self.pred = wrapper.run(impath, model=model)
        self.target_layer = self.get_target_layer(model=model)
        self.heatmap = self.get_heatmap(impath, model)

    def get_target_layer(self, model):
        '''Return the final convolutional layer in the network.

        Arguments:
            model The model to find the target layer from. 
        '''
        for layer in reversed(model.layers):
            # check layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def get_heatmap(self, impath, model):
        '''Return the heatmap for the target image. 
        
        Arguments:
            impath The path to the target image. 
            model The classification model. 
        '''
        # retrieve the image array        
        image_arr = wrapper.prepare_image(impath)

        # construct the Grad-CAM model
        gradModel = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(self.target_layer).output, model.output])
        
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image_arr, tf.float32)
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
        (w, h) = (image_arr.shape[2], image_arr.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range [0, 1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + 1e-8 # eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap
    
    def overlay_heatmap(self, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        '''Overlay the target image with the heatmap.
        
        Arguments:
            image The target image being overlayed. 
        '''
        # apply the supplied color map to the heatmap and overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(self.heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

    def show(self):
        '''Display the target image with the heatmap imposed.'''
        overlay_image = self.overlay_heatmap(self.image)[-1]

        fig, ax = plt.subplots(1, 2)
        
        # display both original image and overlayed image. 
        ax[0].imshow(self.image)
        ax[1].imshow(overlay_image)

        plt.show()
