#!/usr/bin/env python
import numpy as np
import cv2
import imutils


def crop(image):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    return image[extTop[1] : extBot[1], extLeft[0] : extRight[0]]


def prepare_image(path, x=240, y=240):
    img = cv2.imread(path)
    img = crop(img)
    img = cv2.resize(img, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def get_prediction(path, model, x=240, y=240):
    img = prepare_image(path, x, y)
    return model.predict(img, verbose=0)

def run(path, model):
    predictions = get_prediction(path, model)
    #return np.where(predictions[0] > 0.5, 1, 0)[0]
    return np.where(predictions[0] > 0.5, 'Tumour', 'Benign')[0]
