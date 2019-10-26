import cv2
import numpy as np
from math import copysign, log10




def rgb2gray(rgb):
    return np.uint8(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))


def extract_features_hu(image):



    _,im_th = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    #th, im_th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

    im_th = np.dstack((im_th, im_th , im_th ))


    # connecting line breaks
    kernel = np.ones((8,8), np.uint8)
    d_im = cv2.dilate(im_th, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1) 


    # Copy the thresholded image.
    im_floodfill = e_im.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255); # filled by blue

    # get only the flooded part
    im_out = im_floodfill - e_im.copy()
    th, im_th = cv2.threshold(im_out, 200, 255, cv2.THRESH_BINARY_INV) # invert it 



    grayy = rgb2gray(im_th)  # gray scale

    #print(grayy.shape)
    th, im_th = cv2.threshold(grayy, 200, 255, cv2.THRESH_BINARY) # binarize


    #cv2.imshow("im_th", im_th)
    #cv2.waitKey(0)



    # Calculate Moments
    moments = cv2.moments(im_th)
    #moments = cv2.moments(image)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0,7):
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * (log10(abs( huMoments[i] )))
        

    dsc = huMoments.flatten()
    #print(dsc)
    return dsc



