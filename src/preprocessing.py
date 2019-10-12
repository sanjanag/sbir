
import numpy as np

import cv2

import glob

from matplotlib import pyplot as plt

from feature_extract import feature_extract


image_list = glob.glob('../test/chairs/images/*')
dump = [] # TODO: initialize 

for filepath in image_list:

    curr_img = cv2.imread(filepath,0)  # read in grayscale
    # blurr?

    edges = cv2.Canny(curr_img,16,100) # works pretty well

    features = feature_extract(edges)



    tup = (filepath, features)
    dump.append(tup)



np.save("dump.npy", dump)





