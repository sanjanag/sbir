import cv2
import numpy as np
from math import copysign, log10


def extract_features_hu(image):
	_,im = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
	# Calculate Moments
	moments = cv2.moments(im)
	# Calculate Hu Moments
	huMoments = cv2.HuMoments(moments)
	# Log scale hu moments
	for i in range(0,7):
		huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))
	dsc = huMoments.flatten()
	return dsc
