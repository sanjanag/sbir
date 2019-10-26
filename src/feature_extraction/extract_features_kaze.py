import cv2
import numpy as np


def extract_features_kaze(image):
	vector_size=32
	alg = cv2.KAZE_create()
	kps = alg.detect(image)
	kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
	kps, dsc = alg.compute(image, kps)
	dsc = dsc.flatten()
	needed_size = (vector_size * 64)
	if dsc.size < needed_size:
		dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
	return dsc
