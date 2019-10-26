# from skimage.io import imread
from skimage import feature
# from skimage.color import rgb2gray
from sklearn.metrics import pairwise_distances
import numpy as np
from extract_features import extract_features
import cv2
from calculate_distances_sift import calculate_distances_sift
from feature_extraction.extract_features_sift import extract_features_sift

H = []

imgs = ['test/images/10.jpg', 'test/images/12.jpg', 'test/images/13.jpg',
 'test/images/30.jpg', 'test/images/32.jpg' ,'test/images/42.jpg']
for im in imgs:
    img = cv2.imread(im)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H.append(extract_features_sift(grayscale_img))
    
for h in H:
    print(h.shape)

sketch = cv2.imread('test/sketches/10.png', cv2.IMREAD_UNCHANGED)
grayscale_sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
Hs = extract_features_sift(grayscale_sketch)
print("query")
print(Hs.shape)
# print(sum(Hs))

# hs compare with h1 to h6
dists = calculate_distances_sift(Hs, H)
print(dists)