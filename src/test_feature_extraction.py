# from skimage.io import imread
from skimage import feature
# from skimage.color import rgb2gray
from sklearn.metrics import pairwise_distances
import numpy as np
from extract_features import extract_features
from feature_extraction.extract_features_hog import extract_features_hog
import cv2

# def extract_features():
H = []
imgs = ['test/images/10.jpg', 'test/images/12.jpg', 'test/images/13.jpg',
 'test/images/30.jpg', 'test/images/32.jpg' ,'test/images/42.jpg']
for im in imgs:
    img = cv2.imread(im)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H.append(extract_features_hog(grayscale_img))


sketch = cv2.imread('test/sketches/24.png', cv2.IMREAD_UNCHANGED)
grayscale_sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
Hs = extract_features_hog(grayscale_sketch)
# print(sum(Hs))

d = pairwise_distances(np.array(H), Y=np.array([Hs]), metric='l2') #cityblock
# print(d.shape)
print(d)



