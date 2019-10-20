from skimage import feature
import cv2

def extract_features_sift(grayscale_img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(grayscale_img, None)
    return des