from feature_extraction.extract_features_hog import extract_features_hog
from feature_extraction.extract_features_sift import extract_features_sift

def extract_features(grayscale_img):
    return extract_features_hog(grayscale_img)
    # return extract_features_sift(grayscale_img)