from feature_extraction.extract_features_hog import extract_features_hog
from feature_extraction.extract_features_sift import extract_features_sift
from feature_extraction.extract_features_hu import extract_features_hu


def extract_features(feature, grayscale_img):
    if feature == "hog":
        return extract_features_hog(grayscale_img)
    elif feature == "sift":
        return extract_features_sift(grayscale_img)
    elif feature == "hu_moments":
        return extract_features_hu(grayscale_img)
    else:
        raise Exception("Feature %s not supported." % feature)
