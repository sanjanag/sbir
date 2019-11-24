import cv2
import numpy as np


def calculate_distances_sift(query_features, image_features_list):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    distances = []
    # k = 20
    for image_features in image_features_list:
        matches = bf.match(query_features, image_features)
        if query_features is None:
            return [1000] * len(image_features_list)
        matches = sorted(matches, key=lambda x: x.distance)
        # matches = matches[:k]
        k = len(matches)
        distance_list = [match.distance for match in matches]
        distance = sum(distance_list)
        distance /= k
        distances.append(distance)
    return np.array(distances)
