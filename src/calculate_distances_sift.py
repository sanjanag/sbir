import cv2


def calculate_distances_sift(query_features, image_features_list):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    distances = []
    k = 15
    for image_features in image_features_list:
        matches = bf.match(query_features, image_features)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:k]
        distance_list = [match.distance for match in matches]
        distance = sum(distance_list)
        distance /= k
        distances.append(distance)
    return distances
