import pickle as pkl

import cv2
from scipy.spatial import distance

from .extract_features import extract_features


def compute_distances(image_features, sketch_feature, distance_measure):
    distances = distance.cdist(image_features, sketch_feature,
                               distance_measure)
    return distances


def get_topk_images(k, distances, image_bank):
    distances_bank = []
    for i in range(len(image_bank)):
        distances_bank.append((image_bank[i][0], distances[i]))
    ordered_list = sorted(distances_bank, key=lambda x: x[1])
    return [tup[0] for tup in ordered_list[:k]]


def retrieve_images(queries, feature_bank_path, k, display=None):
    with open(feature_bank_path, 'rb') as f:
        feature_bank = pkl.load(f)
    image_features = [tup[1] for tup in feature_bank]
    results = []
    for query in queries:
        sketch = cv2.imread(query, 0)
        sketch_features = extract_features(sketch)
        distances = compute_distances(image_features, sketch_features,
                                      'cityblock')
        sim_imgs = get_topk_images(k, distances, feature_bank)
        results.append(sim_imgs)
    return results
