import pickle as pkl

import numpy as np
from scipy.spatial import distance

from calculate_distances_sift import calculate_distances_sift
from extract_features import extract_features
from preprocess import preprocess_sketches


def compute_distances(image_features, sketch_feature, metric='sift_distance'):
    if metric == 'sift_distance':
        distances = calculate_distances_sift(sketch_feature, image_features)
        # print(distances)
        return distances
    else:
        # print(image_features.shape, sketch_feature.shape)
        distances = distance.cdist(image_features,
                                   sketch_feature.reshape(1, -1),
                                   metric)
        return np.array(distances.ravel())


def get_top_results(k, distances):
    sorted_indices = np.argsort(distances)
    return list(sorted_indices[:k])


def get_image_features(feature_bank_path):
    with open(feature_bank_path, 'rb') as f:
        feature_bank = pkl.load(f)
    image_features = [tup[1] for tup in feature_bank]
    return image_features


def retrieve(queries, feature_name,  image_features, k, \
                            distance_metric="sift_distance"):
    queries = preprocess_sketches(queries)

    results = []

    for query in queries:
        sketch_features = extract_features(feature_name, query)
        distances = compute_distances(image_features, sketch_features,
                                      distance_metric)
        top_results = get_top_results(k, distances)
        # print(top_results)
        results.append(top_results)
    return results


def cascaded_retrieval(queries, ord_features, feature_dict, k_list,
                       metric_dict):
    queries = preprocess_sketches(queries)

    results = []

    for query in queries:
        img_indices = np.arange(len(feature_dict[ord_features[0]]))
        for i, feature in enumerate(ord_features):
            image_features = np.asarray(feature_dict[feature])
            image_features = image_features[img_indices]
            # print(image_features.shape)
            sketch_features = extract_features(feature, query)
            distances = compute_distances(image_features,
                                          sketch_features,
                                          metric_dict[feature])
            top_results = get_top_results(k_list[i], distances)
            img_indices = img_indices[top_results]
        results.append(img_indices)
    return results




