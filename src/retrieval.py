import pickle as pkl

import numpy as np
from scipy.spatial import distance

from calculate_distances_sift import calculate_distances_sift
from extract_features import extract_features
from preprocess import preprocess_sketches


def compute_distances(image_features, sketch_features, metric='sift_distance'):
    if metric == 'sift_distance':
        distances = []
        for sketch_feature in sketch_features:
            distances.append(
                calculate_distances_sift(sketch_feature, image_features))
        return np.array(distances)
    else:
        distances = distance.cdist(sketch_features, image_features, metric)
        print(f"Distances shape: {distances.shape}")
        return distances


def get_top_results(k, distances):
    sorted_indices = np.argsort(distances)
    topk_sorted_indices  = sorted_indices[:, :k]
    # print(topk_sorted_indices)
    return topk_sorted_indices


def get_image_features(feature_bank_path):
    with open(feature_bank_path, 'rb') as f:
        feature_bank = pkl.load(f)
    image_features = [tup[1] for tup in feature_bank]
    return image_features


def retrieve(queries, feature_name, image_features, k, \
             distance_metric="sift_distance"):
    queries = preprocess_sketches(queries)
    sketch_features_list = []

    for query in queries:
        sketch_features = extract_features(feature_name, query)
        sketch_features_list.append(sketch_features)

    if distance_metric != "sift_distance":
        print(f"Image: {np.array(image_features).shape}")
        print(f"Sketch: {np.array(sketch_features_list).shape}")
        distances = compute_distances(np.array(image_features), np.array(
            sketch_features_list), distance_metric)
    else:
        distances = compute_distances(image_features,sketch_features_list)
    results = get_top_results(k, distances)
    return results


def cascaded_retrieval(queries, ord_features, feature_dict, k_list,
                       metric_dict):
    queries = preprocess_sketches(queries)

    results = []

    for query in queries:
        img_indices = np.arange(len(feature_dict[ord_features[0]]))
        for i, feature in enumerate(ord_features):
            sketch_features = extract_features(feature,query)
            if feature != "sift":
                image_features = np.asarray(feature_dict[feature])
                image_features = image_features[img_indices]
                sketch_features = np.array(sketch_features).reshape(1, -1)
                print(f"Image: {np.array(image_features).shape}")
                print(f"Sketch: {np.array(sketch_features).shape}")
            else:
                image_features = np.asarray(feature_dict[feature])
                image_features =[image_features[idx] for idx in img_indices]
                sketch_features = [sketch_features]
            distances = compute_distances(image_features,
                                          sketch_features,
                                          metric_dict[feature])
            top_results = get_top_results(k_list[i], distances)
            img_indices = img_indices[top_results[0]]
        results.append(img_indices)
    return results
