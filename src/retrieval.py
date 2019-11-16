import pickle as pkl

import numpy as np
from scipy.spatial import distance

from extract_features import extract_features
from preprocess import preprocess_sketches
from calculate_distances_sift import calculate_distances_sift


def compute_distances(image_features, sketch_features, distance_measure):
    distances = distance.cdist(np.array(sketch_features), np.array(image_features),
                               distance_measure)
    return distances


def get_top_results(k, distances):
    distance_idx_list = []
    for i, distance in enumerate(distances):
        distance_idx_list.append((distance, i))
    distance_idx_list.sort()
    return distance_idx_list[:k]


def get_image_features(feature_bank_path):
    with open(feature_bank_path, 'rb') as f:
        feature_bank = pkl.load(f)
    image_features = [tup[1] for tup in feature_bank]
    return image_features


def retrieve(queries, image_features, k, distance_metric, display=False):
    queries = preprocess_sketches(queries)

    results = []
    sketch_features_list = []

    for query in queries:
        sketch_features = extract_features(query)
        sketch_features_list.append(sketch_features)
    
    if distance_metric is not None:
        distances = compute_distances(image_features, sketch_features_list,
                                      distance_metric)
        for i in range(len(distances)):
            query_distances = distances[i]
            top_results = get_top_results(k, query_distances)
            results.append(top_results)
    else:
        for sketch_features in sketch_features_list:
            distances = calculate_distances_sift(sketch_features, image_features)
            top_results = get_top_results(k, distances)
            results.append(top_results)
    return results

# if __name__ == '__main__':
#     cfg = read_config()
#     categories, sketches = load_images(cfg['sketch_dir'])
