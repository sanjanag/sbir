import os
import pickle as pkl

import matplotlib.pyplot as plt

from load import load_images_from_files
from retrieval import cascaded_retrieval
from retrieval import retrieve
from util import display_results
from util import get_filenames_from_indices
from util import read_config

cfg = read_config()

query_filename = cfg['querypath']
_, _, queries = load_images_from_files([query_filename])

features = cfg['features']
feature_dict = {}
for feature in features:
    img_features = pkl.load(open(os.path.join(cfg['feature_bank'], feature +
                                              ".pkl"), "rb"))
    feature_dict[feature] = img_features

# k_dict = {'hu_moments': 50, 'hog': 10}

metric_dict = {'hu_moments': 'euclidean', 'hog': 'cityblock',
               'sift': 'sift_distance'}

results1 = cascaded_retrieval(queries, ['hog', 'hu_moments'],
                              feature_dict,
                              [100, 10], metric_dict)

results2 = retrieve(queries, 'hu_moments', feature_dict['hu_moments'], 10,
                    metric_dict['hu_moments'])

results3 = retrieve(queries, 'hog', feature_dict['hog'], 10,
                    metric_dict['hog'])

results4 = retrieve(queries, 'sift', feature_dict['sift'], 10,
                    metric_dict['sift'])

plt.imshow(queries[0])
plt.show()


def display(results, save_file):
    filenames = get_filenames_from_indices(results[0], cfg)
    _, _, images = load_images_from_files(filenames)
    display_results(images, save_file)


display(results1, "cascaded.png")
display(results2, "hu_moments.png")
display(results3, "hog.png")
display(results4, "sift.png")
