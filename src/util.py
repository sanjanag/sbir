import pickle

import yaml


def read_config():
    with open("../config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def get_category(image_path):
    return image_path.split('/')[-2]


def get_categories(path_list):
    categories = []
    for path in path_list:
        categories.append(get_category(path))
    return categories


def get_categories_from_indices(indices, categories_file):
    all_categories = pickle.load(open(categories_file, "rb"))
    categories = []
    for idx in indices:
        categories.append(all_categories[idx])
    return categories


def calc_precision(sketch_category, retrieved_categories):
    return len(
        retrieved_categories[retrieved_categories == sketch_category]) / len(
        retrieved_categories)
