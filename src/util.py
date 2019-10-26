import pickle

import yaml
import cv2

def read_config():
    with open("../config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def get_category(image_path):
    if '/' in image_path:
        return image_path.split('/')[-2]
    else:
        return image_path.split('\\')[-2]


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
    true_positives = [category for category in retrieved_categories if category == sketch_category]
    return len(true_positives) / len(retrieved_categories)

def calc_recall(sketch_category, retrieved_categories, total_positives):
    true_positives = [category for category in retrieved_categories if category == sketch_category]
    return len(true_positives) / total_positives

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

def bgr_to_rgb(bgrImage):
    cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)

def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()