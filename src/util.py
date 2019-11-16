import pickle

import cv2
import matplotlib.pyplot as plt
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid


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
    true_positives = [category for category in retrieved_categories if
                      category == sketch_category]
    return len(true_positives) / len(retrieved_categories)


def calc_recall(sketch_category, retrieved_categories, total_positives):
    true_positives = [category for category in retrieved_categories if
                      category == sketch_category]
    return len(true_positives) / total_positives


def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


def bgr_to_rgb(bgrImage):
    cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)


def display_results(images, save_file='result.png'):
    fig = plt.figure(1, (20, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(images) // 5, 5),
                     share_all=True, label_mode="1")
    for i in range(len(images)):
        grid[i].imshow(images[i])
    grid.axes_llc.set_xticks([])
    grid.axes_llc.set_yticks([])
    plt.savefig(save_file)
    plt.show()


def get_result_filenames(query_result, cfg):
    img_indices = [res[1] for res in query_result]
    return get_filenames_from_indices(img_indices, cfg)


def get_filenames_from_indices(img_indices, cfg):
    filenames = pickle.load(open(cfg['filenames'], "rb"))
    res_files = []
    for idx in img_indices:
        res_files.append(filenames[idx])
    return res_files
