import os

import cv2

from util import get_categories


def get_filepaths(dir_path):
    filenames = []
    for root, subdirs, files in os.walk(dir_path):
        for f in files:
            filenames.append(os.path.join(root, f))
    return filenames


def load_images(image_dir):
    filenames = get_filepaths(image_dir)
    images = []
    for path in filenames:
        images.append(cv2.imread(path))
    assert len(filenames) == len(images)
    return filenames, get_categories(filenames), images
