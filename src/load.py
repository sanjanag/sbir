import os

import cv2

from util import get_categories


def get_filenames(dir_path):
    filenames = []
    for root, subdirs, files in os.walk(dir_path):
        for f in files:
            filenames.append(os.path.join(root, f))
    return filenames


def load_images_from_dir(image_dir):
    filenames = get_filenames(image_dir)
    return load_images_from_files(filenames)


def load_images_from_files(filenames):
    images = []
    for path in filenames:
        images.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    assert len(filenames) == len(images)
    return filenames, get_categories(filenames), images
