import glob
import os
import cv2


def load_images(image_dir):
    filepaths = []
    for root, subdirs, files in os.walk(image_dir):
        for f in files:
            filepaths.append(os.path.join(root, f))

    image_tuples = []
    for path in filepaths:
        image_tuples.append((path, cv2.imread(path, 0)))
    return image_tuples
