import glob
import cv2


def load_images(image_dir):
    files = glob.glob(image_dir + "/*")
    image_tuples = []
    for filepath in files:
        image_tuples.append((filepath, cv2.imread(filepath, 0)))
    return image_tuples
