import pickle

from load import load_images
from retrieval import retrieve
from util import calc_precision
from util import get_categories_from_indices
from util import read_config
from util import read_image, display_image, bgr_to_rgb
import cv2

cfg = read_config()
filenames, image_categories, images = load_images(cfg['image_dir'])
query_image = read_image(cfg['querypath'])
display_image("query sketch", query_image)
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))
k = 10
results = retrieve([query_image], feature_bank, k, 'cityblock')[0]

for res in results:
    images[res[1]]
    display_image(str(res[1]), images[res[1]])
