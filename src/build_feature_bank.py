import pickle

from extract_features import extract_features
from load import load_images
from preprocess import preprocess_images
from util import read_config

cfg = read_config()

print("Loading images")
filenames, categories, images = load_images(cfg['image_dir'])
pickle.dump(filenames, open(cfg['filenames'], 'wb'))
pickle.dump(categories, open(cfg['categories'], 'wb'))
print("Num of images loaded: " + str(len(images)))

print("Preprocessing images")
processed_imgs = preprocess_images(images)

print("Building feature bank")
feature_list = []
for img in processed_imgs:
    feature_list.append(extract_features(img))
pickle.dump(feature_list, open(cfg['feature_bank'], 'wb'))
print("Dumped feature bank in " + cfg['feature_bank'])
