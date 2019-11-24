import os
import pickle

from extract_features import extract_features
from load import load_images_from_dir
from preprocess import preprocess_images
from util import read_config

cfg = read_config()

print("Loading images")
filenames, categories, images = load_images_from_dir(cfg['image_dir'])
pickle.dump(filenames, open(cfg['filenames'], 'wb'))
pickle.dump(categories, open(cfg['categories'], 'wb'))
print("Num of images loaded: " + str(len(images)))

print("Preprocessing images")
processed_imgs = preprocess_images(images)

print("Building feature bank")

features = cfg['features']

for feature in features:
    print("Extracting feature " + feature)
    feature_list = []
    for img in processed_imgs:
        feature_list.append(extract_features(feature, img))
    filepath = os.path.join(cfg['feature_bank'], feature + ".pkl")
    pickle.dump(feature_list, open(filepath, 'wb'))
    print("Dumped feature " + feature + " in " + filepath)
