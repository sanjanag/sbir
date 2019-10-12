import pickle as pkl

import yaml

from extract_features import extract_features
from load import load_images
from preprocess import preprocess


def read_config():
    with open("../config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


if __name__ == '__main__':
    cfg = read_config()
    image_bank = load_images(cfg['image_dir'])
    print(len(image_bank))
    images = [tup[1] for tup in image_bank]
    processed_imgs = preprocess(images)
    feature_list = []

    # TODO: FIX HOG FEATURE
    for img in processed_imgs:
        feature_list.append(extract_features(processed_imgs))
    feature_bank = []
    for i in range(len(image_bank)):
        feature_bank.append((image_bank[i][0], feature_list[i]))
    with open('feature_bank.pkl', 'wb') as filehandle:
        pkl.dump(feature_bank, filehandle)
