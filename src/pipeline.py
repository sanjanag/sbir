import yaml

def read_config(filepath):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


# load images
# preprocess
# extract features
# store features