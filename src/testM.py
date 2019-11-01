import pickle
from util import read_config

cfg = read_config()
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))

for f in feature_bank:
    print(f.shape)