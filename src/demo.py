import pickle

from load import load_images_from_files
from retrieval import retrieve
from util import read_config, get_result_filenames, display_results

cfg = read_config()

# read sketch
_, _, query_images = load_images_from_files([cfg['querypath']])
query_image = query_images[0]

# get results
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))
k = 10
results = retrieve([query_image], feature_bank, k, 'cityblock')[0]
res_files = get_result_filenames(results, cfg)
_, _, res_images = load_images_from_files(res_files)

# display
display_results(res_images)
