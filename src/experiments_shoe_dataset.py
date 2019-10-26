import pickle

from load import load_images_from_dir
from retrieval import retrieve
from util import calc_precision
from util import get_categories_from_indices
from util import read_config

cfg = read_config()
filenames, sketch_categories, sketches = load_images_from_dir(cfg['sketch_dir'])
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))
k = 10
print(len(sketches))
results = retrieve(sketches, feature_bank, k, 'cityblock')

ppq = []  # precision per query
for i, result in enumerate(results):
    distances = [tup[0] for tup in result]
    indices = [tup[1] for tup in result]
    img_categories = get_categories_from_indices(indices, cfg['categories'])
    ppq.append(calc_precision(sketch_categories[i], img_categories))

# # calculate precision per category
# # shoes: 1 to 200
# path = os.path.abspath('../data/shoedataset/sketches/shoes/')
# shoe_sketch_paths = [path + str(i) + '.png' for i in range(1, 201)]
# image_categories = retrieve(shoe_sketch_paths, k, False)
#
# precisions = [calc_precision('shoes', image_categories[i]) for i in
#               range(1, 201)]
# shoe_total_precision = sum(precisions)
# print("Average precision at k for shoes: ", shoe_total_precision / 200)
#
# # chairs: 1 to 304
# path = os.path.abspath('../data/shoedataset/sketches/chairs/')
# chair_sketch_paths = [path + str(i) + '.png' for i in range(1, 201)]
# image_categories = retrieve(chair_sketch_paths, k, False)
#
# precisions = [calc_precision('chairs', image_categories[i]) for i in
#               range(1, 201)]
# chair_total_precision = sum(precisions)
# print("Average precision at k for chairs: ", chair_total_precision / 200)
#
# # calculate average precision
# average_precision = (shoe_total_precision + chair_total_precision) / 2
# print("Average precision of shoe dataset: ", average_precision)
