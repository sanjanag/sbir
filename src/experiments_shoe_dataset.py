import pickle

from load import load_images_from_dir
from retrieval import retrieve
from util import calc_precision, calc_recall
from util import get_categories_from_indices
from util import read_config
import matplotlib.pyplot as plt

cfg = read_config()
filenames, sketch_categories, sketches = load_images_from_dir(cfg['sketch_dir'])
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))

klist = [5, 10, 20, 30, 90, 100, 110, 175, 185, 195, 200]
# klist = [10]
avg_precision_values = []
avg_recall_values = []
total_category_images = 200

avg_precision_values_per_category = dict() 
avg_recall_values_per_category = dict()

for category in sketch_categories:
    avg_precision_values_per_category[category] = []
    avg_recall_values_per_category[category] = []

for k in klist:
    print("Performance metrics at k = ", k)
    results = retrieve(sketches, feature_bank, k, 'cityblock')

    ppq = []  # precision per query
    rpq = []
    ppc = dict() # precision per category
    rpc = dict()

    for category in sketch_categories:
        ppc[category] = []
        rpc[category] = []

    total_queries = len(sketches)
    for i, result in enumerate(results):
        distances = [tup[0] for tup in result]
        indices = [tup[1] for tup in result]
        img_categories = get_categories_from_indices(indices, cfg['categories'])
        query_precision = calc_precision(sketch_categories[i], img_categories)
        ppq.append(query_precision)
        query_recall = calc_recall(sketch_categories[i], img_categories, total_category_images)
        rpq.append(query_recall)
        ppc[sketch_categories[i]].append(query_precision)
        rpc[sketch_categories[i]].append(query_recall)

    avg_precision = sum(ppq)/total_queries
    avg_recall = sum(rpq)/total_queries
    avg_precision_values.append(avg_precision)
    avg_recall_values.append(avg_recall)
    print("Average precision: ", avg_precision)
    print("Average recall: ", avg_recall)

    for category in ppc:
        total_queries_per_category = len(ppc[category])
        avg_ppc = sum(ppc[category])/total_queries_per_category
        avg_rpc = sum(rpc[category])/total_queries_per_category
        avg_precision_values_per_category[category].append(avg_ppc)
        avg_recall_values_per_category[category].append(avg_rpc)
        print("Average precision per category-", category, " : ", avg_ppc)
        print("Average recall per category-", category, " : ", avg_rpc)

plt.plot(avg_precision_values, avg_recall_values)
plt.title("Precision Recall curve")
plt.show()

for category in avg_precision_values_per_category:
    precision_values = avg_precision_values_per_category[category]
    recall_values = avg_recall_values_per_category[category]
    plt.plot(precision_values, recall_values)
    title = "Precision Recall curve for category- " + category
    plt.title(title)
    plt.show()
