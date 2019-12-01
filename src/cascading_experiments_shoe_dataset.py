import os
import pickle as pkl
from inspect import signature

import matplotlib.pyplot as plt

from load import load_images_from_dir
from retrieval import retrieve
from util import calc_precision, calc_recall
from util import get_categories_from_indices
from util import read_config
from retrieval import cascaded_retrieval


cfg = read_config()
filenames, sketch_categories, sketches = load_images_from_dir(
    cfg['sketch_dir'])

features = cfg['features']
feature_dict = {}
for feature in features:
    img_features = pkl.load(open(os.path.join(cfg['feature_bank'], feature +
                                              ".pkl"), "rb"))
    feature_dict[feature] = img_features

metric_dict = {'hu_moments': 'euclidean', 'hog': 'cityblock',
               'sift': 'sift_distance'}

results = cascaded_retrieval(sketches, ['sift', 'hu_moments', 'sift', 'hu_moments'],
                              feature_dict,
                              [350, 150, 50, 20], metric_dict)

# klist = [5 , 10, 20, 30]#, 90, 100, 110, 170, 180, 190, 195]
klist = [20]
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
    ppq = []  # precision per query
    rpq = []
    ppc = dict()  # precision per category
    rpc = dict()

    for category in sketch_categories:
        ppc[category] = []
        rpc[category] = []

    total_queries = len(sketches)
    for i in range(len(results)):
        indices = results[i]
        img_categories = get_categories_from_indices(indices,
                                                     cfg['categories'])
        query_precision = calc_precision(sketch_categories[i], img_categories)
        ppq.append(query_precision)
        query_recall = calc_recall(sketch_categories[i], img_categories,
                                   total_category_images)
        rpq.append(query_recall)
        ppc[sketch_categories[i]].append(query_precision)
        rpc[sketch_categories[i]].append(query_recall)

    avg_precision = sum(ppq) / total_queries
    avg_recall = sum(rpq) / total_queries
    avg_precision_values.append(avg_precision)
    avg_recall_values.append(avg_recall)
    print("Average precision: ", avg_precision)
    print("Average recall: ", avg_recall)

    for category in ppc:
        total_queries_per_category = len(ppc[category])
        avg_ppc = sum(ppc[category]) / total_queries_per_category
        avg_rpc = sum(rpc[category]) / total_queries_per_category
        avg_precision_values_per_category[category].append(avg_ppc)
        avg_recall_values_per_category[category].append(avg_rpc)
        print("Average precision per category-", category, " : ", avg_ppc)
        print("Average recall per category-", category, " : ", avg_rpc)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

print("avg precision values", avg_precision_values)
print("avg recall values", avg_recall_values)

plt.step([0] + avg_recall_values,
         [avg_precision_values[0]] + avg_precision_values)
plt.fill_between([0] + avg_recall_values, avg_precision_values + [0],
                 alpha=0.2, **step_kwargs)
plt.title("Precision Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim((0, 1))
plt.show()

for category in avg_precision_values_per_category:
    precision_values = avg_precision_values_per_category[category]
    recall_values = avg_recall_values_per_category[category]
    print(category, "avg precision values", precision_values)
    print(category, "avg recall values", recall_values)
    plt.step([0] + recall_values, [precision_values[0]] + precision_values)
    plt.fill_between([0] + recall_values, precision_values + [0], alpha=0.2,
                     **step_kwargs)
    title = "Precision Recall curve for category- " + category
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim((0, 1))
    plt.show()
