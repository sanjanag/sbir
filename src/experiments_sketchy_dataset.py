import pickle

from load import load_images_from_dir
from retrieval import retrieve
from util import calc_precision, calc_recall
from util import get_categories_from_indices
from util import read_config
import matplotlib.pyplot as plt
from inspect import signature

#########################################
# switching between sift and hog
#1. change extract_features method to call corresponding method
#2. for SIFT, pass None as distance_metric parameter to retrieve method.
#3. for HOG, pass 'cityblock' as distance_metric parameter to retrieve method.
#########################################

cfg = read_config()
filenames, sketch_categories, sketches = load_images_from_dir(cfg['sketch_dir'])
print(len(filenames))
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))

# klist = [5, 10, 15, 40, 50, 60, 85, 90, 95]
klist = [5, 10]
avg_precision_values = []
avg_recall_values = []
total_category_images = 100

avg_precision_values_per_category = dict()
avg_recall_values_per_category = dict()

for category in sketch_categories:
    avg_precision_values_per_category[category] = []
    avg_recall_values_per_category[category] = []

best_worst_categories = ['abc', 'xyz'] #best, worst
max_ppc = -1
min_ppc = 100

for k in klist:
    print("Performance metrics at k = ", k)
    results = retrieve(sketches, feature_bank, k, 'cityblock') #'cityblock'

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
        if k == 10:
            if avg_ppc > max_ppc:
                max_ppc = avg_ppc
                best_worst_categories[0] = category
            if avg_ppc < min_ppc:
                min_ppc = avg_ppc
                best_worst_categories[1] = category
            print("Average precision per category-", category, " : ", avg_ppc)
            print("Average recall per category-", category, " : ", avg_rpc)

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

print("avg precision values", avg_precision_values)
print("avg recall values", avg_recall_values)

plt.step([0] + avg_recall_values, [avg_precision_values[0]] + avg_precision_values)
plt.fill_between([0] + avg_recall_values, avg_precision_values + [0], alpha=0.2, **step_kwargs)
plt.title("Precision Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim((0, 1))
plt.show()

for category in best_worst_categories:
    precision_values = avg_precision_values_per_category[category]
    recall_values = avg_recall_values_per_category[category]
    print(category, "avg precision values", precision_values)
    print(category, "avg recall values", recall_values)
    plt.step([0] + recall_values, [precision_values[0]] + precision_values)
    plt.fill_between([0] + recall_values, precision_values + [0], alpha=0.2, **step_kwargs)
    title = "Precision Recall curve for category- " + category
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim((0, 1))
    plt.show()
