from calculate_precision import calculate_precision
from retrieve import retrieve
import os

k = 10
# calculate precision per category
# shoes: 1 to 200
path = os.path.abspath('../data/shoedataset/sketches/shoes/')
shoe_sketch_paths =  [path + str(i) + '.png' for i in range(1,201)]
image_categories = retrieve(shoe_sketch_paths, k, False)

precisions = [calculate_precision('shoes', image_categories[i]) for i in range(1,201)]
shoe_total_precision = sum(precisions)
print("Average precision at k for shoes: ", shoe_total_precision/200)

# chairs: 1 to 304
path = os.path.abspath('../data/shoedataset/sketches/chairs/')
chair_sketch_paths = [path + str(i) + '.png' for i in range(1,201)]
image_categories = retrieve(chair_sketch_paths, k, False)

precisions = [calculate_precision('chairs', image_categories[i]) for i in range(1,201)]
chair_total_precision = sum(precisions)
print("Average precision at k for chairs: ", chair_total_precision/200)

# calculate average precision
average_precision = (shoe_total_precision + chair_total_precision)/2
print("Average precision of shoe dataset: ",  average_precision) 