import pickle
import cv2

from load import load_images_from_files
from retrieval import retrieve
from util import read_config, get_result_filenames, display_results

#########################################
# switching between sift and hog
#1. change extract_features method to call corresponding method
#2. for SIFT, pass None as distance_metric parameter to retrieve method.
#3. for HOG, pass 'cityblock' as distance_metric parameter to retrieve method.
#########################################


cfg = read_config()

# read sketch
_, _, query_images = load_images_from_files([cfg['querypath']])
query_image = query_images[0]
cv2.imshow("query", query_image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# get results
feature_bank = pickle.load(open(cfg['feature_bank'], "rb"))
k = 10
results = retrieve([query_image], feature_bank, k, 'cityblock')[0] #'cityblock'
res_files = get_result_filenames(results, cfg)
_, _, res_images = load_images_from_files(res_files)

# display
display_results(res_images)
