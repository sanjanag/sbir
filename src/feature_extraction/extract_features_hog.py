from skimage import feature


def extract_features_hog(grayscale_img):
    histogram = feature.hog(grayscale_img, orientations=9,
                            pixels_per_cell=(64, 64), cells_per_block=(2, 2),
                            transform_sqrt=True, block_norm="L1")
    return histogram
