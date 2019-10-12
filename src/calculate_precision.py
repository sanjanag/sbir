def calculate_precision(sketch_category, image_categories):
    return len(image_categories[image_categories == sketch_category])/len(image_categories)