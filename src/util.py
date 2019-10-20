def get_category(image_path):
    return image_path.split('/')[-2]


def get_categories(path_list):
    categories = []
    for path in path_list:
        categories.append(get_category(path))


def calc_precision(sketch_category, retrieved_categories):
    return len(
        retrieved_categories[retrieved_categories == sketch_category]) / len(
        retrieved_categories)
