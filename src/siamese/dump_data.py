import glob
import os
import pickle as pkl

import cv2


def get_categories(path):
    all_files = os.listdir(path)
    categories = []
    for f in all_files:
        if os.path.isdir(os.path.join(path, f)):
            categories.append(f)
    return categories


def read_photos(data_path, categories, label_dict, format, is_image=True):
    images = []
    labels = []
    img_paths = []
    for category in categories:
        print("Reading photos of cateogory " + category)
        path = os.path.join(data_path, category, "*." + format)
        filenames = glob.glob(path)
        filenames = [filenames[i] for i in range(100)]
        print(path, len(filenames))
        for filename in filenames:
            im = cv2.imread(filename, 0)
            if is_image:
                im = cv2.Canny(im, 80, 200)
            images.append(im)
            labels.append(label_dict[category])
            img_paths.append(filename)
    return images, labels, img_paths


if __name__ == "__main__":
    photo_path = "../data/sketchy/photo/tx_000000000000"
    sketch_path = "../data/sketchy/sketch/tx_000000000000"
    model_path = "../model_files/"
    categories = get_categories(photo_path)
    print(f"Num categories {len(categories)}")

    label_dict = {category: i for i, category in enumerate(categories)}
    with open(os.path.join(model_path, "label_dict.pkl"), "wb") as f:
        pkl.dump(label_dict, f)

    images, labels, filenames = read_photos(photo_path, categories, label_dict,
                                            "jpg")
    print(f"Number of photos {len(images)}")
    assert len(images) == len(labels) == len(filenames)
    with open(os.path.join(model_path, "photo.pkl"), "wb") as f:
        pkl.dump([images, labels, filenames], f)

    images, labels, filenames = read_photos(sketch_path, categories,
                                            label_dict,
                                            "png", False)
    print(f"Number of sketches {len(images)}")
    assert len(images) == len(labels) == len(filenames)
    with open(os.path.join(model_path, "sketch.pkl"), "wb") as f:
        pkl.dump([images, labels, filenames], f)
