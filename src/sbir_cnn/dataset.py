import glob
import os

import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SketchyDataset(Dataset):

    def __init__(self, sketchy_path, label_dict=None, format="jpg"):
        self.format = format
        self.paths = []
        if label_dict is None:
            categories = self.get_categories(sketchy_path)
            self.label_dict = {category: i for category, i in
                          zip(categories, range(len(categories)))}
        else:
            categories = label_dict.keys()
            self.label_dict = label_dict
        self.images, self.labels = self.read_photos(sketchy_path, categories,
                                                    self.label_dict)
        print("Loaded dataset")
        assert len(self.images) == len(self.labels)
        self.inp_transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def get_categories(self, sketchy_path):
        all_files = os.listdir(sketchy_path)
        categories = []
        for f in all_files:
            if os.path.isdir(os.path.join(sketchy_path, f)):
                categories.append(f)
        return categories

    def read_photos(self, sketchy_path, categories, label_dict):
        print("Reading categories")
        print(len(categories))
        images = []
        labels = []
        for category in categories:
            path = os.path.join(sketchy_path, category, "*."+self.format)
            # print(path)
            filenames = glob.glob(path)
            self.paths = self.paths + filenames
            for filename in filenames:
                images.append(cv2.imread(filename))
                labels.append(label_dict[category])
        #             print(f"Reading category {category} completed")
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.inp_transform(self.images[idx]),
                self.labels[idx])
