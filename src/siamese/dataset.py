import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, sketch_path, photo_path):
        with open(photo_path, "rb") as f:
            imgs, labels, _ = pickle.load(f)
        self.photo_data = list(zip(imgs, labels))
        with open(sketch_path, "rb") as f:
            imgs, labels, _ = pickle.load(f)
        self.sketch_data = list(zip(imgs, labels))
        self.inp_transform = transforms.Compose(
            [transforms.Resize((100, 100)),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.photo_data)

    def __getitem__(self, idx):

        img0_tuple = random.choice(self.photo_data)
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.sketch_data)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.sketch_data)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        return (
        self.inp_transform(Image.fromarray(img0_tuple[0]).convert('L')),
        self.inp_transform(Image.fromarray(img1_tuple[0]).convert('L')),
        torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[
            1])], dtype=np.float32)),
        img0_tuple[1],
        img1_tuple[1])
