import pickle as pkl

import torch
from torch.utils.data import DataLoader

from classnetwork import Net
from dataset import SketchyDataset

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

train_path = "../data/sketchy/photo/tx_000000000000/"

with open("../label_dict", "rb") as f:
    label_dict = pkl.load(f)

trainset = SketchyDataset(train_path, label_dict)
with open("../photo_paths", "wb") as f:
    pkl.dump(trainset.paths, f)
trainloader = DataLoader(trainset, batch_size=256)

model = Net().to(device)
model.load_state_dict(torch.load("../model.pt"))
features = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        features = model.features(inputs)
        features = features.view(features.shape[0], -1)
        print(features.shape)
        torch.save(features,f"../train_features{i}.pt")
