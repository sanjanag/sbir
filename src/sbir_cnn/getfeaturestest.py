import pickle as pkl

import torch
from torch.utils.data import DataLoader

from classnetwork import Net
from dataset import SketchyDataset

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

test_path = "../data/sketchy/sketch/tx_000000000000/"

with open("../label_dict", "rb") as f:
    label_dict = pkl.load(f)

testset = SketchyDataset(test_path, label_dict, "png")
with open("../sketch_paths", "wb") as f:
    pkl.dump(testset.paths,f)
testloader = DataLoader(testset)

model = Net().to(device)
model.load_state_dict(torch.load("../model.pt"))
features = []
with torch.no_grad():
    model.eval()
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        features = model.features(inputs)
        features = features.view(features.shape[0], -1)
        print(features.shape)
        torch.save(features, "../test_features.pt")
        break
